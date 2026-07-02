# KV Flat Pool + Dynamic Block Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-layer independent KV allocations with a single flat GPU buffer and serve block IDs to requests from a global free list, while keeping the existing PagedAttention / `reshape_and_cache` tensor contract unchanged.

**Architecture:** Introduce `FlatKVCacheAllocator` and `BlockAllocator`, teach `KVBlockManager` to maintain per-request block tables, add a `compute_slot_mapping` Triton kernel, and update `InputBatchBuilder`, runtime assembly, and the backend lifecycle to allocate/free blocks dynamically per step.

**Tech Stack:** Python 3.12, PyTorch, Triton (via `vllm.triton_utils`), `uv`, pytest.

## Global Constraints

- Python 3.12 only; use `uv run ...` for commands.
- No C++ code; all kernels are Triton.
- Do not import `triton` directly; import through `vllm.triton_utils`.
- Do not read `FASTINFERENCE_*` environment variables in runtime code; configuration flows through `inf_config` / `attn_metadata["config"]`.
- Every new Triton kernel must include ASCII comments describing memory layout and thread/block tiling.
- Follow existing typing style and run `uv run ruff check . && uv run ruff format .` before each commit.
- Verify with `bash tests/run_regression_suite.sh` and `SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh`.

---

## File Structure

| File | Responsibility |
|---|---|
| `docs/superpowers/specs/2026-07-02-kv-flat-pool-dynamic-blocks-design.md` | Approved design spec (read-only reference). |
| `vllm/engine/initialization/flat_kv_cache_allocator.py` | New: allocate one flat KV buffer + scale buffer, return per-layer views. |
| `vllm/engine/block_allocator.py` | New: global block ID free list with null-block reservation. |
| `vllm/kernels/triton/compute_slot_mapping.py` | New: Triton kernel mapping `(position, block_table)` -> `slot_mapping`. |
| `vllm/engine/kv_block_manager.py` | Modify: own `BlockAllocator`, maintain per-request block IDs and a shared device block-table buffer. |
| `vllm/engine/input_batch_builder.py` | Modify: build `slot_mapping` via the new kernel; use dynamic block-table rows. |
| `vllm/engine/lite_engine.py` | Modify: construct new allocator / block allocator; remove static `_fast_block_tables`. |
| `vllm/engine/initialization/runtime_component_factory.py` | Modify: pass `block_allocator` through assembly. |
| `vllm/engine/runtime_factory.py` | Modify: construct `KVBlockManager` with allocator; update context dataclass. |
| `vllm/engine/backend/lite_single_gpu.py` | Modify: prefix-cache store/materialize use `request_id`; free blocks on request end. |
| `vllm/engine/runtime_controller.py` | Modify: call `ensure_blocks_for_requests` before each step. |
| `tests/test_flat_kv_cache_allocator.py` | New: unit tests for flat allocator views and scales. |
| `tests/test_block_allocator.py` | New: unit tests for block free list. |
| `tests/test_compute_slot_mapping.py` | New: Triton kernel reference tests. |
| `tests/test_kv_block_manager.py` | New: dynamic block allocation + prefix-cache capture/materialize tests. |
| `tests/test_input_batch_builder_dynamic.py` | New: slot-mapping construction with dynamic block tables. |

---

## Task 1: Branch and Minimal Scaffolding

**Files:**
- Modify: `.git/HEAD` (via shell)

**Interfaces:**
- Produces: feature branch `feat/kv-flat-pool`

- [ ] **Step 1: Create and check out the feature branch**

```bash
git checkout -b feat/kv-flat-pool
```

- [ ] **Step 2: Create empty new files**

```bash
touch vllm/engine/initialization/flat_kv_cache_allocator.py
touch vllm/engine/block_allocator.py
touch vllm/kernels/triton/compute_slot_mapping.py
touch tests/test_flat_kv_cache_allocator.py
touch tests/test_block_allocator.py
touch tests/test_compute_slot_mapping.py
touch tests/test_kv_block_manager.py
touch tests/test_input_batch_builder_dynamic.py
```

- [ ] **Step 3: Verify repo status**

```bash
git status
```

Expected: new branch with 8 untracked files.

- [ ] **Step 4: Commit scaffolding**

```bash
git add -N vllm/engine/initialization/flat_kv_cache_allocator.py \
  vllm/engine/block_allocator.py \
  vllm/kernels/triton/compute_slot_mapping.py \
  tests/test_flat_kv_cache_allocator.py \
  tests/test_block_allocator.py \
  tests/test_compute_slot_mapping.py \
  tests/test_kv_block_manager.py \
  tests/test_input_batch_builder_dynamic.py
git commit -m "chore(kv-flat-pool): scaffold new allocator, block allocator and tests"
```

---

## Task 2: `FlatKVCacheAllocator`

**Files:**
- Create: `vllm/engine/initialization/flat_kv_cache_allocator.py`
- Modify: `vllm/engine/initialization/__init__.py`
- Test: `tests/test_flat_kv_cache_allocator.py`

**Interfaces:**
- Produces: `FlatKVCacheAllocator.allocate(...)` returning `(kv_caches, kv_scale_caches, num_total_blocks)`.
- Consumes: `layer_kv_cache_shape_for_layer`, `compute_kv_theory_bytes`, `compute_kv_scale_theory_bytes` from existing `vllm/engine/initialization/kv_cache_allocator.py`.

- [ ] **Step 1: Write the failing unit test**

`tests/test_flat_kv_cache_allocator.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.engine.initialization.flat_kv_cache_allocator import FlatKVCacheAllocator


def test_flat_allocator_uniform_layers():
    allocator = FlatKVCacheAllocator(
        num_layers=4,
        num_total_blocks=32,
        block_size=16,
        device=torch.device("cpu"),
    )
    kv_caches, kv_scale_caches, returned_num_blocks = allocator.allocate(
        layer_kv_specs=None,
        kv_dtype=torch.float16,
        kv_head_dim=64,
        fallback_num_kv_heads=8,
        fallback_kv_head_dim=64,
        needs_scale_cache=False,
    )
    assert returned_num_blocks == 32
    assert len(kv_caches) == 4
    for k, v in kv_caches:
        assert k.shape == (32, 16, 8, 64)
        assert v.shape == k.shape
        assert k.is_contiguous()
        assert k.storage().data_ptr() == v.storage().data_ptr()


def test_flat_allocator_shared_storage_across_layers():
    allocator = FlatKVCacheAllocator(
        num_layers=2,
        num_total_blocks=16,
        block_size=8,
        device=torch.device("cpu"),
    )
    kv_caches, _, _ = allocator.allocate(
        layer_kv_specs=None,
        kv_dtype=torch.float16,
        kv_head_dim=32,
        fallback_num_kv_heads=4,
        fallback_kv_head_dim=32,
        needs_scale_cache=False,
    )
    ptrs = [k.untyped_storage().data_ptr() for k, _ in kv_caches]
    ptrs += [v.untyped_storage().data_ptr() for _, v in kv_caches]
    assert len(set(ptrs)) == 1
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
uv run pytest tests/test_flat_kv_cache_allocator.py -v
```

Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `FlatKVCacheAllocator`**

`vllm/engine/initialization/flat_kv_cache_allocator.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from vllm.engine.initialization.kv_cache_allocator import (
    layer_kv_cache_shape_for_layer,
)


class FlatKVCacheAllocator:
    """Allocates a single flat GPU buffer for all per-layer K/V caches."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_total_blocks: int,
        block_size: int,
        device: torch.device,
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_total_blocks = int(num_total_blocks)
        self.block_size = int(block_size)
        self.device = device

    def allocate(
        self,
        layer_kv_specs: list[tuple[int, int]] | None,
        kv_dtype: torch.dtype,
        kv_head_dim: int,
        fallback_num_kv_heads: int,
        fallback_kv_head_dim: int,
        needs_scale_cache: bool,
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor]],
        list[tuple[torch.Tensor | None, torch.Tensor | None]],
        int,
    ]:
        block_elems_per_layer: list[int] = []
        for layer_idx in range(self.num_layers):
            nkv, hdim = layer_kv_cache_shape_for_layer(
                layer_kv_specs,
                layer_idx,
                kv_dtype,
                fallback_num_kv_heads,
                fallback_kv_head_dim,
            )
            block_elems_per_layer.append(
                self.block_size * nkv * hdim
            )

        total_block_elems = sum(block_elems_per_layer)
        half_size = self.num_total_blocks * total_block_elems
        kv_buffer = torch.zeros(
            half_size * 2,
            dtype=kv_dtype,
            device=self.device,
        )

        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        offset = 0
        for layer_idx, block_elems in enumerate(block_elems_per_layer):
            layer_elems = self.num_total_blocks * block_elems
            k_view = kv_buffer[offset : offset + layer_elems].view(
                self.num_total_blocks,
                self.block_size,
                layer_kv_cache_shape_for_layer(
                    layer_kv_specs,
                    layer_idx,
                    kv_dtype,
                    fallback_num_kv_heads,
                    fallback_kv_head_dim,
                )[0],
                layer_kv_cache_shape_for_layer(
                    layer_kv_specs,
                    layer_idx,
                    kv_dtype,
                    fallback_num_kv_heads,
                    fallback_kv_head_dim,
                )[1],
            )
            offset += layer_elems
            v_view = kv_buffer[offset : offset + layer_elems].view_as(k_view)
            offset += layer_elems
            kv_caches.append((k_view, v_view))

        kv_scale_caches: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
        if needs_scale_cache:
            scale_buffer = torch.zeros(
                half_size * 2,
                dtype=torch.float32,
                device=self.device,
            )
            scale_offset = 0
            for layer_idx in range(self.num_layers):
                nkv, _ = layer_kv_cache_shape_for_layer(
                    layer_kv_specs,
                    layer_idx,
                    torch.uint8,
                    fallback_num_kv_heads,
                    fallback_kv_head_dim,
                )
                layer_scale_elems = self.num_total_blocks * self.block_size * nkv
                ks_view = scale_buffer[scale_offset : scale_offset + layer_scale_elems].view(
                    self.num_total_blocks, self.block_size, nkv, 1
                )
                scale_offset += layer_scale_elems
                vs_view = scale_buffer[scale_offset : scale_offset + layer_scale_elems].view_as(ks_view)
                scale_offset += layer_scale_elems
                kv_scale_caches.append((ks_view, vs_view))
        else:
            kv_scale_caches = [(None, None)] * self.num_layers

        return kv_caches, kv_scale_caches, self.num_total_blocks
```

Refactor duplicate `layer_kv_cache_shape_for_layer` calls into a helper inside the method.

- [ ] **Step 4: Register in `vllm/engine/initialization/__init__.py`**

```python
from vllm.engine.initialization.flat_kv_cache_allocator import FlatKVCacheAllocator

__all__ = [
    "KVCacheAllocator",
    "FlatKVCacheAllocator",
    "MemoryAuditor",
    "LiteRuntimeAssembler",
]
```

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/test_flat_kv_cache_allocator.py -v
```

Expected: PASS.

- [ ] **Step 6: Add a test for per-layer specs and scale caches**

Append to `tests/test_flat_kv_cache_allocator.py`:

```python
def test_flat_allocator_per_layer_specs_and_scales():
    allocator = FlatKVCacheAllocator(
        num_layers=2,
        num_total_blocks=8,
        block_size=16,
        device=torch.device("cpu"),
    )
    layer_kv_specs = [(4, 64), (2, 128)]
    kv_caches, kv_scale_caches, _ = allocator.allocate(
        layer_kv_specs=layer_kv_specs,
        kv_dtype=torch.uint8,
        kv_head_dim=64,
        fallback_num_kv_heads=4,
        fallback_kv_head_dim=64,
        needs_scale_cache=True,
    )
    assert kv_caches[0][0].shape == (8, 16, 4, 32)  # uint8 packed halves head_dim
    assert kv_caches[1][0].shape == (8, 16, 2, 64)
    assert kv_scale_caches[0][0].shape == (8, 16, 4, 1)
    assert kv_scale_caches[1][0].shape == (8, 16, 2, 1)
```

Run:

```bash
uv run pytest tests/test_flat_kv_cache_allocator.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
uv run ruff check vllm/engine/initialization/flat_kv_cache_allocator.py tests/test_flat_kv_cache_allocator.py
uv run ruff format vllm/engine/initialization/flat_kv_cache_allocator.py tests/test_flat_kv_cache_allocator.py
git add vllm/engine/initialization/flat_kv_cache_allocator.py \
  vllm/engine/initialization/__init__.py \
  tests/test_flat_kv_cache_allocator.py
git commit -m "feat(kv-flat-pool): flat KV cache allocator with per-layer views"
```

---

## Task 3: `BlockAllocator`

**Files:**
- Create: `vllm/engine/block_allocator.py`
- Test: `tests/test_block_allocator.py`

**Interfaces:**
- Produces: `BlockAllocator(num_total_blocks)` with `allocate(n)`, `free(ids)`, `num_free`, `can_allocate(n)`.

- [ ] **Step 1: Write the failing unit test**

`tests/test_block_allocator.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm.engine.block_allocator import BlockAllocator


def test_null_block_reserved():
    ba = BlockAllocator(num_total_blocks=8)
    ids = ba.allocate(7)
    assert 0 not in ids
    assert ba.num_free == 0
    with pytest.raises(RuntimeError):
        ba.allocate(1)


def test_allocate_and_free():
    ba = BlockAllocator(num_total_blocks=5)
    ids1 = ba.allocate(2)
    ids2 = ba.allocate(2)
    assert len(set(ids1 + ids2)) == 4
    assert 0 not in ids1 + ids2
    ba.free(ids1)
    assert ba.num_free == 2
    ids3 = ba.allocate(2)
    assert set(ids3) == set(ids1)
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_block_allocator.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `BlockAllocator`**

`vllm/engine/block_allocator.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import deque
from typing import Iterable


class BlockAllocator:
    """Manages a pool of physical KV cache block IDs.

    Block ID 0 is reserved as the zeroed null block and is never handed out.
    """

    def __init__(self, num_total_blocks: int) -> None:
        self.num_total_blocks = int(num_total_blocks)
        if self.num_total_blocks < 1:
            raise ValueError("num_total_blocks must be positive")
        # Reserve ID 0; hand out IDs 1..N-1.
        self._free_ids: deque[int] = deque(range(1, self.num_total_blocks))

    def allocate(self, n: int) -> list[int]:
        n = int(n)
        if n < 0:
            raise ValueError("n must be non-negative")
        if n > len(self._free_ids):
            raise RuntimeError(
                f"Cannot allocate {n} blocks; only {len(self._free_ids)} free"
            )
        return [self._free_ids.popleft() for _ in range(n)]

    def free(self, block_ids: Iterable[int]) -> None:
        for bid in block_ids:
            bid = int(bid)
            if bid <= 0 or bid >= self.num_total_blocks:
                raise ValueError(f"Invalid block id to free: {bid}")
            self._free_ids.append(bid)

    @property
    def num_free(self) -> int:
        return len(self._free_ids)

    def can_allocate(self, n: int) -> bool:
        return int(n) <= len(self._free_ids)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_block_allocator.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check vllm/engine/block_allocator.py tests/test_block_allocator.py
uv run ruff format vllm/engine/block_allocator.py tests/test_block_allocator.py
git add vllm/engine/block_allocator.py tests/test_block_allocator.py
git commit -m "feat(kv-flat-pool): global block ID allocator with null block"
```

---

## Task 4: `compute_slot_mapping` Triton Kernel

**Files:**
- Create: `vllm/kernels/triton/compute_slot_mapping.py`
- Test: `tests/test_compute_slot_mapping.py`

**Interfaces:**
- Produces: `compute_slot_mapping(query_start_loc, positions, block_table, block_size, slot_mapping, pad_id)`.

- [ ] **Step 1: Write the failing test**

`tests/test_compute_slot_mapping.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.kernels.triton.compute_slot_mapping import compute_slot_mapping


def _cpu_reference(query_start_loc, positions, block_table, block_size, pad_id):
    slot_mapping = torch.full_like(positions, pad_id)
    for req_idx in range(len(query_start_loc) - 1):
        start = int(query_start_loc[req_idx].item())
        end = int(query_start_loc[req_idx + 1].item())
        for t in range(start, end):
            pos = int(positions[t].item())
            block_idx = pos // block_size
            block_id = int(block_table[req_idx, block_idx].item())
            slot_mapping[t] = block_id * block_size + (pos % block_size)
    return slot_mapping


def test_compute_slot_mapping_basic():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    block_size = 16
    block_table = torch.tensor(
        [[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int32, device=device
    )
    positions = torch.tensor([0, 15, 16, 31, 0, 17], dtype=torch.long, device=device)
    query_start_loc = torch.tensor([0, 4, 6], dtype=torch.int32, device=device)
    slot_mapping = torch.empty_like(positions)
    compute_slot_mapping(query_start_loc, positions, block_table, block_size, slot_mapping, pad_id=-1)
    expected = _cpu_reference(query_start_loc, positions, block_table, block_size, -1)
    assert torch.equal(slot_mapping.cpu(), expected.cpu())
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_compute_slot_mapping.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the Triton kernel**

`vllm/kernels/triton/compute_slot_mapping.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""
Compute slot mapping from positions and block tables.

Memory layout:
  query_start_loc: (num_reqs + 1,) int32, prefix sum of token counts.
  positions:       (num_tokens,) int64, token positions within each request.
  block_table:     (max_reqs, max_blocks_per_req) int32, block IDs per request.
  slot_mapping:    (num_tokens,) int64, output linear slot index.

Tiling:
  Grid: (num_tokens,)
  Each program handles one token.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compute_slot_mapping_kernel(
    query_start_loc_ptr,
    positions_ptr,
    block_table_ptr,
    slot_mapping_ptr,
    num_tokens,
    block_size,
    max_num_reqs,
    stride_bt_req,
    stride_bt_block,
    PAD_ID: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        tl.store(slot_mapping_ptr + token_idx, PAD_ID)
        return

    # Find request index by scanning query_start_loc. max_num_reqs equals
    # max_active_requests and is small (<= 16), so a linear scan is fine.
    req_idx = 0
    found = False
    for r in range(max_num_reqs):
        start = tl.load(query_start_loc_ptr + r).to(tl.int64)
        end = tl.load(query_start_loc_ptr + r + 1).to(tl.int64)
        if token_idx >= start and token_idx < end:
            req_idx = r
            found = True
            break

    if not found:
        tl.store(slot_mapping_ptr + token_idx, PAD_ID)
        return

    pos = tl.load(positions_ptr + token_idx).to(tl.int64)
    block_idx = pos // block_size
    block_id = tl.load(
        block_table_ptr + req_idx * stride_bt_req + block_idx * stride_bt_block
    ).to(tl.int64)
    slot_id = block_id * block_size + (pos % block_size)
    tl.store(slot_mapping_ptr + token_idx, slot_id)


def compute_slot_mapping(
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    pad_id: int = -1,
) -> None:
    """Fill ``slot_mapping`` in-place."""
    num_tokens = positions.shape[0]
    if num_tokens == 0:
        return
    grid = (slot_mapping.shape[0],)
    _compute_slot_mapping_kernel[grid](
        query_start_loc,
        positions,
        block_table,
        slot_mapping,
        num_tokens,
        block_size,
        block_table.shape[0],
        block_table.stride(0),
        block_table.stride(1),
        PAD_ID=pad_id,
    )
```

- [ ] **Step 4: Run the test and iterate until PASS**

```bash
uv run pytest tests/test_compute_slot_mapping.py -v
```

- [ ] **Step 5: Add a padding test**

Append to `tests/test_compute_slot_mapping.py`:

```python
def test_compute_slot_mapping_with_padding():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    block_size = 16
    block_table = torch.tensor([[1, 2, 0, 0]], dtype=torch.int32, device=device)
    positions = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
    query_start_loc = torch.tensor([0, 3, 3], dtype=torch.int32, device=device)
    slot_mapping = torch.full((4,), -999, dtype=torch.long, device=device)
    compute_slot_mapping(query_start_loc, positions, block_table, block_size, slot_mapping, pad_id=-1)
    assert slot_mapping[3].item() == -1
```

Run:

```bash
uv run pytest tests/test_compute_slot_mapping.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
uv run ruff check vllm/kernels/triton/compute_slot_mapping.py tests/test_compute_slot_mapping.py
uv run ruff format vllm/kernels/triton/compute_slot_mapping.py tests/test_compute_slot_mapping.py
git add vllm/kernels/triton/compute_slot_mapping.py tests/test_compute_slot_mapping.py
git commit -m "feat(kv-flat-pool): triton compute_slot_mapping kernel"
```

---

## Task 5: `KVBlockManager` Dynamic Block Support

**Files:**
- Modify: `vllm/engine/kv_block_manager.py`
- Test: `tests/test_kv_block_manager.py`

**Interfaces:**
- Consumes: `BlockAllocator`, per-layer `(K,V)` views from `FlatKVCacheAllocator`.
- Produces: `ensure_blocks_for_requests`, `ensure_blocks`, `free_request_blocks`, `block_table_for_slot`, `update_block_table_row`, `capture_prefix_entry(key, request_id, ...)`, `materialize_prefix_entry(request_id, entry, prefix_len)`.

- [ ] **Step 1: Write the failing test**

`tests/test_kv_block_manager.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.engine.block_allocator import BlockAllocator
from vllm.engine.kv_block_manager import KVBlockManager


def _make_manager(num_layers=2, num_total_blocks=8, block_size=16, num_blocks_per_seq=4):
    kv_caches = []
    for _ in range(num_layers):
        k = torch.zeros(num_total_blocks, block_size, 2, 32)
        v = torch.zeros_like(k)
        kv_caches.append((k, v))
    return KVBlockManager(
        kv_caches=kv_caches,
        kv_scale_caches=[(None, None)] * num_layers,
        num_blocks_per_seq=num_blocks_per_seq,
        block_size=block_size,
        max_active_requests=4,
        block_allocator=BlockAllocator(num_total_blocks),
    )


def test_ensure_blocks_grows_and_pads():
    mgr = _make_manager()
    mgr.ensure_blocks("r0", 20)
    row = mgr.block_table_for_slot(0)
    assert row.shape == (4,)
    assert row[0].item() != 0
    assert row[1].item() != 0
    assert row[2].item() == 0
    assert row[3].item() == 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_kv_block_manager.py -v
```

Expected: TypeError / signature mismatch.

- [ ] **Step 3: Update `KVBlockManager` constructor and add new methods**

Modify `vllm/engine/kv_block_manager.py`:

```python
def __init__(
    self,
    *,
    kv_caches: Any,
    kv_scale_caches: Any,
    num_blocks_per_seq: int,
    block_size: int,
    max_active_requests: int,
    block_allocator: BlockAllocator,
) -> None:
    self.kv_caches = kv_caches
    self.kv_scale_caches = kv_scale_caches
    self.num_blocks_per_seq = int(num_blocks_per_seq)
    self.block_size = int(block_size)
    self._allocator = block_allocator
    self._request_blocks: dict[str, list[int]] = {}
    self._block_table_buffer = torch.zeros(
        (max_active_requests, num_blocks_per_seq),
        dtype=torch.int32,
        device=kv_caches[0][0].device,
    )
```

Implement `ensure_blocks`, `ensure_blocks_for_requests`, `free_request_blocks`, `block_table_for_slot`, `update_block_table_row`. Use `cdiv` helper (from `vllm.utils.math_utils`) for block count.

- [ ] **Step 4: Update prefix-cache methods to use request IDs**

Change signatures:

```python
def capture_prefix_entry(
    self,
    *,
    key: tuple[int, ...],
    request_id: str,
    prompt_len: int,
    last_prompt_logits: torch.Tensor,
) -> PrefixCacheEntry:
    ...

def materialize_prefix_entry(
    self,
    *,
    request_id: str,
    entry: PrefixCacheEntry,
    prefix_len: int,
) -> None:
    ...
```

Capture uses advanced indexing over the request's block IDs. Materialize allocates blocks for the target request, copies entry blocks back, and updates the block-table row (the caller must supply `slot_idx` separately if needed).

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_kv_block_manager.py -v
```

Expected: PASS.

- [ ] **Step 6: Add prefix-cache capture/materialize test**

Append to `tests/test_kv_block_manager.py`:

```python
def test_prefix_capture_and_materialize():
    mgr = _make_manager(num_total_blocks=16)
    mgr.ensure_blocks("r0", 32)
    # Write a marker into r0's first block of layer 0 K cache
    k0, _ = mgr.kv_caches[0]
    block_id = mgr._request_blocks["r0"][0]
    k0[block_id, 0, 0, 0] = 3.14

    entry = mgr.capture_prefix_entry(
        key=(1, 2, 3),
        request_id="r0",
        prompt_len=32,
        last_prompt_logits=torch.zeros(10),
    )
    mgr.free_request_blocks("r0")
    mgr.ensure_blocks("r1", 32)
    mgr.materialize_prefix_entry(request_id="r1", entry=entry, prefix_len=32)

    new_block_id = mgr._request_blocks["r1"][0]
    assert torch.isclose(k0[new_block_id, 0, 0, 0], torch.tensor(3.14))
```

Run:

```bash
uv run pytest tests/test_kv_block_manager.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
uv run ruff check vllm/engine/kv_block_manager.py tests/test_kv_block_manager.py
uv run ruff format vllm/engine/kv_block_manager.py tests/test_kv_block_manager.py
git add vllm/engine/kv_block_manager.py tests/test_kv_block_manager.py
git commit -m "feat(kv-flat-pool): dynamic block tables in KVBlockManager"
```

---

## Task 6: `InputBatchBuilder` Dynamic Slot Mapping

**Files:**
- Modify: `vllm/engine/input_batch_builder.py`
- Test: `tests/test_input_batch_builder_dynamic.py`

**Interfaces:**
- Consumes: `KVBlockManager.block_table_for_slot`, `compute_slot_mapping`.

- [ ] **Step 1: Write a failing test that exercises the new slot-mapping path**

`tests/test_input_batch_builder_dynamic.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.engine.block_allocator import BlockAllocator
from vllm.engine.input_batch_builder import InputBatchBuilder
from vllm.engine.kv_block_manager import KVBlockManager
from vllm.engine.request_state import RequestState
from vllm.sampling_params import SamplingParams


def _make_builder(num_layers=2, num_blocks_per_seq=4):
    kv_caches = []
    for _ in range(num_layers):
        k = torch.zeros(8, 16, 2, 32)
        v = torch.zeros_like(k)
        kv_caches.append((k, v))
    mgr = KVBlockManager(
        kv_caches=kv_caches,
        kv_scale_caches=[(None, None)] * num_layers,
        num_blocks_per_seq=num_blocks_per_seq,
        block_size=16,
        max_active_requests=2,
        block_allocator=BlockAllocator(8),
    )
    class Cfg:
        kv_type = "fp16"
        k_scale = 1.0
        v_scale = 1.0
    return InputBatchBuilder(
        device=torch.device("cpu"),
        max_model_len=64,
        num_layers=num_layers,
        kv_block_manager=mgr,
        inf_config=Cfg(),
        stack_per_layer_carries=lambda *_: None,
        split_per_layer_carries=lambda *_: None,
    ), mgr


def test_prefill_slot_mapping_dynamic_blocks():
    builder, mgr = _make_builder()
    req = RequestState(
        request_id="r0",
        prompt="hi",
        input_ids=list(range(20)),
        sampling_params=SamplingParams(),
        slot_idx=0,
    )
    mgr.ensure_blocks("r0", 20)
    class Scheduler:
        def get_request(self, rid):
            return req
    curr_input, positions, attn_metadata, _, _ = builder.build_prefill(
        ["r0"], Scheduler(), chunk_len=20
    )
    slot_mapping = attn_metadata["slot_mapping"]
    # First token in first block, 20th token starts second block.
    assert slot_mapping[0].item() == mgr._request_blocks["r0"][0] * 16
    assert slot_mapping[16].item() == mgr._request_blocks["r0"][1] * 16
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_input_batch_builder_dynamic.py -v
```

Expected: assertion failure because slot mapping is still based on static slot index.

- [ ] **Step 3: Update `InputBatchBuilder` to use dynamic block tables**

In `vllm/engine/input_batch_builder.py`:

1. Replace `slot_idx * self.max_model_len + torch.arange(...)` with a call to `compute_slot_mapping`.
2. Build `positions` stacked tensor and `query_start_loc` from the request list.
3. Stack `block_tables` by calling `self.kv_block_manager.block_table_for_slot(slot_idx)` for each request.
4. Pre-allocate `slot_mapping` tensor of shape `(sum(chunk_lens),)` and invoke `compute_slot_mapping(query_start_loc, positions, block_tables, block_size, slot_mapping, pad_id=-1)`.
5. For `build_decode_batch` and `build_decode_fast`, compute per-token slot mapping from the request's block table on the CPU (or reuse the kernel with `num_tokens == batch_size`).

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_input_batch_builder_dynamic.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff check vllm/engine/input_batch_builder.py tests/test_input_batch_builder_dynamic.py
uv run ruff format vllm/engine/input_batch_builder.py tests/test_input_batch_builder_dynamic.py
git add vllm/engine/input_batch_builder.py tests/test_input_batch_builder_dynamic.py
git commit -m "feat(kv-flat-pool): dynamic slot mapping in InputBatchBuilder"
```

---

## Task 7: Runtime Assembly Plumbing

**Files:**
- Modify: `vllm/engine/initialization/runtime_component_factory.py`
- Modify: `vllm/engine/runtime_factory.py`

**Interfaces:**
- Consumes: `BlockAllocator`.
- Produces: `KVBlockManager` constructed with allocator.

- [ ] **Step 1: Add `block_allocator` to `RuntimeAssemblyContext`**

In `vllm/engine/runtime_factory.py`:

```python
from vllm.engine.block_allocator import BlockAllocator

@dataclass(frozen=True)
class RuntimeAssemblyContext:
    block_allocator: BlockAllocator
    kv_caches: list[torch.Tensor]
    ...
```

- [ ] **Step 2: Pass `block_allocator` through `LiteRuntimeAssembler`**

In `vllm/engine/initialization/runtime_component_factory.py`, add `block_allocator` to `assemble()` signature and forward it into `RuntimeAssemblyContext(...)`.

- [ ] **Step 3: Construct `KVBlockManager` with allocator**

In `vllm/engine/runtime_factory.py`, update `LiteRuntimeFactory.build`:

```python
kv_block_manager = KVBlockManager(
    kv_caches=context.kv_caches,
    kv_scale_caches=context.kv_scale_caches,
    num_blocks_per_seq=context.num_blocks_per_seq,
    block_size=context.block_size,
    max_active_requests=context.max_active_requests,
    block_allocator=context.block_allocator,
)
```

- [ ] **Step 4: Run import checks**

```bash
uv run python -c "from vllm.engine.runtime_factory import LiteRuntimeFactory, RuntimeAssemblyContext; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
uv run ruff check vllm/engine/initialization/runtime_component_factory.py vllm/engine/runtime_factory.py
uv run ruff format vllm/engine/initialization/runtime_component_factory.py vllm/engine/runtime_factory.py
git add vllm/engine/initialization/runtime_component_factory.py vllm/engine/runtime_factory.py
git commit -m "feat(kv-flat-pool): pass block allocator through runtime assembly"
```

---

## Task 8: `LiteEngine` Allocation Path

**Files:**
- Modify: `vllm/engine/lite_engine.py`

**Interfaces:**
- Consumes: `FlatKVCacheAllocator`, `BlockAllocator`.

- [ ] **Step 1: Replace `KVCacheAllocator` construction**

In `vllm/engine/lite_engine.py`:

```python
from vllm.engine.initialization import (
    BlockAllocator,
    FlatKVCacheAllocator,
    LiteRuntimeAssembler,
    MemoryAuditor,
)
```

Replace:

```python
allocator = KVCacheAllocator(
    num_layers=self.num_layers,
    num_total_blocks=self.num_total_blocks,
    block_size=self.block_size,
    device=self.device,
)
self.kv_caches, self.kv_scale_caches = allocator.allocate(...)
```

with:

```python
allocator = FlatKVCacheAllocator(
    num_layers=self.num_layers,
    num_total_blocks=self.num_total_blocks,
    block_size=self.block_size,
    device=self.device,
)
self.kv_caches, self.kv_scale_caches, _ = allocator.allocate(
    layer_kv_specs=self._layer_kv_specs,
    kv_dtype=self.kv_dtype,
    kv_head_dim=self.kv_head_dim,
    fallback_num_kv_heads=self.num_kv_heads,
    fallback_kv_head_dim=self.kv_head_dim,
    needs_scale_cache=kv_plan.needs_scale_cache,
)
block_allocator = BlockAllocator(num_total_blocks=self.num_total_blocks)
```

- [ ] **Step 2: Remove static `_fast_block_tables`**

Delete the block of code that precomputes `self._fast_block_tables` and remove it from `LiteRuntimeAssembler.assemble(...)`.

- [ ] **Step 3: Pass `block_allocator` to assembler**

```python
runtime_components = LiteRuntimeAssembler.assemble(
    block_allocator=block_allocator,
    kv_caches=self.kv_caches,
    ...
)
```

- [ ] **Step 4: Verify the engine still imports**

```bash
uv run python -c "from vllm.engine.lite_engine import LiteEngine; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
uv run ruff check vllm/engine/lite_engine.py
uv run ruff format vllm/engine/lite_engine.py
git add vllm/engine/lite_engine.py
git commit -m "feat(kv-flat-pool): wire flat allocator and block allocator into LiteEngine"
```

---

## Task 9: Backend / Controller Block Lifecycle

**Files:**
- Modify: `vllm/engine/runtime_controller.py`
- Modify: `vllm/engine/backend/lite_single_gpu.py`

**Interfaces:**
- Consumes: `KVBlockManager.ensure_blocks_for_requests`, `free_request_blocks`.

- [ ] **Step 1: Add `ensure_kv_blocks` to `LiteSingleGpuBackend`**

In `vllm/engine/backend/lite_single_gpu.py`, add a method that the controller calls before each step:

```python
def ensure_kv_blocks(self, step_plan: StepPlan) -> None:
    request_ids: list[str] = []
    token_counts: list[int] = []

    if step_plan.prefills is not None:
        for rid in step_plan.prefills.request_ids:
            req = self.scheduler.get_request(rid)
            start_pos = max(
                int(req.seq_len),
                int(req.prefix_hit_len or req._prefix_cache_hit_len or 0),
            )
            chunk_len = step_plan.prefills.chunk_len
            request_ids.append(rid)
            token_counts.append(start_pos + chunk_len)

    if step_plan.decodes is not None:
        for rid in step_plan.decodes.request_ids:
            req = self.scheduler.get_request(rid)
            request_ids.append(rid)
            token_counts.append(int(req.seq_len) + 1)

    if request_ids:
        self.kv_block_manager.ensure_blocks_for_requests(request_ids, token_counts)
```

- [ ] **Step 2: Call `ensure_kv_blocks` from `RuntimeController.step`**

In `vllm/engine/runtime_controller.py`, add the call after `_admit_requests` and before `observer.on_step_started`:

```python
self._admit_requests(step_plan, now)

if self.scheduler.running_request_count == 0:
    return []

self.backend.ensure_kv_blocks(step_plan)

self.observer.on_step_started(step_plan)
```

- [ ] **Step 3: Free blocks when a request ends**

In `vllm/engine/backend/lite_single_gpu.py`, in `_free_request`, add block freeing before LoRA cleanup:

```python
def _free_request(self, request_id: str) -> None:
    self.kv_block_manager.free_request_blocks(request_id)
    request = self.scheduler.free_request(request_id)
    if request is not None and self.lora_registry is not None:
        self.lora_registry.on_request_removed(request.lora_id)
```

- [ ] **Step 4: Update prefix-cache methods to use `request_id`**

In `vllm/engine/backend/lite_single_gpu.py`:

```python
def _store_prefix_cache_entry(self, request_id, request_state, last_prompt_logits):
    ...
    entry = self.kv_block_manager.capture_prefix_entry(
        key=key,
        request_id=request_id,
        prompt_len=prompt_len,
        last_prompt_logits=last_prompt_logits,
    )
    ...

def _materialize_prefix_cache_entry(self, request_state, entry, prefix_len):
    ...
    self.kv_block_manager.materialize_prefix_entry(
        request_id=request_state.request_id,
        entry=entry,
        prefix_len=prefix_len,
    )
    self.kv_block_manager.update_block_table_row(
        int(request_state.slot_idx),
        request_state.request_id,
    )
    ...
```

- [ ] **Step 4: Run regression suite**

```bash
bash tests/run_regression_suite.sh
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
uv run ruff check vllm/engine/runtime_controller.py vllm/engine/backend/lite_single_gpu.py
uv run ruff format vllm/engine/runtime_controller.py vllm/engine/backend/lite_single_gpu.py
git add vllm/engine/runtime_controller.py vllm/engine/backend/lite_single_gpu.py
git commit -m "feat(kv-flat-pool): dynamic block allocation lifecycle in controller and backend"
```

---

## Task 10: End-to-End Validation and Cleanup

**Files:**
- Modify: any remaining lint issues.
- Delete: old `vllm/engine/initialization/kv_cache_allocator.py` (after confirming `FlatKVCacheAllocator` fully replaces it).

- [ ] **Step 1: Run lint and format**

```bash
uv run ruff check .
uv run ruff format .
```

Expected: no errors.

- [ ] **Step 2: Run fast regression**

```bash
bash tests/run_regression_suite.sh
```

Expected: PASS.

- [ ] **Step 3: Run correctness regression (skip A tier for speed)**

```bash
SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh
```

Expected: PASS.

- [ ] **Step 4: Run e2e benchmark smoke**

```bash
uv run python tests/e2e_full_benchmark.py
```

Expected: exit 0.

- [ ] **Step 5: Remove the old allocator (optional but recommended)**

After all tests pass, remove `vllm/engine/initialization/kv_cache_allocator.py` and update any remaining imports. Re-run the regression suite.

- [ ] **Step 6: Final commit and push branch**

```bash
git commit -m "feat(kv-flat-pool): remove legacy per-layer allocator"
git push origin feat/kv-flat-pool
```

---

## Spec Coverage Checklist

| Spec Section | Implementing Task |
|---|---|
| Flat buffer layout | Task 2 |
| Null block reservation | Task 3 |
| Per-request block IDs | Task 5 |
| `compute_slot_mapping` kernel | Task 4 |
| Dynamic `slot_mapping` in `InputBatchBuilder` | Task 6 |
| Runtime assembly plumbing | Task 7 |
| `LiteEngine` construction | Task 8 |
| Controller/backend block lifecycle | Task 9 |
| Prefix-cache request-ID interface | Task 5 / Task 9 |
| Block zeroing | Task 5 |
| Testing plan | All tasks |

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-02-kv-flat-pool-dynamic-blocks.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

**Which approach?**
