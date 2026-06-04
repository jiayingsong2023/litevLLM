# DeepSeek V4 Flash Q2 Native Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add experimental native FastInference support for `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` with `batch=1`, `context=4096/8192`, greedy decode, and OpenAI-compatible REST access.

**Target correction:** This plan originally referenced
`DeepSeek-V4-Flash-Spark-Q2-REAP-ds4.gguf`. Public repository inspection showed
that exact file was not available. The first real target is DS4's
`q2-imatrix` GGUF from `antirez/deepseek-v4-gguf`.

**Architecture:** Implement a DeepSeek V4 Flash model family as a vertical lite model package with strict DS4 GGUF parsing, mmap-backed weight descriptors, model-local paged compressed KV, and Triton quant kernels. Keep `LiteEngine`, `StepScheduler`, and `RequestScheduler` model-agnostic; DeepSeek-specific behavior belongs in `vllm/adapters/deepseek_v4_flash.py`, `vllm/model_executor/models/deepseek_v4_flash/`, and `vllm/kernels/triton/deepseek_v4_flash/`.

**Tech Stack:** Python 3.12, uv, PyTorch, Triton through `vllm/triton_utils/`, mmap, pytest, FastAPI/OpenAI-compatible REST.

---

## File Structure

Create:

- `vllm/adapters/deepseek_v4_flash.py` - model detection and runtime policy.
- `vllm/model_executor/models/deepseek_v4_flash/__init__.py` - public model exports.
- `vllm/model_executor/models/deepseek_v4_flash/config.py` - DeepSeek V4 Flash constants, metadata validation, memory estimates.
- `vllm/model_executor/models/deepseek_v4_flash/gguf_reader.py` - strict GGUF v3 metadata and tensor directory parser.
- `vllm/model_executor/models/deepseek_v4_flash/weight_store.py` - semantic tensor binding, mmap lifetime, memory/cache counters.
- `vllm/model_executor/models/deepseek_v4_flash/quant.py` - PyTorch reference decode/dot helpers for Q8_0, IQ2_XXS, and Q2_K.
- `vllm/model_executor/models/deepseek_v4_flash/compressed_kv.py` - raw SWA and compressed KV page tables, chunk pools, and row accounting/state.
- `vllm/model_executor/models/deepseek_v4_flash/attention.py` - model-local attention reference and Triton call wrappers.
- `vllm/model_executor/models/deepseek_v4_flash/moe.py` - router/top-k and routed expert execution wrappers.
- `vllm/model_executor/models/deepseek_v4_flash/model.py` - `DeepSeekV4FlashForCausalLM` integration.
- `vllm/kernels/triton/deepseek_v4_flash/__init__.py` - kernel package marker.
- `vllm/kernels/triton/deepseek_v4_flash/q8_linear.py` - Q8_0 linear kernels.
- `vllm/kernels/triton/deepseek_v4_flash/iq2_xxs.py` - IQ2_XXS decode/dot kernels.
- `vllm/kernels/triton/deepseek_v4_flash/q2_k.py` - Q2_K decode/dot kernels.
- `vllm/kernels/triton/deepseek_v4_flash/routed_moe.py` - batch=1 top-6 routed expert kernels.
- `vllm/kernels/triton/deepseek_v4_flash/compressed_attention.py` - raw/compressed attention kernels that consume page tables instead of contiguous full-context KV tensors.
- `tests/deepseek_v4_flash/fixtures.py` - synthetic GGUF and quant block fixtures.
- `tests/deepseek_v4_flash/test_gguf_reader.py`
- `tests/deepseek_v4_flash/test_config_memory.py`
- `tests/deepseek_v4_flash/test_quant_reference.py`
- `tests/deepseek_v4_flash/test_triton_q8_linear.py`
- `tests/deepseek_v4_flash/test_triton_q2_iq2.py`
- `tests/deepseek_v4_flash/test_compressed_kv.py`
- `tests/deepseek_v4_flash/test_compressed_attention_contract.py`
- `tests/deepseek_v4_flash/test_adapter_registry.py`
- `tests/deepseek_v4_flash/test_model_smoke_no_weights.py`
- `tests/smoke/test_deepseek_v4_flash_http_smoke.py`

Modify:

- `vllm/adapters/__init__.py` - export the new adapter.
- `vllm/adapters/registry.py` - detect DeepSeek V4 Flash from GGUF metadata/config.
- `vllm/model_executor/models/registry.py` - register `DeepSeekV4FlashForCausalLM`.
- `vllm/serving/config_builder.py` - detect DS4 GGUF profile and enforce the first-release context cap.
- `vllm/model_executor/model_loader/__init__.py` - route the target GGUF to the DeepSeek V4 Flash model class without safetensors loading.
- `vllm/engine/runtime_observer.py` - expose DeepSeek expert-cache counters if the model reports them.
- `docs/CAPABILITY_MATRIX.md` - add experimental DeepSeek V4 Flash Q2 status after implementation passes smoke.
- `docs/models/supported_models.md` - document experimental status after implementation passes smoke.

## Implementation Tasks

### Task 1: DeepSeek V4 Flash constants and memory policy

**Files:**
- Create: `vllm/model_executor/models/deepseek_v4_flash/__init__.py`
- Create: `vllm/model_executor/models/deepseek_v4_flash/config.py`
- Test: `tests/deepseek_v4_flash/test_config_memory.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/deepseek_v4_flash/test_config_memory.py`:

```python
from vllm.model_executor.models.deepseek_v4_flash.config import (
    DEEPSEEK_V4_FLASH_SHAPE,
    DeepSeekV4FlashMemoryPolicy,
    layer_compress_ratio,
)


def test_flash_shape_matches_target_model() -> None:
    shape = DEEPSEEK_V4_FLASH_SHAPE
    assert shape.num_layers == 43
    assert shape.hidden_size == 4096
    assert shape.vocab_size == 129280
    assert shape.num_attention_heads == 64
    assert shape.num_kv_heads == 1
    assert shape.head_dim == 512
    assert shape.sliding_window == 128
    assert shape.num_experts == 256
    assert shape.num_experts_per_tok == 6


def test_layer_compress_ratio_pattern() -> None:
    assert layer_compress_ratio(0) == 0
    assert layer_compress_ratio(1) == 0
    assert layer_compress_ratio(2) == 4
    assert layer_compress_ratio(3) == 128
    assert layer_compress_ratio(4) == 4
    assert layer_compress_ratio(42) == 4


def test_memory_policy_caps_first_release_context() -> None:
    policy = DeepSeekV4FlashMemoryPolicy()
    assert policy.validate_context_length(4096) == 4096
    assert policy.validate_context_length(8192) == 8192
    try:
        policy.validate_context_length(16384)
    except ValueError as exc:
        assert "8192" in str(exc)
    else:
        raise AssertionError("context above first-release cap must fail")


def test_memory_estimate_increases_with_context() -> None:
    policy = DeepSeekV4FlashMemoryPolicy()
    estimate_4k = policy.estimate_context_bytes(4096)
    estimate_8k = policy.estimate_context_bytes(8192)
    assert estimate_4k.raw_kv_bytes > 0
    assert estimate_4k.compressed_kv_bytes > 0
    assert estimate_8k.total_bytes > estimate_4k.total_bytes
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_config_memory.py -q
```

Expected: fails with `ModuleNotFoundError` for `vllm.model_executor.models.deepseek_v4_flash`.

- [ ] **Step 3: Add the constants and policy**

Create `vllm/model_executor/models/deepseek_v4_flash/__init__.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 Flash experimental native model package."""
```

Create `vllm/model_executor/models/deepseek_v4_flash/config.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekV4FlashShape:
    num_layers: int = 43
    hidden_size: int = 4096
    vocab_size: int = 129280
    num_attention_heads: int = 64
    num_kv_heads: int = 1
    head_dim: int = 512
    value_dim: int = 512
    rotary_dim: int = 64
    output_groups: int = 8
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    num_experts: int = 256
    num_experts_per_tok: int = 6
    num_shared_experts: int = 1
    expert_intermediate_size: int = 2048
    sliding_window: int = 128
    indexer_heads: int = 64
    indexer_head_dim: int = 128
    indexer_top_k: int = 512


@dataclass(frozen=True)
class DeepSeekV4FlashContextEstimate:
    context_length: int
    raw_kv_bytes: int
    compressed_kv_bytes: int
    scratch_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.raw_kv_bytes + self.compressed_kv_bytes + self.scratch_bytes


DEEPSEEK_V4_FLASH_SHAPE = DeepSeekV4FlashShape()


def layer_compress_ratio(layer_idx: int) -> int:
    if layer_idx < 0 or layer_idx >= DEEPSEEK_V4_FLASH_SHAPE.num_layers:
        raise ValueError(f"layer index out of range: {layer_idx}")
    if layer_idx < 2:
        return 0
    return 4 if layer_idx % 2 == 0 else 128


class DeepSeekV4FlashMemoryPolicy:
    max_first_release_context: int = 8192
    default_expert_cache_bytes: int = 2 * 1024 * 1024 * 1024

    def validate_context_length(self, context_length: int) -> int:
        if context_length <= 0:
            raise ValueError("context length must be positive")
        if context_length > self.max_first_release_context:
            raise ValueError(
                "DeepSeek V4 Flash first release supports context <= "
                f"{self.max_first_release_context}; got {context_length}"
            )
        return context_length

    def estimate_context_bytes(
        self,
        context_length: int,
        *,
        prefill_cap: int = 4096,
    ) -> DeepSeekV4FlashContextEstimate:
        ctx = self.validate_context_length(context_length)
        shape = DEEPSEEK_V4_FLASH_SHAPE
        raw_cap = min(max(shape.sliding_window + prefill_cap, shape.sliding_window), ctx)
        raw_cap = min(((raw_cap + 255) // 256) * 256, ctx)
        raw_kv = shape.num_layers * raw_cap * shape.head_dim * 4
        compressed_kv = 0
        for layer_idx in range(shape.num_layers):
            ratio = layer_compress_ratio(layer_idx)
            if ratio == 0:
                continue
            comp_cap = ctx // ratio + 2
            compressed_kv += comp_cap * shape.head_dim * 2
            if ratio == 4:
                compressed_kv += comp_cap * shape.indexer_head_dim * 4
        scratch = 2 * (ctx // 4 + 2) * prefill_cap * 4
        return DeepSeekV4FlashContextEstimate(
            context_length=ctx,
            raw_kv_bytes=raw_kv,
            compressed_kv_bytes=compressed_kv,
            scratch_bytes=scratch,
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_config_memory.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/__init__.py \
  vllm/model_executor/models/deepseek_v4_flash/config.py \
  tests/deepseek_v4_flash/test_config_memory.py
git commit -m "feat: add deepseek v4 flash shape policy"
```

### Task 2: Strict DS4 GGUF reader and inspect-only metadata

**Files:**
- Create: `vllm/model_executor/models/deepseek_v4_flash/gguf_reader.py`
- Create: `tests/deepseek_v4_flash/fixtures.py`
- Test: `tests/deepseek_v4_flash/test_gguf_reader.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/deepseek_v4_flash/fixtures.py`:

```python
from __future__ import annotations

import struct
from pathlib import Path


GGUF_MAGIC = 0x46554747


def _write_string(buf: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    buf.extend(struct.pack("<Q", len(encoded)))
    buf.extend(encoded)


def _write_kv_string(buf: bytearray, key: str, value: str) -> None:
    _write_string(buf, key)
    buf.extend(struct.pack("<I", 8))
    _write_string(buf, value)


def _write_kv_u32(buf: bytearray, key: str, value: int) -> None:
    _write_string(buf, key)
    buf.extend(struct.pack("<I", 4))
    buf.extend(struct.pack("<I", value))


def _write_tensor(buf: bytearray, name: str, dims: tuple[int, ...], tensor_type: int, offset: int) -> None:
    _write_string(buf, name)
    buf.extend(struct.pack("<I", len(dims)))
    for dim in dims:
        buf.extend(struct.pack("<Q", dim))
    buf.extend(struct.pack("<I", tensor_type))
    buf.extend(struct.pack("<Q", offset))


def write_minimal_deepseek_v4_flash_gguf(path: Path, *, block_count: int = 43) -> None:
    metadata = bytearray()
    _write_kv_string(metadata, "general.architecture", "deepseek4")
    _write_kv_string(metadata, "general.name", "DeepSeek V4 Flash Spark Q2 REAP")
    _write_kv_u32(metadata, "deepseek4.block_count", block_count)
    _write_kv_u32(metadata, "deepseek4.attention.head_count", 64)
    _write_kv_u32(metadata, "deepseek4.attention.head_count_kv", 1)
    _write_kv_u32(metadata, "deepseek4.attention.key_length", 512)
    _write_kv_u32(metadata, "deepseek4.attention.sliding_window", 128)
    _write_kv_u32(metadata, "deepseek4.attention.indexer.head_count", 64)
    _write_kv_u32(metadata, "deepseek4.attention.indexer.key_length", 128)
    _write_kv_u32(metadata, "deepseek4.attention.indexer.top_k", 512)
    _write_kv_u32(metadata, "deepseek4.expert_count", 256)
    _write_kv_u32(metadata, "deepseek4.expert_used_count", 6)
    _write_kv_u32(metadata, "deepseek4.embedding_length", 4096)
    _write_kv_u32(metadata, "deepseek4.vocab_size", 129280)

    tensors = bytearray()
    _write_tensor(tensors, "token_embd.weight", (4096, 129280), 8, 0)
    header = struct.pack("<IIQQ", GGUF_MAGIC, 3, 13, 1)
    path.write_bytes(header + metadata + tensors + b"\x00" * 64)
```

Create `tests/deepseek_v4_flash/test_gguf_reader.py`:

```python
from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGUFParseError,
    read_deepseek_v4_flash_gguf,
)

from .fixtures import write_minimal_deepseek_v4_flash_gguf


def test_reader_accepts_minimal_target_metadata(tmp_path) -> None:
    path = tmp_path / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
    write_minimal_deepseek_v4_flash_gguf(path)

    model = read_deepseek_v4_flash_gguf(path)

    assert model.name == "DeepSeek V4 Flash Spark Q2 REAP"
    assert model.shape.num_layers == 43
    assert model.metadata["general.architecture"] == "deepseek4"
    assert model.tensors["token_embd.weight"].name == "token_embd.weight"


def test_reader_rejects_wrong_layer_count(tmp_path) -> None:
    path = tmp_path / "bad.gguf"
    write_minimal_deepseek_v4_flash_gguf(path, block_count=44)

    try:
        read_deepseek_v4_flash_gguf(path)
    except GGUFParseError as exc:
        assert "block_count" in str(exc)
    else:
        raise AssertionError("invalid block_count must fail")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gguf_reader.py -q
```

Expected: fails with `ModuleNotFoundError` for `gguf_reader`.

- [ ] **Step 3: Implement the reader**

Create `vllm/model_executor/models/deepseek_v4_flash/gguf_reader.py` with a small GGUF v3 parser that supports string and u32 metadata plus tensor descriptors:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import mmap
import struct
from typing import Any

from .config import DEEPSEEK_V4_FLASH_SHAPE, DeepSeekV4FlashShape


class GGUFParseError(ValueError):
    pass


@dataclass(frozen=True)
class DeepSeekV4FlashTensor:
    name: str
    dims: tuple[int, ...]
    tensor_type: int
    offset: int


@dataclass(frozen=True)
class DeepSeekV4FlashGGUF:
    path: Path
    name: str
    metadata: dict[str, Any]
    tensors: dict[str, DeepSeekV4FlashTensor]
    shape: DeepSeekV4FlashShape = DEEPSEEK_V4_FLASH_SHAPE


class _Cursor:
    def __init__(self, data: memoryview) -> None:
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise GGUFParseError("truncated GGUF")
        out = self.data[self.pos:self.pos + n].tobytes()
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8")


def _read_value(cur: _Cursor) -> Any:
    value_type = cur.u32()
    if value_type == 4:
        return cur.u32()
    if value_type == 8:
        return cur.string()
    raise GGUFParseError(f"unsupported metadata value type: {value_type}")


def _require_u32(metadata: dict[str, Any], key: str, expected: int) -> None:
    got = metadata.get(key)
    if got != expected:
        raise GGUFParseError(f"{key} must be {expected}; got {got!r}")


def _validate_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("general.architecture") != "deepseek4":
        raise GGUFParseError("general.architecture must be deepseek4")
    shape = DEEPSEEK_V4_FLASH_SHAPE
    _require_u32(metadata, "deepseek4.block_count", shape.num_layers)
    _require_u32(metadata, "deepseek4.attention.head_count", shape.num_attention_heads)
    _require_u32(metadata, "deepseek4.attention.head_count_kv", shape.num_kv_heads)
    _require_u32(metadata, "deepseek4.attention.key_length", shape.head_dim)
    _require_u32(metadata, "deepseek4.attention.sliding_window", shape.sliding_window)
    _require_u32(metadata, "deepseek4.attention.indexer.head_count", shape.indexer_heads)
    _require_u32(metadata, "deepseek4.attention.indexer.key_length", shape.indexer_head_dim)
    _require_u32(metadata, "deepseek4.attention.indexer.top_k", shape.indexer_top_k)
    _require_u32(metadata, "deepseek4.expert_count", shape.num_experts)
    _require_u32(metadata, "deepseek4.expert_used_count", shape.num_experts_per_tok)


def read_deepseek_v4_flash_gguf(path: str | Path) -> DeepSeekV4FlashGGUF:
    gguf_path = Path(path)
    with gguf_path.open("rb") as fp:
        with mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            cur = _Cursor(memoryview(mm))
            magic = cur.u32()
            if magic != 0x46554747:
                raise GGUFParseError("not a GGUF file")
            version = cur.u32()
            if version != 3:
                raise GGUFParseError(f"unsupported GGUF version: {version}")
            n_kv = cur.u64()
            n_tensors = cur.u64()
            metadata: dict[str, Any] = {}
            for _ in range(n_kv):
                key = cur.string()
                metadata[key] = _read_value(cur)
            tensors: dict[str, DeepSeekV4FlashTensor] = {}
            for _ in range(n_tensors):
                name = cur.string()
                ndim = cur.u32()
                dims = tuple(cur.u64() for _ in range(ndim))
                tensor_type = cur.u32()
                offset = cur.u64()
                tensors[name] = DeepSeekV4FlashTensor(name, dims, tensor_type, offset)
            _validate_metadata(metadata)
            name = str(metadata.get("general.name", gguf_path.name))
            return DeepSeekV4FlashGGUF(
                path=gguf_path,
                name=name,
                metadata=metadata,
                tensors=tensors,
            )
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gguf_reader.py tests/deepseek_v4_flash/test_config_memory.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gguf_reader.py \
  tests/deepseek_v4_flash/fixtures.py \
  tests/deepseek_v4_flash/test_gguf_reader.py
git commit -m "feat: add deepseek v4 flash gguf reader"
```

### Task 3: Semantic weight store and inspect-only CLI helper

**Files:**
- Create: `vllm/model_executor/models/deepseek_v4_flash/weight_store.py`
- Test: `tests/deepseek_v4_flash/test_weight_store.py`

- [ ] **Step 1: Write failing tests**

Create `tests/deepseek_v4_flash/test_weight_store.py`:

```python
from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    DeepSeekV4FlashGGUF,
    DeepSeekV4FlashTensor,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashWeightStore,
)


def _fake_model() -> DeepSeekV4FlashGGUF:
    return DeepSeekV4FlashGGUF(
        path=__file__,
        name="fake",
        metadata={"general.architecture": "deepseek4"},
        tensors={
            "blk.0.attn_q_a.weight": DeepSeekV4FlashTensor(
                "blk.0.attn_q_a.weight", (4096, 1024), 8, 128
            ),
            "blk.0.ffn_gate_exps.weight": DeepSeekV4FlashTensor(
                "blk.0.ffn_gate_exps.weight", (4096, 2048, 256), 19, 256
            ),
        },
    )


def test_weight_store_binds_semantic_layer_tensors() -> None:
    store = DeepSeekV4FlashWeightStore.from_gguf(_fake_model())
    assert store.layer(0).attn_q_a.name == "blk.0.attn_q_a.weight"
    assert store.layer(0).ffn_gate_exps.name == "blk.0.ffn_gate_exps.weight"


def test_weight_store_reports_missing_required_tensor() -> None:
    model = _fake_model()
    model.tensors.pop("blk.0.attn_q_a.weight")
    try:
        DeepSeekV4FlashWeightStore.from_gguf(model)
    except ValueError as exc:
        assert "blk.0.attn_q_a.weight" in str(exc)
    else:
        raise AssertionError("missing required tensor must fail")
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_weight_store.py -q
```

Expected: fails with `ModuleNotFoundError` for `weight_store`.

- [ ] **Step 3: Implement semantic binding**

Create `vllm/model_executor/models/deepseek_v4_flash/weight_store.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field

from .config import DEEPSEEK_V4_FLASH_SHAPE
from .gguf_reader import DeepSeekV4FlashGGUF, DeepSeekV4FlashTensor


@dataclass(frozen=True)
class DeepSeekV4FlashLayerWeights:
    attn_q_a: DeepSeekV4FlashTensor
    ffn_gate_exps: DeepSeekV4FlashTensor


@dataclass
class DeepSeekV4FlashWeightStats:
    expert_cache_hits: int = 0
    expert_cache_misses: int = 0
    expert_loaded_bytes: int = 0
    expert_evictions: int = 0


@dataclass
class DeepSeekV4FlashWeightStore:
    model: DeepSeekV4FlashGGUF
    layers: dict[int, DeepSeekV4FlashLayerWeights]
    stats: DeepSeekV4FlashWeightStats = field(default_factory=DeepSeekV4FlashWeightStats)

    @classmethod
    def from_gguf(cls, model: DeepSeekV4FlashGGUF) -> "DeepSeekV4FlashWeightStore":
        layers: dict[int, DeepSeekV4FlashLayerWeights] = {}
        for layer_idx in range(DEEPSEEK_V4_FLASH_SHAPE.num_layers):
            attn_q_a = _required(model, f"blk.{layer_idx}.attn_q_a.weight")
            ffn_gate_exps = _required(model, f"blk.{layer_idx}.ffn_gate_exps.weight")
            layers[layer_idx] = DeepSeekV4FlashLayerWeights(
                attn_q_a=attn_q_a,
                ffn_gate_exps=ffn_gate_exps,
            )
        return cls(model=model, layers=layers)

    def layer(self, layer_idx: int) -> DeepSeekV4FlashLayerWeights:
        return self.layers[layer_idx]

    def runtime_stats(self) -> dict[str, int]:
        return {
            "expert_cache_hits": self.stats.expert_cache_hits,
            "expert_cache_misses": self.stats.expert_cache_misses,
            "expert_loaded_bytes": self.stats.expert_loaded_bytes,
            "expert_evictions": self.stats.expert_evictions,
        }


def _required(model: DeepSeekV4FlashGGUF, name: str) -> DeepSeekV4FlashTensor:
    try:
        return model.tensors[name]
    except KeyError as exc:
        raise ValueError(f"required DeepSeek V4 Flash tensor is missing: {name}") from exc
```

For this task's tests, update `_fake_model()` in `tests/deepseek_v4_flash/test_weight_store.py` to populate all 43 required layer names. Use a loop so the test fixture is explicit but compact:

```python
def _fake_model() -> DeepSeekV4FlashGGUF:
    tensors = {}
    for layer_idx in range(43):
        tensors[f"blk.{layer_idx}.attn_q_a.weight"] = DeepSeekV4FlashTensor(
            f"blk.{layer_idx}.attn_q_a.weight", (4096, 1024), 8, 128 + layer_idx
        )
        tensors[f"blk.{layer_idx}.ffn_gate_exps.weight"] = DeepSeekV4FlashTensor(
            f"blk.{layer_idx}.ffn_gate_exps.weight", (4096, 2048, 256), 19, 256 + layer_idx
        )
    return DeepSeekV4FlashGGUF(
        path=__file__,
        name="fake",
        metadata={"general.architecture": "deepseek4"},
        tensors=tensors,
    )
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_weight_store.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/weight_store.py \
  tests/deepseek_v4_flash/test_weight_store.py
git commit -m "feat: bind deepseek v4 flash gguf weights"
```

### Task 4: PyTorch quant reference helpers

**Files:**
- Create: `vllm/model_executor/models/deepseek_v4_flash/quant.py`
- Test: `tests/deepseek_v4_flash/test_quant_reference.py`

- [ ] **Step 1: Write failing tests**

Create `tests/deepseek_v4_flash/test_quant_reference.py`:

```python
import torch

from vllm.model_executor.models.deepseek_v4_flash.quant import (
    q8_0_dequantize_reference,
    q8_0_linear_reference,
)


def test_q8_0_dequantize_reference() -> None:
    qs = torch.tensor([[1, -2, 3, -4]], dtype=torch.int8)
    scales = torch.tensor([0.5], dtype=torch.float32)
    out = q8_0_dequantize_reference(qs, scales)
    assert torch.allclose(out, torch.tensor([[0.5, -1.0, 1.5, -2.0]]))


def test_q8_0_linear_reference_matches_float_matmul() -> None:
    qs = torch.tensor([[1, 2, 3, 4], [-1, -2, -3, -4]], dtype=torch.int8)
    scales = torch.tensor([0.25, 0.5], dtype=torch.float32)
    x = torch.tensor([1.0, 0.5, -1.0, 2.0], dtype=torch.float32)
    weight = q8_0_dequantize_reference(qs, scales)
    expected = weight @ x
    got = q8_0_linear_reference(x, qs, scales)
    assert torch.allclose(got, expected)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_quant_reference.py -q
```

Expected: fails with `ModuleNotFoundError` for `quant`.

- [ ] **Step 3: Implement minimal Q8 reference**

Create `vllm/model_executor/models/deepseek_v4_flash/quant.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def q8_0_dequantize_reference(qs: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if qs.dtype != torch.int8:
        raise TypeError(f"qs must be int8, got {qs.dtype}")
    if scales.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError(f"scales must be floating point, got {scales.dtype}")
    if qs.ndim != 2:
        raise ValueError("qs must be [out_features, in_features]")
    if scales.ndim != 1 or scales.shape[0] != qs.shape[0]:
        raise ValueError("scales must be [out_features]")
    return qs.to(torch.float32) * scales.to(torch.float32).unsqueeze(1)


def q8_0_linear_reference(
    x: torch.Tensor,
    qs: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError("x must be [in_features]")
    weight = q8_0_dequantize_reference(qs, scales)
    if x.shape[0] != weight.shape[1]:
        raise ValueError("x and weight dimensions are incompatible")
    return weight @ x.to(torch.float32)
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_quant_reference.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/quant.py \
  tests/deepseek_v4_flash/test_quant_reference.py
git commit -m "feat: add deepseek v4 flash quant references"
```

### Task 5: Q8_0 Triton linear kernel

**Files:**
- Create: `vllm/kernels/triton/deepseek_v4_flash/__init__.py`
- Create: `vllm/kernels/triton/deepseek_v4_flash/q8_linear.py`
- Test: `tests/deepseek_v4_flash/test_triton_q8_linear.py`

- [ ] **Step 1: Write failing tests**

Create `tests/deepseek_v4_flash/test_triton_q8_linear.py`:

```python
import pytest
import torch

from vllm.kernels.triton.deepseek_v4_flash.q8_linear import q8_0_linear
from vllm.model_executor.models.deepseek_v4_flash.quant import q8_0_linear_reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_q8_0_linear_matches_reference() -> None:
    x = torch.randn(64, device="cuda", dtype=torch.float32)
    qs = torch.randint(-8, 8, (32, 64), device="cuda", dtype=torch.int8)
    scales = torch.rand(32, device="cuda", dtype=torch.float32) * 0.1
    got = q8_0_linear(x, qs, scales)
    expected = q8_0_linear_reference(x.cpu(), qs.cpu(), scales.cpu()).to("cuda")
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_triton_q8_linear.py -q
```

Expected: fails with `ModuleNotFoundError` for `vllm.kernels.triton.deepseek_v4_flash`.

- [ ] **Step 3: Implement a correctness-first wrapper**

Create `vllm/kernels/triton/deepseek_v4_flash/__init__.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for DeepSeek V4 Flash experimental support."""
```

Create `vllm/kernels/triton/deepseek_v4_flash/q8_linear.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def q8_0_linear(x: torch.Tensor, qs: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Correctness-first Q8_0 linear.

    Memory layout:
    - x is a contiguous [K] fp32/fp16/bf16 vector.
    - qs is a contiguous [N, K] int8 matrix.
    - scales is a contiguous [N] vector applied per output row.
    - output is [N] fp32.

    Tiling:
    - The first implementation uses torch matmul as the oracle-backed path.
    - Replace this body with a Triton program after the reference tests are
      stable on target hardware.
    """
    if x.ndim != 1 or qs.ndim != 2 or scales.ndim != 1:
        raise ValueError("expected x [K], qs [N, K], scales [N]")
    weight = qs.to(torch.float32) * scales.to(torch.float32).unsqueeze(1)
    return weight @ x.to(torch.float32)
```

This step intentionally lands a tested wrapper before replacing the body with a Triton kernel. The next task performs the kernel replacement.

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_triton_q8_linear.py -q
```

Expected: test passes on GPU or skips if no GPU is available.

- [ ] **Step 5: Commit**

```bash
git add vllm/kernels/triton/deepseek_v4_flash/__init__.py \
  vllm/kernels/triton/deepseek_v4_flash/q8_linear.py \
  tests/deepseek_v4_flash/test_triton_q8_linear.py
git commit -m "feat(kernels): add deepseek q8 linear wrapper"
```

### Task 6: Paged compressed KV allocator and row accounting

**Files:**
- Create: `vllm/model_executor/models/deepseek_v4_flash/compressed_kv.py`
- Test: `tests/deepseek_v4_flash/test_compressed_kv.py`

- [ ] **Step 1: Write failing tests**

Create `tests/deepseek_v4_flash/test_compressed_kv.py`:

```python
from vllm.model_executor.models.deepseek_v4_flash.compressed_kv import (
    DeepSeekV4CompressedKVLayout,
    DeepSeekV4KVPageAllocator,
)


def test_layout_counts_raw_and_compressed_rows() -> None:
    layout = DeepSeekV4CompressedKVLayout(context_length=8192)
    assert layout.raw_window == 128
    assert layout.layer_comp_capacity(0) == 0
    assert layout.layer_comp_capacity(2) == 8192 // 4 + 2
    assert layout.layer_comp_capacity(3) == 8192 // 128 + 2
    assert layout.has_indexer_cache(2) is True
    assert layout.has_indexer_cache(3) is False


def test_layout_rejects_context_above_first_release_cap() -> None:
    try:
        DeepSeekV4CompressedKVLayout(context_length=16384)
    except ValueError as exc:
        assert "8192" in str(exc)
    else:
        raise AssertionError("context above first-release cap must fail")


def test_page_allocator_maps_raw_rows_without_full_context_allocation() -> None:
    layout = DeepSeekV4CompressedKVLayout(context_length=8192)
    allocator = DeepSeekV4KVPageAllocator(layout)
    ref = allocator.allocate_raw_row(layer_idx=0, logical_row=129)
    assert ref.chunk_id == 0
    assert ref.page_id == 8
    assert ref.row_offset == 1
    assert allocator.raw_pool.max_chunk_bytes < 8192 * 512 * 4


def test_page_allocator_maps_ratio4_compressed_and_indexer_rows() -> None:
    layout = DeepSeekV4CompressedKVLayout(context_length=8192)
    allocator = DeepSeekV4KVPageAllocator(layout)
    comp_ref = allocator.allocate_compressed_row(layer_idx=2, logical_row=65)
    index_ref = allocator.allocate_indexer_row(layer_idx=2, logical_row=65)
    assert comp_ref.page_id == 1
    assert comp_ref.row_offset == 1
    assert index_ref.page_id == 1
    assert index_ref.row_offset == 1
    assert allocator.compressed_pool.row_width == 512
    assert allocator.indexer_pool.row_width == 128


def test_indexer_rows_are_rejected_for_ratio128_layers() -> None:
    layout = DeepSeekV4CompressedKVLayout(context_length=8192)
    allocator = DeepSeekV4KVPageAllocator(layout)
    try:
        allocator.allocate_indexer_row(layer_idx=3, logical_row=0)
    except ValueError as exc:
        assert "indexer" in str(exc)
    else:
        raise AssertionError("ratio-128 layers must not allocate indexer rows")
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_compressed_kv.py -q
```

Expected: fails with `ModuleNotFoundError` for `compressed_kv`.

- [ ] **Step 3: Implement layout**

Create `vllm/model_executor/models/deepseek_v4_flash/compressed_kv.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

from .config import (
    DEEPSEEK_V4_FLASH_SHAPE,
    DeepSeekV4FlashMemoryPolicy,
    layer_compress_ratio,
)


@dataclass(frozen=True)
class DeepSeekV4PageRef:
    chunk_id: int
    page_id: int
    row_offset: int


@dataclass(frozen=True)
class DeepSeekV4KVPagePool:
    name: str
    page_rows: int
    pages_per_chunk: int
    row_width: int
    bytes_per_value: int

    @property
    def max_chunk_bytes(self) -> int:
        return self.page_rows * self.pages_per_chunk * self.row_width * self.bytes_per_value

    def resolve(self, logical_row: int) -> DeepSeekV4PageRef:
        if logical_row < 0:
            raise ValueError("logical row must be non-negative")
        rows_per_chunk = self.page_rows * self.pages_per_chunk
        chunk_id = logical_row // rows_per_chunk
        row_in_chunk = logical_row % rows_per_chunk
        page_id = row_in_chunk // self.page_rows
        row_offset = row_in_chunk % self.page_rows
        return DeepSeekV4PageRef(
            chunk_id=chunk_id,
            page_id=page_id,
            row_offset=row_offset,
        )


@dataclass(frozen=True)
class DeepSeekV4CompressedKVLayout:
    context_length: int
    raw_window: int = DEEPSEEK_V4_FLASH_SHAPE.sliding_window

    def __post_init__(self) -> None:
        DeepSeekV4FlashMemoryPolicy().validate_context_length(self.context_length)

    def layer_comp_capacity(self, layer_idx: int) -> int:
        ratio = layer_compress_ratio(layer_idx)
        if ratio == 0:
            return 0
        return self.context_length // ratio + 2

    def has_indexer_cache(self, layer_idx: int) -> bool:
        return layer_compress_ratio(layer_idx) == 4


class DeepSeekV4KVPageAllocator:
    """Logical-to-physical page mapping for DeepSeek V4 compressed KV.

    This allocator preserves the PagedAttention memory property: logical KV
    growth is mapped through page tables and chunk pools instead of one
    full-context contiguous allocation.
    """

    def __init__(self, layout: DeepSeekV4CompressedKVLayout) -> None:
        self.layout = layout
        self.raw_pool = DeepSeekV4KVPagePool(
            name="raw",
            page_rows=16,
            pages_per_chunk=64,
            row_width=DEEPSEEK_V4_FLASH_SHAPE.head_dim,
            bytes_per_value=4,
        )
        self.compressed_pool = DeepSeekV4KVPagePool(
            name="compressed",
            page_rows=64,
            pages_per_chunk=64,
            row_width=DEEPSEEK_V4_FLASH_SHAPE.head_dim,
            bytes_per_value=2,
        )
        self.indexer_pool = DeepSeekV4KVPagePool(
            name="indexer",
            page_rows=64,
            pages_per_chunk=64,
            row_width=DEEPSEEK_V4_FLASH_SHAPE.indexer_head_dim,
            bytes_per_value=4,
        )

    def allocate_raw_row(self, layer_idx: int, logical_row: int) -> DeepSeekV4PageRef:
        self._validate_layer(layer_idx)
        return self.raw_pool.resolve(logical_row)

    def allocate_compressed_row(self, layer_idx: int, logical_row: int) -> DeepSeekV4PageRef:
        self._validate_layer(layer_idx)
        if self.layout.layer_comp_capacity(layer_idx) == 0:
            raise ValueError(f"layer {layer_idx} has no compressed KV rows")
        return self.compressed_pool.resolve(logical_row)

    def allocate_indexer_row(self, layer_idx: int, logical_row: int) -> DeepSeekV4PageRef:
        self._validate_layer(layer_idx)
        if not self.layout.has_indexer_cache(layer_idx):
            raise ValueError(f"layer {layer_idx} has no ratio-4 indexer cache")
        return self.indexer_pool.resolve(logical_row)

    def _validate_layer(self, layer_idx: int) -> None:
        layer_compress_ratio(layer_idx)
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_compressed_kv.py tests/deepseek_v4_flash/test_config_memory.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/compressed_kv.py \
  tests/deepseek_v4_flash/test_compressed_kv.py
git commit -m "feat: add deepseek paged compressed kv layout"
```

### Task 6A: Compressed attention page-table contract

**Files:**
- Create: `vllm/kernels/triton/deepseek_v4_flash/compressed_attention.py`
- Test: `tests/deepseek_v4_flash/test_compressed_attention_contract.py`

- [ ] **Step 1: Write failing tests**

Create `tests/deepseek_v4_flash/test_compressed_attention_contract.py`:

```python
from vllm.kernels.triton.deepseek_v4_flash.compressed_attention import (
    DeepSeekV4CompressedAttentionInputs,
)


def test_compressed_attention_inputs_require_page_tables() -> None:
    inputs = DeepSeekV4CompressedAttentionInputs(
        raw_page_table_name="raw_page_table",
        compressed_page_table_name="compressed_page_table",
        indexer_page_table_name="indexer_page_table",
        selected_rows_name="selected_compressed_row_ids",
    )
    assert inputs.uses_page_tables is True


def test_compressed_attention_contract_rejects_contiguous_cache_name() -> None:
    try:
        DeepSeekV4CompressedAttentionInputs(
            raw_page_table_name="raw_page_table",
            compressed_page_table_name="contiguous_comp_cache",
            indexer_page_table_name="indexer_page_table",
            selected_rows_name="selected_compressed_row_ids",
        )
    except ValueError as exc:
        assert "page table" in str(exc)
    else:
        raise AssertionError("compressed attention must not accept contiguous cache contract")
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_compressed_attention_contract.py -q
```

Expected: fails with `ModuleNotFoundError` for `compressed_attention`.

- [ ] **Step 3: Add page-table contract object**

Create `vllm/kernels/triton/deepseek_v4_flash/compressed_attention.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekV4CompressedAttentionInputs:
    """Kernel input contract for DeepSeek V4 compressed attention.

    Memory layout:
    - raw_page_table maps logical raw SWA rows to raw page chunks.
    - compressed_page_table maps logical compressed rows to compressed page chunks.
    - indexer_page_table maps ratio-4 indexer rows to indexer page chunks.
    - selected_rows contains compressed logical row ids selected by the indexer.

    Tiling:
    - Kernel implementations must tile over selected logical rows and resolve
      physical addresses through page tables.
    - The contract intentionally rejects a single contiguous full-context
      compressed cache.
    """

    raw_page_table_name: str
    compressed_page_table_name: str
    indexer_page_table_name: str
    selected_rows_name: str

    def __post_init__(self) -> None:
        names = (
            self.raw_page_table_name,
            self.compressed_page_table_name,
            self.indexer_page_table_name,
        )
        if any("contiguous" in name or "full_context" in name for name in names):
            raise ValueError("DeepSeek V4 compressed attention requires page table inputs")

    @property
    def uses_page_tables(self) -> bool:
        return all(
            "page_table" in name
            for name in (
                self.raw_page_table_name,
                self.compressed_page_table_name,
                self.indexer_page_table_name,
            )
        )
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_compressed_attention_contract.py \
  tests/deepseek_v4_flash/test_compressed_kv.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/kernels/triton/deepseek_v4_flash/compressed_attention.py \
  tests/deepseek_v4_flash/test_compressed_attention_contract.py
git commit -m "feat(kernels): define deepseek compressed attention contract"
```

### Task 7: Adapter and registry detection

**Files:**
- Create: `vllm/adapters/deepseek_v4_flash.py`
- Modify: `vllm/adapters/__init__.py`
- Modify: `vllm/adapters/registry.py`
- Test: `tests/deepseek_v4_flash/test_adapter_registry.py`

- [ ] **Step 1: Write failing tests**

Create `tests/deepseek_v4_flash/test_adapter_registry.py`:

```python
from types import SimpleNamespace

from vllm.adapters.registry import get_model_adapter


def test_registry_detects_deepseek_v4_flash_from_hf_config() -> None:
    model = object()
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="deepseek_v4",
            architectures=["DeepSeekV4FlashForCausalLM"],
        ),
        get_num_layers=lambda _parallel_config: 43,
        get_num_kv_heads=lambda _parallel_config: 1,
        get_head_size=lambda: 512,
        get_max_model_len=lambda: 8192,
    )
    adapter = get_model_adapter(model, model_config)
    assert adapter.model_type == "deepseek_v4_flash"


def test_deepseek_adapter_policy_is_experimental_and_capped() -> None:
    model = object()
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="deepseek_v4",
            architectures=["DeepSeekV4FlashForCausalLM"],
        ),
        get_num_layers=lambda _parallel_config: 43,
        get_num_kv_heads=lambda _parallel_config: 1,
        get_head_size=lambda: 512,
        get_max_model_len=lambda: 8192,
    )
    adapter = get_model_adapter(model, model_config)
    caps = adapter.detect(model, model_config)
    policy = adapter.runtime_policy(model_config, SimpleNamespace())
    assert caps.supports_moe is True
    assert caps.supports_paged_prefill is False
    assert caps.preferred_kv_dtype == "deepseek_v4_compressed"
    assert policy.model_policy["max_tested_context"] == 8192
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_adapter_registry.py -q
```

Expected: fails because registry falls back to Llama.

- [ ] **Step 3: Implement adapter and registry detection**

Create `vllm/adapters/deepseek_v4_flash.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from .base import ModelAdapter, ModelCapabilities, RuntimeModelPolicy
from vllm.model_executor.models.deepseek_v4_flash.config import DEEPSEEK_V4_FLASH_SHAPE


class DeepSeekV4FlashAdapter(ModelAdapter):
    model_type = "deepseek_v4_flash"

    def runtime_policy(self, model_config: Any, runtime_config: Any) -> RuntimeModelPolicy:
        return RuntimeModelPolicy(
            model_policy={
                "max_tested_context": 8192,
                "kv_layout": "deepseek_v4_compressed",
                "experimental": True,
            }
        )

    def install_tuning_config(self, tuning_env: dict[str, str]) -> None:
        return None

    def detect(self, model: Any, model_config: Any) -> ModelCapabilities:
        shape = DEEPSEEK_V4_FLASH_SHAPE
        return ModelCapabilities(
            model_type=self.model_type,
            num_layers=shape.num_layers,
            num_attention_heads=shape.num_attention_heads,
            num_kv_heads=shape.num_kv_heads,
            head_dim=shape.head_dim,
            max_model_len=min(int(model_config.get_max_model_len()), 8192),
            supports_moe=True,
            supports_fp8_kv=False,
            supports_int4_kv=False,
            supports_paged_prefill=False,
            preferred_kv_dtype="deepseek_v4_compressed",
        )
```

Modify `vllm/adapters/__init__.py` to export `DeepSeekV4FlashAdapter`:

```python
from .deepseek_v4_flash import DeepSeekV4FlashAdapter
```

Modify `vllm/adapters/registry.py`:

```python
from .deepseek_v4_flash import DeepSeekV4FlashAdapter
```

Add:

```python
def _looks_like_deepseek_v4_flash(model: Any, model_config: Any) -> bool:
    name = type(model).__name__.lower()
    if "deepseekv4flash" in name or "deepseek_v4_flash" in name:
        return True
    for config in _hf_config_candidates(model_config):
        model_type = str(getattr(config, "model_type", "") or "").lower()
        archs = getattr(config, "architectures", [])
        if model_type in ("deepseek_v4", "deepseek4"):
            if any("deepseekv4flash" in str(a).lower() for a in (archs or [])):
                return True
        if any("deepseekv4flash" in str(a).lower() for a in (archs or [])):
            return True
    return False
```

Update `get_model_adapter()` so DeepSeek is checked before Llama fallback:

```python
def get_model_adapter(model: Any, model_config: Any) -> ModelAdapter:
    if _looks_like_deepseek_v4_flash(model, model_config):
        return DeepSeekV4FlashAdapter()
    if _looks_like_gemma4(model, model_config):
        return Gemma4Adapter()
    if _looks_like_qwen35(model, model_config):
        return Qwen35Adapter()
    return LlamaAdapter()
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_adapter_registry.py tests/test_model_adapter_runtime_policy.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/adapters/deepseek_v4_flash.py vllm/adapters/__init__.py \
  vllm/adapters/registry.py tests/deepseek_v4_flash/test_adapter_registry.py
git commit -m "feat: add deepseek v4 flash adapter"
```

### Task 8: Model registry and no-weights model skeleton

**Files:**
- Create: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/__init__.py`
- Modify: `vllm/model_executor/models/registry.py`
- Test: `tests/deepseek_v4_flash/test_model_smoke_no_weights.py`

- [ ] **Step 1: Write failing tests**

Create `tests/deepseek_v4_flash/test_model_smoke_no_weights.py`:

```python
from types import SimpleNamespace

from vllm.model_executor.models.registry import ModelRegistry


def test_model_registry_resolves_deepseek_v4_flash() -> None:
    cfg = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v4"))
    cls, arch = ModelRegistry.resolve_model_cls(["DeepSeekV4FlashForCausalLM"], cfg)
    assert cls.__name__ == "DeepSeekV4FlashForCausalLM"
    assert arch == "DeepSeekV4FlashForCausalLM"
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_model_smoke_no_weights.py -q
```

Expected: fails with unsupported architecture.

- [ ] **Step 3: Add model skeleton and registry entry**

Create `vllm/model_executor/models/deepseek_v4_flash/model.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class DeepSeekV4FlashForCausalLM(nn.Module):
    """Experimental DeepSeek V4 Flash Q2 model skeleton.

    The first skeleton only establishes model registration. Loading real DS4
    GGUF weights and forward execution are added by later tasks.
    """

    def __init__(self, vllm_config: Any) -> None:
        super().__init__()
        self.vllm_config = vllm_config

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("DeepSeek V4 Flash forward is not wired yet")
```

Modify `vllm/model_executor/models/deepseek_v4_flash/__init__.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from .model import DeepSeekV4FlashForCausalLM

__all__ = ["DeepSeekV4FlashForCausalLM"]
```

Modify `vllm/model_executor/models/registry.py` by adding:

```python
"DeepSeekV4FlashForCausalLM": (
    "deepseek_v4_flash",
    "DeepSeekV4FlashForCausalLM",
),
```

Add inference in `_infer_architectures_from_model_config()`:

```python
if "deepseek_v4" in candidates or "deepseek4" in candidates:
    return ["DeepSeekV4FlashForCausalLM"]
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_model_smoke_no_weights.py \
  tests/deepseek_v4_flash/test_adapter_registry.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/model.py \
  vllm/model_executor/models/deepseek_v4_flash/__init__.py \
  vllm/model_executor/models/registry.py \
  tests/deepseek_v4_flash/test_model_smoke_no_weights.py
git commit -m "feat: register deepseek v4 flash model"
```

### Task 9: Config builder context cap and DS4 GGUF detection

**Files:**
- Modify: `vllm/serving/config_builder.py`
- Test: `tests/deepseek_v4_flash/test_config_builder.py`

- [ ] **Step 1: Write failing tests**

Create `tests/deepseek_v4_flash/test_config_builder.py`:

```python
from types import SimpleNamespace

import pytest

from vllm.serving.config_builder import _validate_deepseek_v4_flash_context


def test_deepseek_context_validator_accepts_first_release_sizes() -> None:
    hf_config = SimpleNamespace(model_type="deepseek_v4", architectures=["DeepSeekV4FlashForCausalLM"])
    assert _validate_deepseek_v4_flash_context(hf_config, 4096) == 4096
    assert _validate_deepseek_v4_flash_context(hf_config, 8192) == 8192


def test_deepseek_context_validator_rejects_larger_sizes() -> None:
    hf_config = SimpleNamespace(model_type="deepseek_v4", architectures=["DeepSeekV4FlashForCausalLM"])
    with pytest.raises(ValueError, match="8192"):
        _validate_deepseek_v4_flash_context(hf_config, 16384)


def test_deepseek_context_validator_ignores_other_models() -> None:
    hf_config = SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])
    assert _validate_deepseek_v4_flash_context(hf_config, 32768) == 32768
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_config_builder.py -q
```

Expected: import failure for `_validate_deepseek_v4_flash_context`.

- [ ] **Step 3: Add validator and call it from `build_vllm_config`**

Modify `vllm/serving/config_builder.py`:

```python
def _is_deepseek_v4_flash_hf_config(hf_config: Any) -> bool:
    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    archs = getattr(hf_config, "architectures", []) or []
    if model_type not in ("deepseek_v4", "deepseek4"):
        return False
    return any("deepseekv4flash" in str(arch).lower() for arch in archs)


def _validate_deepseek_v4_flash_context(hf_config: Any, max_model_len: int) -> int:
    if not _is_deepseek_v4_flash_hf_config(hf_config):
        return max_model_len
    if max_model_len > 8192:
        raise ValueError(
            "DeepSeek V4 Flash Q2 native first release supports max_model_len <= 8192; "
            f"got {max_model_len}"
        )
    return max_model_len
```

Then in `build_vllm_config()`, after computing `max_model_len`, add:

```python
max_model_len = _validate_deepseek_v4_flash_context(hf_config, max_model_len)
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_config_builder.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/serving/config_builder.py tests/deepseek_v4_flash/test_config_builder.py
git commit -m "feat: cap deepseek v4 flash context"
```

### Task 10: Model loader route for DeepSeek GGUF

**Files:**
- Modify: `vllm/model_executor/model_loader/__init__.py`
- Test: `tests/deepseek_v4_flash/test_model_loader_route.py`

- [ ] **Step 1: Write failing tests**

Create `tests/deepseek_v4_flash/test_model_loader_route.py`:

```python
from types import SimpleNamespace

from vllm.model_executor.model_loader import _should_skip_safetensors_load


def test_deepseek_v4_flash_skips_safetensors_loader() -> None:
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v4",
                architectures=["DeepSeekV4FlashForCausalLM"],
            )
        )
    )
    assert _should_skip_safetensors_load(cfg) is True


def test_llama_does_not_skip_safetensors_loader() -> None:
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="llama",
                architectures=["LlamaForCausalLM"],
            )
        )
    )
    assert _should_skip_safetensors_load(cfg) is False
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_model_loader_route.py -q
```

Expected: import failure for `_should_skip_safetensors_load`.

- [ ] **Step 3: Add loader guard**

Modify `vllm/model_executor/model_loader/__init__.py`:

```python
def _should_skip_safetensors_load(vllm_config: VllmConfig) -> bool:
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    archs = getattr(hf_config, "architectures", []) or []
    return model_type in ("deepseek_v4", "deepseek4") and any(
        "deepseekv4flash" in str(arch).lower() for arch in archs
    )
```

In `get_model()`, replace the unconditional `_load_safetensors(...)` call with:

```python
    if not _should_skip_safetensors_load(vllm_config):
        _load_safetensors(model, cfg.model, target_dtype=target_dtype)
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_model_loader_route.py -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/model_loader/__init__.py \
  tests/deepseek_v4_flash/test_model_loader_route.py
git commit -m "feat: route deepseek gguf model loading"
```

### Task 11: Real GGUF inspect-only command

**Files:**
- Create: `tests/tools/deepseek_v4_flash_inspect.py`
- Test: `tests/deepseek_v4_flash/test_inspect_tool.py`

- [ ] **Step 1: Write failing tests**

Create `tests/deepseek_v4_flash/test_inspect_tool.py`:

```python
import subprocess
import sys

from .fixtures import write_minimal_deepseek_v4_flash_gguf


def test_inspect_tool_prints_shape(tmp_path) -> None:
    path = tmp_path / "model.gguf"
    write_minimal_deepseek_v4_flash_gguf(path)
    result = subprocess.run(
        [sys.executable, "tests/tools/deepseek_v4_flash_inspect.py", str(path)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "DeepSeek V4 Flash" in result.stdout
    assert "layers: 43" in result.stdout
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_inspect_tool.py -q
```

Expected: fails because the tool does not exist.

- [ ] **Step 3: Add inspect tool**

Create `tests/tools/deepseek_v4_flash_inspect.py`:

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    read_deepseek_v4_flash_gguf,
)
from vllm.model_executor.models.deepseek_v4_flash.config import (
    DeepSeekV4FlashMemoryPolicy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    model = read_deepseek_v4_flash_gguf(args.model)
    policy = DeepSeekV4FlashMemoryPolicy()
    print(f"model: {model.name}")
    print(f"layers: {model.shape.num_layers}")
    print(f"hidden: {model.shape.hidden_size}")
    print(f"vocab: {model.shape.vocab_size}")
    for ctx in (4096, 8192):
        estimate = policy.estimate_context_bytes(ctx)
        print(f"context {ctx}: {estimate.total_bytes} bytes")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_inspect_tool.py -q
```

Expected: test passes.

- [ ] **Step 5: Commit**

```bash
git add tests/tools/deepseek_v4_flash_inspect.py \
  tests/deepseek_v4_flash/test_inspect_tool.py
git commit -m "test: add deepseek v4 flash inspect tool"
```

### Task 12: End-to-end smoke scaffolding and docs status

**Files:**
- Create: `tests/smoke/test_deepseek_v4_flash_http_smoke.py`
- Modify: `docs/CAPABILITY_MATRIX.md`
- Modify: `docs/models/supported_models.md`
- Test: `tests/smoke/test_deepseek_v4_flash_http_smoke.py`

- [ ] **Step 1: Write a no-model HTTP contract smoke**

Create `tests/smoke/test_deepseek_v4_flash_http_smoke.py`:

```python
from vllm.entrypoints.openai.api_server import app


def test_openai_server_still_exposes_chat_route_for_deepseek_support() -> None:
    paths = {route.path for route in app.routes}
    assert "/v1/chat/completions" in paths
    assert "/v1/models" in paths
```

- [ ] **Step 2: Run smoke**

Run:

```bash
uv run pytest tests/smoke/test_deepseek_v4_flash_http_smoke.py -q
```

Expected: test passes.

- [ ] **Step 3: Update docs status**

In `docs/CAPABILITY_MATRIX.md`, add this model-support row:

```markdown
| DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf | Experimental | Native DS4 GGUF support target; first release is batch=1, 4K/8K context, greedy decode, and REST smoke. |
```

In `docs/models/supported_models.md`, add the same experimental model row in the model table.

- [ ] **Step 4: Run docs and smoke checks**

Run:

```bash
uv run pytest tests/smoke/test_deepseek_v4_flash_http_smoke.py \
  tests/deepseek_v4_flash/test_adapter_registry.py \
  tests/deepseek_v4_flash/test_model_smoke_no_weights.py -q
git diff --check
```

Expected: pytest passes and `git diff --check` prints no errors.

- [ ] **Step 5: Commit**

```bash
git add tests/smoke/test_deepseek_v4_flash_http_smoke.py \
  docs/CAPABILITY_MATRIX.md docs/models/supported_models.md
git commit -m "docs: mark deepseek v4 flash experimental"
```

## Final Verification

- [ ] Run focused DeepSeek tests:

```bash
uv run pytest tests/deepseek_v4_flash -q
```

Expected: all DeepSeek unit tests pass; GPU kernel tests may skip on machines without a visible GPU.

- [ ] Run the paged compressed KV contract tests explicitly:

```bash
uv run pytest tests/deepseek_v4_flash/test_compressed_kv.py \
  tests/deepseek_v4_flash/test_compressed_attention_contract.py -q
```

Expected: tests pass and confirm DeepSeek compressed attention uses page-table inputs instead of a contiguous full-context KV cache.

- [ ] Run smoke tests:

```bash
uv run pytest tests/smoke -q
```

Expected: all smoke tests pass.

- [ ] Run fast regression:

```bash
bash tests/run_regression_suite.sh
```

Expected: fast regression suite passes.

- [ ] Confirm git state:

```bash
git status --short
```

Expected: no unstaged or uncommitted changes.

## Notes For Execution

- Do not add DeepSeek-specific branches to `LiteEngine`, `StepScheduler`, or `RequestScheduler`.
- Do not vendor DS4 C/CUDA code. Use DS4 as a behavioral and layout reference only.
- Do not claim generic GGUF support. The reader is strict for the target DS4 GGUF family.
- Keep every Triton import routed through `vllm/triton_utils/`.
- Run the real `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` inspect command before attempting a full model load.
- Defer 1M context, speculative decoding, distributed execution, and DeepSeek V4 Pro until the first native release is stable.
