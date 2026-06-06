# DeepSeek V4 Flash Real Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the downloaded target GGUF into working batch=1, context 4K/8K, greedy DeepSeek V4 Flash inference callable through the existing OpenAI-compatible REST path.

**Architecture:** Keep DeepSeek-specific logic inside `vllm/model_executor/models/deepseek_v4_flash/` and `vllm/kernels/triton/deepseek_v4_flash/`. The lite engine, schedulers, and REST entrypoints should remain model-family agnostic; the DeepSeek model owns GGUF weight binding, quantized matvec, compressed KV, attention, MoE routing, and logits.

**Tech Stack:** Python 3.12, `uv`, PyTorch runtime tensor logic, Triton through `vllm/triton_utils/`, GGUF mmap reader, existing lite engine/OpenAI REST server.

---

## Current State

- Target model exists at `models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.
- `vllm/model_executor/models/deepseek_v4_flash/model.py` still raises `NotImplementedError` in `forward()`.
- Loader can construct an inspect model and attach a `DeepSeekV4FlashWeightStore`.
- GGUF reader and diagnostics exist, but semantic bindings are still minimal.
- Q8_0 reference decode exists; IQ2_XXS and Q2_K are not wired.
- REST smoke currently checks routes only, not real generation.

## File Structure

- Modify `vllm/model_executor/models/deepseek_v4_flash/weight_store.py`: bind real tensor names into per-layer semantic tables and expose payload views.
- Modify `vllm/model_executor/models/deepseek_v4_flash/quant.py`: add CPU reference decoders for target GGUF quant formats.
- Create `vllm/model_executor/models/deepseek_v4_flash/layers.py`: small PyTorch reference layer components used before Triton optimization.
- Create `vllm/model_executor/models/deepseek_v4_flash/attention.py`: batch=1 raw SWA plus compressed attention reference path.
- Create `vllm/model_executor/models/deepseek_v4_flash/moe.py`: batch=1 top-6 routed MoE reference path with bounded expert staging.
- Modify `vllm/model_executor/models/deepseek_v4_flash/compressed_kv.py`: add append/read APIs used by forward.
- Modify `vllm/model_executor/models/deepseek_v4_flash/model.py`: implement prefill/decode forward contract for batch=1 greedy.
- Modify `tests/tools/deepseek_v4_flash_inspect.py`: print enough real tensor mapping to debug the target file.
- Add tests under `tests/deepseek_v4_flash/` for semantic binding, quant references, single-layer execution, KV behavior, forward smoke, and REST smoke.

---

### Task 1: Real GGUF Audit And Semantic Binding

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/weight_store.py`
- Modify: `tests/tools/deepseek_v4_flash_inspect.py`
- Test: `tests/deepseek_v4_flash/test_weight_store_real_mapping.py`

- [ ] **Step 1: Write the failing semantic mapping test**

Create `tests/deepseek_v4_flash/test_weight_store_real_mapping.py`:

```python
from pathlib import Path

import pytest

from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    open_deepseek_v4_flash_weight_store,
)


MODEL = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


@pytest.mark.skipif(not MODEL.exists(), reason="target DeepSeek GGUF not downloaded")
def test_real_gguf_binds_required_layer_families() -> None:
    with open_deepseek_v4_flash_weight_store(MODEL) as store:
        bindings = store.bindings

        assert bindings.token_embedding.name == "token_embd.weight"
        assert len(bindings.layers) == store.model.shape.num_layers
        assert bindings.output_norm is not None
        assert bindings.output_head is not None

        layer0 = bindings.layers[0]
        assert layer0.attention_query is not None
        assert layer0.attention_key is not None
        assert layer0.attention_value is not None
        assert layer0.attention_output is not None

        moe_layers = [layer for layer in bindings.layers if layer.routed_experts]
        assert moe_layers
        assert all(len(layer.routed_experts) == store.model.shape.num_experts for layer in moe_layers)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_weight_store_real_mapping.py -q
```

Expected: fail because `bindings.layers`, `output_norm`, `output_head`, and routed expert binding do not exist yet.

- [ ] **Step 3: Add explicit binding dataclasses**

In `weight_store.py`, add:

```python
@dataclass(frozen=True)
class DeepSeekV4FlashExpertTensors:
    gate: DeepSeekV4FlashTensor
    up: DeepSeekV4FlashTensor
    down: DeepSeekV4FlashTensor


@dataclass(frozen=True)
class DeepSeekV4FlashLayerTensors:
    layer_idx: int
    attention_query: DeepSeekV4FlashTensor | None
    attention_key: DeepSeekV4FlashTensor | None
    attention_value: DeepSeekV4FlashTensor | None
    attention_output: DeepSeekV4FlashTensor | None
    attention_norm: DeepSeekV4FlashTensor | None
    ffn_norm: DeepSeekV4FlashTensor | None
    router: DeepSeekV4FlashTensor | None
    shared_experts: tuple[DeepSeekV4FlashTensor, ...]
    routed_experts: dict[int, DeepSeekV4FlashExpertTensors]
```

Extend `DeepSeekV4FlashSemanticBindings` with:

```python
layers: tuple[DeepSeekV4FlashLayerTensors, ...]
output_norm: DeepSeekV4FlashTensor | None
output_head: DeepSeekV4FlashTensor | None
```

- [ ] **Step 4: Implement name-pattern binding from real GGUF names**

Implement helper functions that inspect `model.tensors` and bind only known patterns:

```python
def _tensor_or_none(model: DeepSeekV4FlashGGUF, name: str) -> DeepSeekV4FlashTensor | None:
    return model.tensors.get(name)


def _bind_layer_tensors(model: DeepSeekV4FlashGGUF) -> tuple[DeepSeekV4FlashLayerTensors, ...]:
    layers: list[DeepSeekV4FlashLayerTensors] = []
    for layer_idx in range(DEEPSEEK_V4_FLASH_SHAPE.num_layers):
        prefix = f"blk.{layer_idx}."
        routed: dict[int, DeepSeekV4FlashExpertTensors] = {}
        for expert_id in range(DEEPSEEK_V4_FLASH_SHAPE.num_experts):
            gate = _tensor_or_none(model, f"{prefix}ffn_gate_exps.weight.{expert_id}")
            up = _tensor_or_none(model, f"{prefix}ffn_up_exps.weight.{expert_id}")
            down = _tensor_or_none(model, f"{prefix}ffn_down_exps.weight.{expert_id}")
            if gate is not None and up is not None and down is not None:
                routed[expert_id] = DeepSeekV4FlashExpertTensors(gate=gate, up=up, down=down)
        layers.append(
            DeepSeekV4FlashLayerTensors(
                layer_idx=layer_idx,
                attention_query=_tensor_or_none(model, f"{prefix}attn_q.weight"),
                attention_key=_tensor_or_none(model, f"{prefix}attn_k.weight"),
                attention_value=_tensor_or_none(model, f"{prefix}attn_v.weight"),
                attention_output=_tensor_or_none(model, f"{prefix}attn_output.weight"),
                attention_norm=_tensor_or_none(model, f"{prefix}attn_norm.weight"),
                ffn_norm=_tensor_or_none(model, f"{prefix}ffn_norm.weight"),
                router=_tensor_or_none(model, f"{prefix}ffn_gate_inp.weight"),
                shared_experts=tuple(
                    tensor
                    for name, tensor in sorted(model.tensors.items())
                    if name.startswith(f"{prefix}ffn_") and "_shexp" in name
                ),
                routed_experts=routed,
            )
        )
    return tuple(layers)
```

If the real audit shows different target names, update these exact strings and preserve the failing test assertions against the observed names.

- [ ] **Step 5: Run the inspect tool on the downloaded model**

Run:

```bash
uv run tests/tools/deepseek_v4_flash_inspect.py \
  models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

Expected: prints layer count, tensor type counts, representative names, and nonzero bound tensor count without raising.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_weight_store_real_mapping.py tests/deepseek_v4_flash/test_weight_store.py -q
git add vllm/model_executor/models/deepseek_v4_flash/weight_store.py tests/tools/deepseek_v4_flash_inspect.py tests/deepseek_v4_flash/test_weight_store_real_mapping.py
git commit -m "feat: bind deepseek v4 flash gguf tensors"
```

---

### Task 2: Target GGUF Quant Reference Decoders

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/quant.py`
- Test: `tests/deepseek_v4_flash/test_quant_target_layout.py`

- [ ] **Step 1: Write failing layout tests for Q8_0, IQ2_XXS, and Q2_K**

Create `tests/deepseek_v4_flash/test_quant_target_layout.py`:

```python
import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.quant import (
    decode_iq2_xxs_gguf_blocks_reference,
    decode_q2_k_gguf_blocks_reference,
    q8_0_matrix_from_gguf_payload,
)


def test_q8_0_rejects_partial_block() -> None:
    with pytest.raises(ValueError, match="multiple of Q8_0 block bytes"):
        q8_0_matrix_from_gguf_payload(b"\x00" * 33, rows=1, columns=32)


def test_iq2_xxs_reference_returns_float32_vector() -> None:
    payload = bytes(256)
    decoded = decode_iq2_xxs_gguf_blocks_reference(payload, values_per_block=256)
    assert decoded.dtype == torch.float32
    assert decoded.shape == (256,)


def test_q2_k_reference_returns_float32_vector() -> None:
    payload = bytes(84)
    decoded = decode_q2_k_gguf_blocks_reference(payload, values_per_block=256)
    assert decoded.dtype == torch.float32
    assert decoded.shape == (256,)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_quant_target_layout.py -q
```

Expected: fail because IQ2_XXS and Q2_K reference functions do not exist.

- [ ] **Step 3: Implement conservative CPU reference decoders**

Add functions in `quant.py`:

```python
def decode_iq2_xxs_gguf_blocks_reference(
    payload: bytes | bytearray | memoryview,
    *,
    values_per_block: int = 256,
) -> torch.Tensor:
    """Reference-only IQ2_XXS decoder for layout bring-up.

    This first version validates block sizing and returns dequantized float32.
    Fill the exact bit unpacking from the target GGUF audit before using it in
    model execution.
    """
    _validate_positive_block_size(values_per_block)
    if len(payload) == 0:
        return torch.empty(0, dtype=torch.float32)
    if len(payload) % 256 != 0:
        raise ValueError(f"IQ2_XXS payload length must be divisible by 256; got {len(payload)}")
    block_count = len(payload) // 256
    return torch.zeros(block_count * values_per_block, dtype=torch.float32)


def decode_q2_k_gguf_blocks_reference(
    payload: bytes | bytearray | memoryview,
    *,
    values_per_block: int = 256,
) -> torch.Tensor:
    """Reference-only Q2_K decoder for layout bring-up."""
    _validate_positive_block_size(values_per_block)
    if len(payload) == 0:
        return torch.empty(0, dtype=torch.float32)
    if len(payload) % 84 != 0:
        raise ValueError(f"Q2_K payload length must be divisible by 84; got {len(payload)}")
    block_count = len(payload) // 84
    return torch.zeros(block_count * values_per_block, dtype=torch.float32)
```

This deliberately blocks real execution until exact bit layouts are filled from the DS4/GGML reference. Do not connect these zero decoders to `forward()`.

- [ ] **Step 4: Replace zero decoders with exact DS4/GGML bit unpacking**

Use the target model audit plus DS4/GGML quant layout to implement:

```python
def iq2_xxs_matrix_from_gguf_payload(
    payload: bytes | bytearray | memoryview,
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    decoded = decode_iq2_xxs_gguf_blocks_reference(payload)
    expected = rows * columns
    if decoded.numel() != expected:
        raise ValueError(f"IQ2_XXS decoded value count mismatch: {decoded.numel()} != {expected}")
    return decoded.reshape(rows, columns)


def q2_k_matrix_from_gguf_payload(
    payload: bytes | bytearray | memoryview,
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    decoded = decode_q2_k_gguf_blocks_reference(payload)
    expected = rows * columns
    if decoded.numel() != expected:
        raise ValueError(f"Q2_K decoded value count mismatch: {decoded.numel()} != {expected}")
    return decoded.reshape(rows, columns)
```

Add known-byte fixtures from the DS4/GGML implementation so the tests assert exact nonzero decoded values.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_quant_target_layout.py tests/deepseek_v4_flash/test_quant_reference.py -q
git add vllm/model_executor/models/deepseek_v4_flash/quant.py tests/deepseek_v4_flash/test_quant_target_layout.py
git commit -m "feat: decode deepseek target gguf quant blocks"
```

---

### Task 3: Batch=1 Weight Access And Reference Linear Path

**Files:**
- Create: `vllm/model_executor/models/deepseek_v4_flash/layers.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/weight_store.py`
- Test: `tests/deepseek_v4_flash/test_layers_reference.py`

- [ ] **Step 1: Write failing tests for tensor payload extraction and linear matvec**

Create `tests/deepseek_v4_flash/test_layers_reference.py`:

```python
import torch

from vllm.model_executor.models.deepseek_v4_flash.layers import (
    deepseek_linear_reference,
)


def test_deepseek_linear_reference_multiplies_row_major_weight() -> None:
    hidden = torch.tensor([1.0, 2.0], dtype=torch.float32)
    weight = torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    out = deepseek_linear_reference(hidden, weight)
    torch.testing.assert_close(out, torch.tensor([11.0, 17.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_layers_reference.py -q
```

Expected: fail because `layers.py` does not exist.

- [ ] **Step 3: Implement the reference linear helper**

Create `layers.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def deepseek_linear_reference(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D for batch=1; got {hidden.ndim}-D")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D; got {weight.ndim}-D")
    if weight.shape[1] != hidden.numel():
        raise ValueError(
            f"weight columns must match hidden size; got {weight.shape[1]} and {hidden.numel()}"
        )
    return weight.to(torch.float32).matmul(hidden.to(torch.float32))
```

- [ ] **Step 4: Add weight-store payload API**

Add to `DeepSeekV4FlashWeightStore`:

```python
def tensor_payload(self, tensor: DeepSeekV4FlashTensor) -> memoryview:
    start = tensor.offset
    end = start + tensor.nbytes
    return memoryview(self._mmap)[start:end]
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_layers_reference.py tests/deepseek_v4_flash/test_weight_store.py -q
git add vllm/model_executor/models/deepseek_v4_flash/layers.py vllm/model_executor/models/deepseek_v4_flash/weight_store.py tests/deepseek_v4_flash/test_layers_reference.py
git commit -m "feat: add deepseek reference linear path"
```

---

### Task 4: Compressed KV Append/Read Contract For Batch=1

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/compressed_kv.py`
- Test: `tests/deepseek_v4_flash/test_compressed_kv_runtime.py`

- [ ] **Step 1: Write failing batch=1 KV append/read tests**

Create `tests/deepseek_v4_flash/test_compressed_kv_runtime.py`:

```python
import torch

from vllm.model_executor.models.deepseek_v4_flash.compressed_kv import (
    DeepSeekV4CompressedKVCache,
)


def test_raw_swa_cache_keeps_last_128_rows() -> None:
    cache = DeepSeekV4CompressedKVCache(context_length=256, hidden_size=4)
    for token_idx in range(130):
        row = torch.full((4,), float(token_idx))
        cache.append_raw(layer_idx=0, token_idx=token_idx, key=row, value=row + 1)

    keys, values = cache.read_raw_window(layer_idx=0, token_idx=129, window=128)

    assert keys.shape == (128, 4)
    assert values.shape == (128, 4)
    torch.testing.assert_close(keys[0], torch.full((4,), 2.0))
    torch.testing.assert_close(values[-1], torch.full((4,), 130.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_compressed_kv_runtime.py -q
```

Expected: fail because runtime append/read APIs are missing.

- [ ] **Step 3: Implement batch=1 raw cache APIs**

In `compressed_kv.py`, add or extend `DeepSeekV4CompressedKVCache` with:

```python
def append_raw(
    self,
    *,
    layer_idx: int,
    token_idx: int,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    slot = token_idx % self.raw_window
    self.raw_keys[layer_idx][slot].copy_(key)
    self.raw_values[layer_idx][slot].copy_(value)
    self.raw_token_indices[layer_idx][slot] = token_idx


def read_raw_window(
    self,
    *,
    layer_idx: int,
    token_idx: int,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = max(0, token_idx - window + 1)
    rows: list[tuple[int, int]] = []
    for slot, stored_idx in enumerate(self.raw_token_indices[layer_idx].tolist()):
        if start <= stored_idx <= token_idx:
            rows.append((stored_idx, slot))
    rows.sort()
    slots = [slot for _, slot in rows]
    return self.raw_keys[layer_idx][slots], self.raw_values[layer_idx][slots]
```

Keep existing page-table tests passing; if the current class has a different constructor, add a narrow runtime constructor or factory instead of breaking existing tests.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_compressed_kv.py tests/deepseek_v4_flash/test_compressed_kv_runtime.py -q
git add vllm/model_executor/models/deepseek_v4_flash/compressed_kv.py tests/deepseek_v4_flash/test_compressed_kv_runtime.py
git commit -m "feat: add deepseek batch one kv runtime"
```

---

### Task 5: Single-Layer Attention And MoE Reference Execution

**Files:**
- Create: `vllm/model_executor/models/deepseek_v4_flash/attention.py`
- Create: `vllm/model_executor/models/deepseek_v4_flash/moe.py`
- Test: `tests/deepseek_v4_flash/test_attention_reference.py`
- Test: `tests/deepseek_v4_flash/test_moe_reference.py`

- [ ] **Step 1: Write failing attention test**

Create `tests/deepseek_v4_flash/test_attention_reference.py`:

```python
import torch

from vllm.model_executor.models.deepseek_v4_flash.attention import (
    raw_swa_attention_reference,
)


def test_raw_swa_attention_reference_uses_causal_scores() -> None:
    query = torch.tensor([1.0, 0.0])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = torch.tensor([[10.0, 0.0], [0.0, 20.0]])

    out = raw_swa_attention_reference(query, keys, values)

    assert out.shape == (2,)
    assert out[0] > out[1]
```

- [ ] **Step 2: Write failing MoE routing test**

Create `tests/deepseek_v4_flash/test_moe_reference.py`:

```python
import torch

from vllm.model_executor.models.deepseek_v4_flash.moe import topk_router_reference


def test_topk_router_reference_returns_sorted_experts() -> None:
    hidden = torch.tensor([1.0, 2.0])
    router_weight = torch.tensor([[1.0, 0.0], [0.0, 3.0], [2.0, 0.0]])

    expert_ids, weights = topk_router_reference(hidden, router_weight, top_k=2)

    assert expert_ids.tolist() == [1, 2]
    assert weights.shape == (2,)
    assert torch.all(weights > 0)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_attention_reference.py tests/deepseek_v4_flash/test_moe_reference.py -q
```

Expected: fail because `attention.py` and `moe.py` do not exist.

- [ ] **Step 4: Implement reference attention**

Create `attention.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import torch


def raw_swa_attention_reference(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    if query.ndim != 1:
        raise ValueError(f"query must be 1-D; got {query.ndim}-D")
    if keys.ndim != 2 or values.ndim != 2:
        raise ValueError("keys and values must be 2-D")
    if keys.shape != values.shape:
        raise ValueError(f"keys and values shape mismatch: {keys.shape} != {values.shape}")
    scores = keys.to(torch.float32).matmul(query.to(torch.float32))
    scores = scores / math.sqrt(float(query.numel()))
    probs = torch.softmax(scores, dim=0)
    return probs.matmul(values.to(torch.float32))
```

- [ ] **Step 5: Implement reference router**

Create `moe.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def topk_router_reference(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D; got {hidden.ndim}-D")
    logits = router_weight.to(torch.float32).matmul(hidden.to(torch.float32))
    scores = torch.softmax(logits, dim=0)
    weights, expert_ids = torch.topk(scores, k=top_k, largest=True, sorted=True)
    return expert_ids.to(torch.int64), weights
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_attention_reference.py tests/deepseek_v4_flash/test_moe_reference.py -q
git add vllm/model_executor/models/deepseek_v4_flash/attention.py vllm/model_executor/models/deepseek_v4_flash/moe.py tests/deepseek_v4_flash/test_attention_reference.py tests/deepseek_v4_flash/test_moe_reference.py
git commit -m "feat: add deepseek reference attention and moe"
```

---

### Task 6: Batch=1 Greedy Forward Smoke

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Test: `tests/deepseek_v4_flash/test_model_forward_real_smoke.py`

- [ ] **Step 1: Write failing real-model forward smoke**

Create `tests/deepseek_v4_flash/test_model_forward_real_smoke.py`:

```python
from pathlib import Path

import pytest
import torch

from vllm.model_executor.model_loader import get_model
from vllm.serving.config_builder import build_vllm_config


MODEL = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


@pytest.mark.skipif(not MODEL.exists(), reason="target DeepSeek GGUF not downloaded")
def test_real_model_forward_returns_vocab_logits_for_one_token() -> None:
    model = get_model(build_vllm_config(str(MODEL), max_model_len=4096))
    try:
        logits = model.forward(torch.tensor([1], dtype=torch.long))
        assert logits.ndim == 2
        assert logits.shape[0] == 1
        assert logits.shape[1] == model.shape.vocab_size
        assert torch.isfinite(logits).all()
    finally:
        model.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_model_forward_real_smoke.py -q
```

Expected: fail with the current `NotImplementedError`.

- [ ] **Step 3: Implement narrow batch=1 forward contract**

In `model.py`, replace `forward()` with a narrow input contract:

```python
def forward(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    if self.weight_store is None:
        raise RuntimeError("DeepSeekV4FlashForCausalLM requires an attached GGUF weight store")
    if input_ids.ndim != 1:
        raise ValueError(f"DeepSeek V4 Flash first release supports batch=1 token vectors only; got {input_ids.ndim}-D")
    if input_ids.numel() == 0:
        return torch.empty((0, self.shape.vocab_size), dtype=torch.float32)
    if input_ids.numel() > 8192:
        raise ValueError(f"DeepSeek V4 Flash first release supports context <= 8192; got {input_ids.numel()}")
    return self._forward_reference_batch_one(input_ids.to(torch.long))
```

Add `_forward_reference_batch_one()` that executes the available bound tensors in layer order and returns `[seq_len, vocab]` logits. It must call exact quant decoders from Task 2, `raw_swa_attention_reference()` from Task 5, and `topk_router_reference()` from Task 5. Keep this path correctness-first and explicitly batch=1.

- [ ] **Step 4: Run real forward smoke and existing DeepSeek tests**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_model_forward_real_smoke.py tests/deepseek_v4_flash -q
```

Expected: real forward smoke passes or exposes the next exact missing tensor/quant mapping. Fix only the missing mapping required by this test.

- [ ] **Step 5: Commit**

Run:

```bash
git add vllm/model_executor/models/deepseek_v4_flash/model.py tests/deepseek_v4_flash/test_model_forward_real_smoke.py
git commit -m "feat: run deepseek batch one forward smoke"
```

---

### Task 7: OpenAI-Compatible REST Greedy Smoke

**Files:**
- Modify: `tests/smoke/test_deepseek_v4_flash_http_smoke.py`
- Optional Modify: `vllm/serving/config_builder.py`
- Optional Modify: `vllm/entrypoints/openai/engine/serving.py`

- [ ] **Step 1: Add REST smoke that checks actual response text**

Extend `tests/smoke/test_deepseek_v4_flash_http_smoke.py`:

```python
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.api_server import app


MODEL = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


@pytest.mark.skipif(not MODEL.exists(), reason="target DeepSeek GGUF not downloaded")
def test_deepseek_chat_completions_real_greedy_smoke(monkeypatch) -> None:
    monkeypatch.setenv("MODEL", str(MODEL))
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": str(MODEL),
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "max_tokens": 1,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"]
    assert payload["choices"][0]["message"]["content"] is not None
```

- [ ] **Step 2: Run test to verify current REST failure mode**

Run:

```bash
uv run pytest tests/smoke/test_deepseek_v4_flash_http_smoke.py -q
```

Expected: route test passes; real generation smoke fails until server initialization and model path plumbing are aligned.

- [ ] **Step 3: Wire DeepSeek model path through existing lite REST config**

If the test shows REST is not constructing the model from the request model path, update the existing config builder/server path so the DeepSeek GGUF path reaches `build_vllm_config()` and `get_model()` without adding DeepSeek-specific branches to the engine.

- [ ] **Step 4: Run REST smoke**

Run:

```bash
uv run pytest tests/smoke/test_deepseek_v4_flash_http_smoke.py -q
```

Expected: both route exposure and real greedy response smoke pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add tests/smoke/test_deepseek_v4_flash_http_smoke.py vllm/serving/config_builder.py vllm/entrypoints/openai/engine/serving.py
git commit -m "feat: serve deepseek greedy rest smoke"
```

---

### Task 8: Validation, Documentation, And Performance Guardrails

**Files:**
- Modify: `docs/design/deepseek_v4_flash_q2_native.md`
- Modify: `docs/superpowers/plans/2026-06-03-deepseek-v4-flash-q2-native.md`
- Optional Modify: `tests/run_regression_suite.sh`

- [ ] **Step 1: Run the fast regression suite**

Run:

```bash
bash tests/run_regression_suite.sh
```

Expected: all fast unit/smoke tests pass.

- [ ] **Step 2: Run DeepSeek real smoke tests**

Run:

```bash
uv run pytest tests/deepseek_v4_flash tests/smoke/test_deepseek_v4_flash_http_smoke.py -q
```

Expected: DeepSeek tests pass. Real-model tests may be slow but should not hang.

- [ ] **Step 3: Add a timeout wrapper for real-model smoke commands**

When running manually on this hardware, use:

```bash
timeout 1800 uv run pytest tests/deepseek_v4_flash/test_model_forward_real_smoke.py -q
timeout 1800 uv run pytest tests/smoke/test_deepseek_v4_flash_http_smoke.py -q
```

Expected: either pass or exit with a bounded failure, never hang indefinitely.

- [ ] **Step 4: Update design docs with actual supported state**

Update `docs/design/deepseek_v4_flash_q2_native.md` to state:

```markdown
## Implemented Real-Inference State

- Target GGUF file: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- First executable target: batch=1, greedy, context up to 8192
- REST surface: `POST /v1/chat/completions`
- Current implementation: correctness-first reference path; Triton optimization follows after real smoke is stable
```

- [ ] **Step 5: Run formatting and commit**

Run:

```bash
uv run ruff check vllm/model_executor/models/deepseek_v4_flash tests/deepseek_v4_flash tests/smoke/test_deepseek_v4_flash_http_smoke.py
uv run ruff format vllm/model_executor/models/deepseek_v4_flash tests/deepseek_v4_flash tests/smoke/test_deepseek_v4_flash_http_smoke.py
git add docs/design/deepseek_v4_flash_q2_native.md docs/superpowers/plans/2026-06-03-deepseek-v4-flash-q2-native.md tests/run_regression_suite.sh
git commit -m "docs: update deepseek real inference state"
```

---

## Execution Notes

- Do not start with Triton optimization. First get exact real-model logits without hangs.
- Do not add DeepSeek conditionals in `vllm/engine/`; model-specific behavior belongs in the DeepSeek model package and adapter.
- Use `timeout` on all real-model smoke commands until repeated runs prove the path cannot hang.
- If the real GGUF tensor names differ from expected patterns, update Task 1 bindings first and do not patch around missing tensors inside `forward()`.
- If IQ2_XXS/Q2_K bit layouts are not understood, stop at Task 2 and audit DS4/GGML source before connecting decoders to real execution.

## Task 8 Outcome Addendum

Task 6 ended with a deliberately limited direct model smoke, not full batch=1
greedy forward. The implemented path accepts exactly one token, runs token
embedding, `output_norm` RMSNorm, and the `Q8_0` output projection, then returns
finite `[1, vocab]` logits. It sets `limited_forward_smoke_only=True` and rejects
multi-token non-empty input. It does not execute transformer layers, factorized
attention, combined KV use, compressed attention, or grouped experts.

Task 7 ended with route exposure plus an honest negative REST smoke. The app
imports, `/v1/chat/completions` and `/v1/models` are exposed, and an
uninitialized engine returns HTTP 503 for chat requests. Initialized
OpenAI-compatible DeepSeek GGUF generation remains blocked because that route
uses the `AsyncLLM`/`LiteEngine` full autoregressive path, not the limited
one-token direct model smoke.

## Self-Review

- Spec coverage: The plan covers model file audit, tensor binding, quant decoding, KV runtime, attention/MoE reference execution, batch=1 forward, REST greedy smoke, and docs/validation.
- Placeholder scan: No red-flag placeholder wording remains. Task 2 explicitly blocks connection of incomplete zero decoders to forward.
- Type consistency: The plan consistently uses `DeepSeekV4FlashWeightStore`, `DeepSeekV4FlashSemanticBindings`, `DeepSeekV4FlashLayerTensors`, and `DeepSeekV4FlashForCausalLM.forward(input_ids)`.
