# DeepSeek V4 Flash E2E Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build true batch=1 DeepSeek V4 Flash greedy autoregressive inference for the downloaded target GGUF, then expose it through the existing OpenAI-compatible REST route.

**Architecture:** First make a direct model runner execute real transformer blocks outside the engine with bounded tests; only after direct greedy decode works should the work connect to `LiteEngine` prefill/decode and REST. DeepSeek-specific execution stays in `vllm/model_executor/models/deepseek_v4_flash/`; `vllm/engine/` remains model-family agnostic except for generic capability hooks that are needed by any nonstandard KV model.

**Tech Stack:** Python 3.12, `uv`, PyTorch reference execution, mmap-backed GGUF payloads, existing quant reference decoders, existing lite engine/OpenAI REST path, later Triton kernels through `vllm/triton_utils/`.

---

## Current Starting Point

Implemented today:

- Target GGUF exists at `models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.
- GGUF reader parses metadata/tensors and computes data-section offsets.
- Weight store binds token embedding, output norm/head, factorized attention tensors, combined KV tensors, grouped expert tensors, router tensors, and metadata tensors.
- Reference decoders exist for `Q8_0`, `IQ2_XXS`, and `Q2_K`.
- Direct model smoke supports exactly one token through embedding, `output_norm`, and `output.weight`.
- REST routes exist, but initialized REST generation is not implemented.

Not implemented:

- Real transformer block execution.
- Factorized Q/O attention execution.
- Shared K=V latent `attn_kv.weight` use.
- Compressed attention / indexer execution.
- Grouped expert execution.
- Batch=1 greedy decode.
- Initialized REST generation.

## Execution Strategy

Do not start with REST. The safe sequence is:

1. Prove all real tensor payload ranges and row layouts are valid.
2. Implement direct model building blocks and single-layer execution.
3. Execute all 43 transformer layers for one token with real weights.
4. Add prefill for short prompts and raw SWA KV update.
5. Add compressed attention state and grouped MoE execution.
6. Make direct greedy decode generate token IDs.
7. Connect the working direct model path to `LiteEngine`.
8. Expose initialized OpenAI-compatible REST generation.

The first successful end-to-end acceptance target is intentionally narrow:

- `batch=1`
- `context <= 4096` first, then `8192`
- `temperature=0`
- `max_tokens=1` first, then `8`
- no speed target

---

## File Structure

- Modify `vllm/model_executor/models/deepseek_v4_flash/gguf_reader.py`: eager tensor range and overlap validation.
- Modify `vllm/model_executor/models/deepseek_v4_flash/weight_store.py`: typed payload accessors and grouped expert slice access.
- Create `vllm/model_executor/models/deepseek_v4_flash/ops.py`: RMSNorm, activation, factorized linear, quantized matvec wrappers.
- Create `vllm/model_executor/models/deepseek_v4_flash/block.py`: one transformer block reference execution.
- Modify `vllm/model_executor/models/deepseek_v4_flash/attention.py`: real DeepSeek V4 factorized attention and raw SWA attention path.
- Modify `vllm/model_executor/models/deepseek_v4_flash/moe.py`: grouped expert execution for selected top-k experts.
- Modify `vllm/model_executor/models/deepseek_v4_flash/compressed_kv.py`: raw and compressed KV runtime state.
- Modify `vllm/model_executor/models/deepseek_v4_flash/model.py`: direct prefill/decode and greedy generation.
- Create `tests/deepseek_v4_flash/test_block_reference.py`: single-layer block tests.
- Create `tests/deepseek_v4_flash/test_direct_decode_real_smoke.py`: real direct greedy smoke.
- Modify `vllm/engine/prefill_executor.py`, `vllm/engine/decode_executor.py`, or model-call boundary only if generic hooks are required for nonstandard KV.
- Modify `tests/smoke/test_deepseek_v4_flash_http_smoke.py`: initialized REST smoke after direct decode works.

---

### Task 1: Eager GGUF Tensor Range Audit

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gguf_reader.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/weight_store.py`
- Test: `tests/deepseek_v4_flash/test_gguf_reader.py`
- Test: `tests/deepseek_v4_flash/test_weight_store_real_mapping.py`

- [ ] **Step 1: Write failing range-validation tests**

Add tests that prove tensor offsets are data-section relative, sorted ranges do not overlap, and out-of-file ranges fail:

```python
def test_gguf_reader_rejects_overlapping_tensor_payloads(tmp_path) -> None:
    path = tmp_path / "overlap.gguf"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight"),
        tensor_types=(0, 0),
        tensor_dims={"token_embd.weight": (1,), "blk.0.attn_q.weight": (1,)},
        tensor_payloads={
            "token_embd.weight": b"\x00\x00\x00\x00",
            "blk.0.attn_q.weight": b"\x01\x00\x00\x00",
        },
        tensor_offsets={"token_embd.weight": 0, "blk.0.attn_q.weight": 0},
    )

    with pytest.raises(GGUFParseError, match="overlap"):
        read_deepseek_v4_flash_gguf(path)
```

Add a real-file smoke:

```python
@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_gguf_all_tensor_ranges_are_inside_file() -> None:
    model = read_deepseek_v4_flash_gguf(TARGET_GGUF)
    file_size = TARGET_GGUF.stat().st_size
    for tensor in model.tensors.values():
        start = model.data_offset + tensor.offset
        assert 0 <= start < file_size
        assert start + tensor.nbytes <= file_size
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_gguf_reader.py -q
```

Expected: overlap test fails until eager validation exists.

- [ ] **Step 3: Implement eager tensor range validation**

Add a helper in `gguf_reader.py`:

```python
def _validate_tensor_ranges(
    tensors: dict[str, DeepSeekV4FlashTensor],
    *,
    data_offset: int,
    file_size: int,
) -> None:
    ranges: list[tuple[int, int, str]] = []
    for tensor in tensors.values():
        start = data_offset + tensor.offset
        end = start + tensor.nbytes
        if tensor.nbytes <= 0:
            raise GGUFParseError(f"tensor {tensor.name} has invalid byte size {tensor.nbytes}")
        if start < data_offset or end > file_size:
            raise GGUFParseError(
                f"tensor {tensor.name} range [{start}, {end}) exceeds file size {file_size}"
            )
        ranges.append((start, end, tensor.name))
    ranges.sort()
    for (_, prev_end, prev_name), (start, _end, name) in zip(ranges, ranges[1:], strict=False):
        if start < prev_end:
            raise GGUFParseError(
                f"tensor payload overlap: {prev_name} ends at {prev_end}, {name} starts at {start}"
            )
```

Call it after `data_offset` is computed in both mmap reader entrypoints. If `read_deepseek_v4_flash_gguf_from_view()` does not know file size, use `len(data)`.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_gguf_reader.py tests/deepseek_v4_flash/test_weight_store_real_mapping.py -q
git add vllm/model_executor/models/deepseek_v4_flash/gguf_reader.py tests/deepseek_v4_flash/test_gguf_reader.py tests/deepseek_v4_flash/test_weight_store_real_mapping.py
git commit -m "feat: validate deepseek gguf tensor ranges"
```

---

### Task 2: Typed Tensor Payload Accessors

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/weight_store.py`
- Create: `tests/deepseek_v4_flash/test_weight_payload_access.py`

- [ ] **Step 1: Write failing payload accessor tests**

Create `tests/deepseek_v4_flash/test_weight_payload_access.py`:

```python
def test_weight_store_reads_f16_vector_payload(tmp_path) -> None:
    path = tmp_path / "payload.gguf"
    payload = torch.tensor([1.0, 2.0], dtype=torch.float16).numpy().tobytes()
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight"),
        tensor_types=(1, 1),
        tensor_dims={"token_embd.weight": (2,), "blk.0.attn_q.weight": (2,)},
        tensor_payloads={"token_embd.weight": payload, "blk.0.attn_q.weight": payload},
    )
    with open_deepseek_v4_flash_weight_store(path) as store:
        tensor = store.tensor_to_torch(store.bindings.token_embedding, dtype=torch.float16)
        torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0], dtype=torch.float16))
```

Add a grouped expert slice shape test using a synthetic tensor descriptor:

```python
def test_grouped_expert_slice_offsets_are_expert_major() -> None:
    offset = grouped_expert_payload_offset(
        expert_id=2,
        projection_dims=(4096, 2048, 256),
        projection_type=GGML_TYPE_IQ2_XXS,
    )
    assert offset > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_weight_payload_access.py -q
```

Expected: missing accessor helpers.

- [ ] **Step 3: Implement accessors**

Add in `weight_store.py`:

```python
def tensor_to_torch(
    self,
    tensor: DeepSeekV4FlashTensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    payload = self.tensor_payload(tensor)
    try:
        out = torch.frombuffer(payload, dtype=dtype).clone()
    finally:
        payload.release()
    return out.reshape(shape or tensor.dims)
```

Add grouped expert offset helper using the target GGUF observed layout `(input, output, expert)`:

```python
def grouped_expert_payload_offset(
    *,
    expert_id: int,
    projection_dims: tuple[int, int, int],
    projection_type: int,
) -> int:
    input_size, output_size, expert_count = projection_dims
    if not 0 <= expert_id < expert_count:
        raise ValueError(f"expert_id out of range: {expert_id}")
    values = input_size * output_size
    nbytes = ggml_tensor_nbytes((input_size, output_size), projection_type)
    return expert_id * nbytes
```

Move `_tensor_nbytes()` to a public helper such as `ggml_tensor_nbytes()`.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_weight_payload_access.py tests/deepseek_v4_flash/test_weight_store.py -q
git add vllm/model_executor/models/deepseek_v4_flash/weight_store.py vllm/model_executor/models/deepseek_v4_flash/gguf_reader.py tests/deepseek_v4_flash/test_weight_payload_access.py
git commit -m "feat: add deepseek typed weight accessors"
```

---

### Task 3: Core Reference Ops For Real Blocks

**Files:**
- Create: `vllm/model_executor/models/deepseek_v4_flash/ops.py`
- Test: `tests/deepseek_v4_flash/test_ops_reference.py`

- [ ] **Step 1: Write failing op tests**

Create `tests/deepseek_v4_flash/test_ops_reference.py`:

```python
def test_rms_norm_reference_matches_formula() -> None:
    hidden = torch.tensor([3.0, 4.0])
    weight = torch.tensor([2.0, 1.0])
    out = rms_norm_reference(hidden, weight, eps=0.0)
    expected = hidden * torch.rsqrt(hidden.pow(2).mean()) * weight
    torch.testing.assert_close(out, expected)


def test_factorized_linear_reference_applies_a_then_b() -> None:
    hidden = torch.tensor([1.0, 2.0])
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
    out = factorized_linear_reference(hidden, a, b)
    torch.testing.assert_close(out, torch.tensor([4.0, 4.0]))
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_ops_reference.py -q
```

Expected: missing `ops.py`.

- [ ] **Step 3: Implement ops**

Create `ops.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import torch.nn.functional as F


def rms_norm_reference(hidden: torch.Tensor, weight: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    if hidden.ndim != 1 or weight.ndim != 1:
        raise ValueError("rms_norm_reference expects 1-D hidden and weight")
    if hidden.numel() != weight.numel():
        raise ValueError("hidden and weight sizes must match")
    variance = hidden.to(torch.float32).pow(2).mean()
    return hidden.to(torch.float32) * torch.rsqrt(variance + eps) * weight.to(torch.float32)


def silu_gate_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError(f"gate/up shape mismatch: {tuple(gate.shape)} != {tuple(up.shape)}")
    return F.silu(gate.to(torch.float32)) * up.to(torch.float32)


def factorized_linear_reference(hidden: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 1 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("factorized_linear_reference expects hidden 1-D and matrices 2-D")
    if a.shape[1] != hidden.numel():
        raise ValueError("A columns must match hidden size")
    mid = a.to(torch.float32).matmul(hidden.to(torch.float32))
    if b.shape[1] != mid.numel():
        raise ValueError("B columns must match factorized intermediate size")
    return b.to(torch.float32).matmul(mid)
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_ops_reference.py -q
git add vllm/model_executor/models/deepseek_v4_flash/ops.py tests/deepseek_v4_flash/test_ops_reference.py
git commit -m "feat: add deepseek block reference ops"
```

---

### Task 4: Real Attention Tensor Mapping

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/attention.py`
- Test: `tests/deepseek_v4_flash/test_attention_mapping.py`

- [ ] **Step 1: Write failing mapping tests**

Create tests that assert the real tensor dimensions and expected output widths:

```python
@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_attention_factor_tensor_shapes() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        layer = store.bindings.layers[0]
        assert layer.attention_query_a is not None
        assert layer.attention_query_b is not None
        assert layer.attention_key_value is not None
        assert layer.attention_output_a is not None
        assert layer.attention_output_b is not None
        assert layer.attention_key_value.dims == (4096, 512)
```

Add synthetic factorized projection test:

```python
def test_factorized_query_projection_returns_expected_width() -> None:
    hidden = torch.ones(4)
    q_a = torch.eye(4)
    q_b = torch.ones((8, 4))
    out = factorized_attention_projection_reference(hidden, q_a, q_b)
    assert out.shape == (8,)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_attention_mapping.py -q
```

Expected: missing factorized attention helper.

- [ ] **Step 3: Implement factorized attention projection helpers**

Add in `attention.py`:

```python
def factorized_attention_projection_reference(
    hidden: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
) -> torch.Tensor:
    return factorized_linear_reference(hidden, a_weight, b_weight)


def split_combined_kv_reference(kv: torch.Tensor, *, key_width: int, value_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    if kv.ndim != 1:
        raise ValueError("combined KV projection must be 1-D")
    if kv.numel() != key_width + value_width:
        raise ValueError(f"combined KV width mismatch: {kv.numel()} != {key_width + value_width}")
    return kv[:key_width], kv[key_width:]
```

The real observed `attn_kv.weight` is a 512-wide shared K=V latent projection,
not a concatenated key/value tensor. The first release attention path must use
`attn_kv.weight` followed by `attn_kv_a_norm.weight`, then use the same row for
key and value. This matches the public DeepSeek V4 Flash reference and DS4
implementation.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_attention_mapping.py tests/deepseek_v4_flash/test_attention_reference.py -q
git add vllm/model_executor/models/deepseek_v4_flash/attention.py tests/deepseek_v4_flash/test_attention_mapping.py
git commit -m "feat: map deepseek factorized attention tensors"
```

---

### Task 5: Grouped Expert Slice And Reference MoE Execution

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/moe.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/weight_store.py`
- Test: `tests/deepseek_v4_flash/test_grouped_expert_reference.py`

- [ ] **Step 1: Write failing grouped expert tests**

Create `tests/deepseek_v4_flash/test_grouped_expert_reference.py`:

```python
def test_grouped_expert_reference_combines_gate_up_down() -> None:
    hidden = torch.tensor([1.0, 2.0])
    gate = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    up = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    down = torch.eye(2)
    out = grouped_expert_reference(hidden, gate, up, down)
    expected = torch.nn.functional.silu(gate.matmul(hidden)) * up.matmul(hidden)
    torch.testing.assert_close(out, expected)
```

Add a real grouped expert slice test that reads only one expert slice and validates decoded matrix shape without loading all experts:

```python
@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_grouped_expert_slice_decodes_one_expert_without_full_tensor() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        layer = store.bindings.layers[0]
        assert layer.grouped_experts is not None
        gate = store.decode_grouped_expert_matrix(layer.grouped_experts.gate, expert_id=0)
        assert gate.shape == (2048, 4096)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
timeout 600 uv run --no-sync pytest tests/deepseek_v4_flash/test_grouped_expert_reference.py -q
```

Expected: missing grouped expert helper/accessor.

- [ ] **Step 3: Implement one-expert decode path**

In `weight_store.py`, add `decode_grouped_expert_matrix(tensor, expert_id)` that slices exactly one expert payload and calls the correct quant decoder based on `tensor.tensor_type`. Do not decode all 256 experts.

In `moe.py`, add:

```python
def grouped_expert_reference(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    gate = gate_weight.to(torch.float32).matmul(hidden.to(torch.float32))
    up = up_weight.to(torch.float32).matmul(hidden.to(torch.float32))
    activated = silu_gate_reference(gate, up)
    return down_weight.to(torch.float32).matmul(activated)
```

- [ ] **Step 4: Add top-k MoE reduction**

Add:

```python
def routed_moe_reference(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    expert_runner: Callable[[int, torch.Tensor], torch.Tensor],
    *,
    top_k: int,
    correction_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    expert_ids, weights = topk_router_reference(
        hidden,
        router_weight,
        top_k=top_k,
        correction_bias=correction_bias,
    )
    result: torch.Tensor | None = None
    for expert_id, weight in zip(expert_ids.tolist(), weights, strict=True):
        expert_out = expert_runner(expert_id, hidden)
        weighted = expert_out * weight.to(torch.float32)
        result = weighted if result is None else result + weighted
    if result is None:
        raise RuntimeError("top-k routing produced no experts")
    return result
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
timeout 600 uv run --no-sync pytest tests/deepseek_v4_flash/test_grouped_expert_reference.py tests/deepseek_v4_flash/test_moe_reference.py -q
git add vllm/model_executor/models/deepseek_v4_flash/moe.py vllm/model_executor/models/deepseek_v4_flash/weight_store.py tests/deepseek_v4_flash/test_grouped_expert_reference.py
git commit -m "feat: execute one deepseek grouped expert"
```

---

### Task 6: Single Transformer Block Reference

**Files:**
- Create: `vllm/model_executor/models/deepseek_v4_flash/block.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Test: `tests/deepseek_v4_flash/test_block_reference.py`

- [x] **Step 1: Write failing synthetic block test**

Create a small block test with synthetic weights:

```python
def test_block_reference_preserves_hidden_shape() -> None:
    block = DeepSeekV4FlashBlockReference(
        layer_idx=0,
        hidden_size=4,
        attention=lambda hidden, token_idx, kv_cache: hidden * 0.5,
        moe=lambda hidden: hidden * 2.0,
        attn_norm_weight=torch.ones(4),
        ffn_norm_weight=torch.ones(4),
    )
    out = block.forward(torch.ones(4), token_idx=0, kv_cache=None)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()
```

- [x] **Step 2: Run test to verify failure**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_block_reference.py -q
```

Expected: missing `block.py`.

- [x] **Step 3: Implement block skeleton**

Create `block.py`:

```python
@dataclass
class DeepSeekV4FlashBlockReference:
    layer_idx: int
    hidden_size: int
    attention: Callable[[torch.Tensor, int, object], torch.Tensor]
    moe: Callable[[torch.Tensor], torch.Tensor]
    attn_norm_weight: torch.Tensor
    ffn_norm_weight: torch.Tensor

    def forward(self, hidden: torch.Tensor, *, token_idx: int, kv_cache: object) -> torch.Tensor:
        if hidden.shape != (self.hidden_size,):
            raise ValueError(f"hidden shape must be ({self.hidden_size},); got {tuple(hidden.shape)}")
        attn_in = rms_norm_reference(hidden, self.attn_norm_weight)
        hidden = hidden + self.attention(attn_in, token_idx, kv_cache)
        ffn_in = rms_norm_reference(hidden, self.ffn_norm_weight)
        return hidden + self.moe(ffn_in)
```

Use the real DeepSeek residual order from reference implementation. If this residual order is wrong after checking DS4/HF, fix this task before proceeding.

- [ ] **Step 4: Add real layer-0 smoke**

Add a real-file test that constructs layer 0 with real tensors, executes a single token hidden vector, and returns finite hidden state. If layer 0 attention/FFN mapping is not complete, this test should be marked expected failure with a precise reason only while Task 4/5 are being completed; before moving to Task 7 it must pass.

Status: `DeepSeekV4FlashBlockReference` exists. Real layer tensors now bound
through `DeepSeekV4FlashLayerSemanticBindings` include layer norms,
`attn_sinks`, Q/KV latent norms, mHC tensors, attention compressor tensors, and
indexer tensors. The real attention+MoE execution test is intentionally `xfail`
until mHC pre/post, layer-0 raw SWA with shared K=V latent rows, grouped
attention output projection, and shared+routed MoE are implemented.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
timeout 900 uv run --no-sync pytest tests/deepseek_v4_flash/test_block_reference.py -q
git add vllm/model_executor/models/deepseek_v4_flash/block.py vllm/model_executor/models/deepseek_v4_flash/model.py tests/deepseek_v4_flash/test_block_reference.py
git commit -m "feat: execute deepseek reference block"
```

---

### Task 7: Direct Full-Model One-Token Decode

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Test: `tests/deepseek_v4_flash/test_direct_decode_real_smoke.py`

- [ ] **Step 1: Write failing direct full-model test**

Create `tests/deepseek_v4_flash/test_direct_decode_real_smoke.py`:

```python
@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_direct_full_model_one_token_decode_returns_logits() -> None:
    model = get_model(build_vllm_config(str(TARGET_GGUF), max_model_len=4096))
    try:
        logits = model.forward_full_reference(torch.tensor([1], dtype=torch.long))
        assert logits.shape == (1, model.shape.vocab_size)
        assert torch.isfinite(logits).all()
        assert model.limited_forward_smoke_only is False
    finally:
        model.close()
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
timeout 1800 uv run --no-sync pytest tests/deepseek_v4_flash/test_direct_decode_real_smoke.py -q
```

Expected: missing `forward_full_reference()`.

- [ ] **Step 3: Implement `forward_full_reference()`**

In `model.py`, keep existing limited `forward()` until the full path is stable. Add:

```python
def forward_full_reference(self, input_ids: torch.Tensor) -> torch.Tensor:
    self._validate_batch_one_input(input_ids, max_tokens=1)
    hidden = self._read_token_embedding(self._require_weight_store(), int(input_ids[0]))
    kv_cache = DeepSeekV4CompressedKVCache(context_length=4096, hidden_size=self.shape.head_dim)
    for layer_idx in range(self.shape.num_layers):
        hidden = self._run_layer_reference(layer_idx, hidden, token_idx=0, kv_cache=kv_cache)
    norm_weight = self._read_output_norm(self._require_weight_store())
    logits = self._q8_0_output_projection(self._require_weight_store(), self._rms_norm(hidden, norm_weight))
    return logits.reshape(1, -1)
```

The actual implementation must use the real block helpers and not reuse the limited output-projection-only smoke.

- [ ] **Step 4: Run test and commit**

Run:

```bash
timeout 1800 uv run --no-sync pytest tests/deepseek_v4_flash/test_direct_decode_real_smoke.py tests/deepseek_v4_flash/test_model_forward_real_smoke.py -q
git add vllm/model_executor/models/deepseek_v4_flash/model.py tests/deepseek_v4_flash/test_direct_decode_real_smoke.py
git commit -m "feat: run deepseek full one-token decode"
```

---

### Task 8: Direct Greedy Decode Loop

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Test: `tests/deepseek_v4_flash/test_direct_greedy_decode_real_smoke.py`

- [ ] **Step 1: Write failing greedy decode test**

Create:

```python
@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_direct_greedy_decode_generates_one_token_id() -> None:
    model = get_model(build_vllm_config(str(TARGET_GGUF), max_model_len=4096))
    try:
        output_ids = model.generate_greedy_reference(
            torch.tensor([1], dtype=torch.long),
            max_new_tokens=1,
        )
        assert output_ids.ndim == 1
        assert output_ids.numel() == 2
        assert 0 <= int(output_ids[-1]) < model.shape.vocab_size
    finally:
        model.close()
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
timeout 1800 uv run --no-sync pytest tests/deepseek_v4_flash/test_direct_greedy_decode_real_smoke.py -q
```

Expected: missing `generate_greedy_reference()`.

- [ ] **Step 3: Implement greedy loop**

Add:

```python
def generate_greedy_reference(self, input_ids: torch.Tensor, *, max_new_tokens: int) -> torch.Tensor:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    ids = input_ids.to(torch.long).clone()
    for _ in range(max_new_tokens):
        logits = self.forward_full_reference(ids[-1:])
        next_id = torch.argmax(logits[-1], dim=-1).reshape(1).to(torch.long)
        ids = torch.cat([ids, next_id])
    return ids
```

For this first pass, recomputing only the last token is acceptable only if KV state is correctly carried by the model object. If KV is not carried, implement a decode state object before this test is allowed to pass.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
timeout 1800 uv run --no-sync pytest tests/deepseek_v4_flash/test_direct_greedy_decode_real_smoke.py tests/deepseek_v4_flash/test_direct_decode_real_smoke.py -q
git add vllm/model_executor/models/deepseek_v4_flash/model.py tests/deepseek_v4_flash/test_direct_greedy_decode_real_smoke.py
git commit -m "feat: add deepseek direct greedy decode"
```

---

### Task 9: Tokenizer And Chat Prompt Binding

**Files:**
- Create: `vllm/model_executor/models/deepseek_v4_flash/tokenizer.py`
- Test: `tests/deepseek_v4_flash/test_tokenizer_binding.py`

- [ ] **Step 1: Locate tokenizer assets**

Check whether `models/DeepSeek-V4-Flash-ds4/` contains tokenizer files. If absent, fetch the tokenizer files from the same model repository without downloading the GGUF again.

Run:

```bash
find models/DeepSeek-V4-Flash-ds4 -maxdepth 1 -type f -name '*token*' -o -name '*.json'
```

- [ ] **Step 2: Write failing tokenizer tests**

Create:

```python
def test_deepseek_tokenizer_encodes_prompt() -> None:
    tokenizer = load_deepseek_v4_flash_tokenizer("models/DeepSeek-V4-Flash-ds4")
    ids = tokenizer.encode("hello")
    assert ids
    assert all(isinstance(token_id, int) for token_id in ids)
```

- [ ] **Step 3: Implement tokenizer loader**

Prefer the existing project tokenizer utility if available. If using Transformers:

```python
def load_deepseek_v4_flash_tokenizer(model_dir: str | Path) -> Any:
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
```

Do not import Transformers at module import time if it slows non-tokenizer tests; import inside the function.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_tokenizer_binding.py -q
git add vllm/model_executor/models/deepseek_v4_flash/tokenizer.py tests/deepseek_v4_flash/test_tokenizer_binding.py models/DeepSeek-V4-Flash-ds4/tokenizer* models/DeepSeek-V4-Flash-ds4/*.json
git commit -m "feat: bind deepseek tokenizer assets"
```

Only add tokenizer/config assets if they are small and appropriate for the repo. Do not commit the 80.7GiB GGUF.

---

### Task 10: LiteEngine Integration Boundary

**Files:**
- Modify: `vllm/engine/prefill_executor.py`
- Modify: `vllm/engine/decode_executor.py`
- Modify: `vllm/engine/lite_engine.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Test: `tests/deepseek_v4_flash/test_lite_engine_deepseek_integration.py`

- [ ] **Step 1: Write failing engine integration test**

Create a test that initializes `LLM` or `AsyncLLM` with the target GGUF and makes a one-token greedy request. If full `AsyncLLM` initialization is too heavy for unit tests, write a prefill/decode executor boundary test with a fake scheduler request.

```python
@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_lite_engine_deepseek_one_token_greedy_request() -> None:
    llm = LLM(model=str(TARGET_GGUF), max_model_len=4096)
    outputs = llm.generate(["hello"], SamplingParams(temperature=0, max_tokens=1))
    assert outputs
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
timeout 1800 uv run --no-sync pytest tests/deepseek_v4_flash/test_lite_engine_deepseek_integration.py -q
```

Expected: failure at tokenizer, KV allocation, or model-call boundary.

- [ ] **Step 3: Add a generic nonstandard-KV model hook**

Do not add DeepSeek-specific branches to the scheduler. Add a generic capability check:

```python
if hasattr(model, "prefill_decode_greedy"):
    return model.prefill_decode_greedy(request_input, sampling_params)
```

Place the hook at the narrowest model-call boundary where Gemma/Qwen paths remain unchanged.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
timeout 1800 uv run --no-sync pytest tests/deepseek_v4_flash/test_lite_engine_deepseek_integration.py tests/run_deepseek_v4_flash_real_smoke.sh -q
git add vllm/engine/prefill_executor.py vllm/engine/decode_executor.py vllm/engine/lite_engine.py vllm/model_executor/models/deepseek_v4_flash/model.py tests/deepseek_v4_flash/test_lite_engine_deepseek_integration.py
git commit -m "feat: route deepseek through lite engine"
```

---

### Task 11: Initialized REST Generation Smoke

**Files:**
- Modify: `tests/smoke/test_deepseek_v4_flash_http_smoke.py`
- Optional Modify: `vllm/entrypoints/openai/api_server.py`
- Optional Modify: `vllm/serving/config_builder.py`

- [ ] **Step 1: Write failing initialized REST test**

Extend `tests/smoke/test_deepseek_v4_flash_http_smoke.py`:

```python
@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_initialized_deepseek_chat_completion_generates_one_token() -> None:
    with deepseek_rest_timeout_guard(1800):
        client = TestClient(api_server.app)
        api_server.engine = AsyncLLM(build_vllm_config(str(TARGET_GGUF), max_model_len=4096))
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": str(TARGET_GGUF),
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0,
                    "max_tokens": 1,
                },
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["choices"]
        finally:
            api_server.engine = None
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
timeout 1800 uv run --no-sync pytest tests/smoke/test_deepseek_v4_flash_http_smoke.py -q
```

Expected: failure until LiteEngine integration and tokenizer are complete.

- [ ] **Step 3: Fix REST path without model-specific engine branches**

Ensure:

- `build_vllm_config()` accepts the GGUF path.
- `AsyncLLM` initializes tokenizer/model without treating local path as HF repo id.
- Chat request preprocessing produces token IDs accepted by DeepSeek direct decode.
- Response detokenization produces content.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
timeout 1800 uv run --no-sync pytest tests/smoke/test_deepseek_v4_flash_http_smoke.py tests/deepseek_v4_flash/test_lite_engine_deepseek_integration.py -q
git add tests/smoke/test_deepseek_v4_flash_http_smoke.py vllm/entrypoints/openai/api_server.py vllm/serving/config_builder.py
git commit -m "feat: serve deepseek initialized rest generation"
```

---

### Task 12: Accuracy, Memory, And Regression Gates

**Files:**
- Modify: `tests/run_deepseek_v4_flash_real_smoke.sh`
- Modify: `docs/design/deepseek_v4_flash_q2_native.md`
- Test: `tests/deepseek_v4_flash/test_direct_decode_real_smoke.py`
- Test: `tests/smoke/test_deepseek_v4_flash_http_smoke.py`

- [ ] **Step 1: Add repeated-request memory smoke**

Add a test that runs three direct one-token decodes and asserts RSS does not grow without bound:

```python
def test_direct_decode_repeated_requests_do_not_leak_unbounded_memory() -> None:
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(3):
        run_one_token_decode()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert after - before < 2_000_000
```

Use a generous threshold and document units.

- [ ] **Step 2: Add bounded script stages**

Update `tests/run_deepseek_v4_flash_real_smoke.sh`:

```bash
timeout 1800 uv run --no-sync pytest tests/deepseek_v4_flash/test_direct_decode_real_smoke.py -q
timeout 1800 uv run --no-sync pytest tests/deepseek_v4_flash/test_direct_greedy_decode_real_smoke.py -q
timeout 1800 uv run --no-sync pytest tests/smoke/test_deepseek_v4_flash_http_smoke.py -q
```

- [ ] **Step 3: Run final bounded validation**

Run:

```bash
tests/run_deepseek_v4_flash_real_smoke.sh
timeout 1200 uv run --no-sync pytest tests/deepseek_v4_flash -q
bash tests/run_regression_suite.sh
```

Expected: all pass. If full regression is slow, record exact timeout/failure and do not claim completion.

- [ ] **Step 4: Update design docs and commit**

Update `docs/design/deepseek_v4_flash_q2_native.md` to move batch=1 greedy decode and initialized REST generation from "Not implemented" to "Implemented" only after the tests above pass.

Run:

```bash
git add tests/run_deepseek_v4_flash_real_smoke.sh docs/design/deepseek_v4_flash_q2_native.md tests/deepseek_v4_flash tests/smoke/test_deepseek_v4_flash_http_smoke.py
git commit -m "test: validate deepseek end to end inference"
```

---

## Stop Conditions

Stop and report `BLOCKED` instead of forcing progress when any of these happen:

- Real tensor layout cannot be mapped to the HF/DS4 architecture with evidence.
- One expert slice cannot be decoded without loading all grouped experts.
- Full layer execution exceeds memory budget before a single block finishes.
- Direct full one-token decode cannot complete under a bounded timeout.
- REST requires model-specific branches in `vllm/engine/` instead of a generic capability hook.

## Completion Criteria

The feature is complete only when all of these are true:

- Direct full-model `forward_full_reference(torch.tensor([token]))` returns finite `[1, vocab]` logits after executing all transformer blocks.
- Direct greedy decode appends at least one generated token ID.
- Initialized `/v1/chat/completions` returns HTTP 200 for `temperature=0`, `max_tokens=1`.
- Context 4096 smoke passes; context 8192 smoke is either passing or explicitly documented as blocked by memory with evidence.
- `tests/run_deepseek_v4_flash_real_smoke.sh` passes.
- `bash tests/run_regression_suite.sh` passes.
- Documentation states the actual implemented capability without claiming performance that has not been measured.

## Self-Review

- Spec coverage: This plan covers tensor audit, expert/attention/block execution, direct greedy decode, tokenizer, LiteEngine integration, REST generation, and final validation.
- Placeholder scan: The plan avoids placeholder wording and names concrete files, tests, and commands.
- Type consistency: The plan consistently uses `DeepSeekV4FlashWeightStore`, `DeepSeekV4CompressedKVCache`, `forward_full_reference()`, and `generate_greedy_reference()` as the direct-model bridge before engine integration.
