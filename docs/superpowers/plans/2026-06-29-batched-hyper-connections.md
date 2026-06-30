# Batched Hyper-Connections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend DeepSeek V4 Flash batched layer forwards and the batched generation kernel to support per-layer `attention_hyper_connection` and `ffn_hyper_connection`, removing the current `NotImplementedError` blocks and enabling the batched e2e workload.

**Architecture:** Add 3-D stream-aware helpers (`[B, hc_mult, hidden_size]`) next to the existing 2-D single-request helpers in `gpu_layers.py`. Then wire them into `deepseek_v4_flash_sliding_layer_forward_batched` and `deepseek_v4_flash_compressed_layer_forward_batched`, mirroring the single-request hyper-connection flow. Finally drop the model-level rejection so `generate_greedy_kernel_batched` can run on the real GGUF.

**Tech Stack:** Python 3.12, PyTorch/ROCm, existing `vllm/model_executor/models/deepseek_v4_flash/` codebase, `uv`, `pytest`, `ruff`.

## Global Constraints

- Use `uv` for all Python/test commands (`uv run pytest ...`, `uv run ruff ...`).
- Keep single-request hyper-connection helpers unchanged; add explicit `*_batched` variants.
- All new code must be typed and pass `uv run ruff check <paths>` and `uv run mypy vllm`.
- GPU tests are gated with `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`.
- No C++ / no direct `import triton`; reuse existing Triton wrappers.
- Synthetic test hyper-connections use `hc_mult=2` or `hc_mult=4` and random small tensors.

---

## File Map

- **`vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`**
  - Add `_GPUHyperConnectionStateBatched` dataclass.
  - Add `_ensure_hyper_connection_streams_batched`.
  - Add `_hyper_connection_pre_cuda_batched`.
  - Add `_hyper_connection_post_cuda_batched`.
  - Modify `deepseek_v4_flash_sliding_layer_forward_batched` to use the helpers.
  - Modify `deepseek_v4_flash_compressed_layer_forward_batched` to use the helpers.
- **`vllm/model_executor/models/deepseek_v4_flash/model.py`**
  - Remove the `_reject_batched_generation_if_hyper_connections()` call in `_generate_greedy_kernel_batched_impl`.
- **`tests/deepseek_v4_flash/test_gpu_batched_hyper_connections.py`** (create)
  - Unit tests for the three batched helpers against the existing single-request helpers.
- **`tests/deepseek_v4_flash/test_gpu_batched_layers.py`**
  - Add hyper-connection variants to fake layer builders.
  - Add `test_batched_sliding_layer_with_hyper_connections_matches_single_slot`.
- **`tests/deepseek_v4_flash/test_gpu_batched_compressed_layers.py`**
  - Add hyper-connection variant to the compressed layer builder.
  - Add `test_batched_compressed_layer_with_hyper_connections_matches_single_slot`.

---

## Task 1: Add Batched Hyper-Connection Helpers

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
- Test: `tests/deepseek_v4_flash/test_gpu_batched_hyper_connections.py`

**Interfaces:**
- Consumes: `DeepSeekV4FlashHyperConnectionTensors`, `DeepSeekV4FlashGPUWeightStager`, existing `_hyper_connection_stream_count`.
- Produces:
  - `_ensure_hyper_connection_streams_batched(hidden, *, stager, hyper_connection) -> torch.Tensor` (shape `[B, hc_mult, hidden_size]`)
  - `_hyper_connection_pre_cuda_batched(streams, *, stager, hyper_connection) -> _GPUHyperConnectionStateBatched`
  - `_hyper_connection_post_cuda_batched(output, residual_streams, state) -> torch.Tensor`

- [ ] **Step 1: Write the failing test**

Create `tests/deepseek_v4_flash/test_gpu_batched_hyper_connections.py`:

```python
from __future__ import annotations

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_F16,
    DeepSeekV4FlashTensor,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_layers import (
    _ensure_hyper_connection_streams_batched,
    _hyper_connection_post_cuda_batched,
    _hyper_connection_pre_cuda_batched,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_weight_staging import (
    DeepSeekV4FlashGPUWeightStager,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashHyperConnectionTensors,
)


class _FakeStore:
    def __init__(self) -> None:
        self.matrices: dict[str, torch.Tensor] = {}
        self.vectors: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        return self.matrices[tensor.name].clone()

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.vectors[(tensor.name, dtype)].clone()


def _tensor(name: str, dims: tuple[int, ...]) -> DeepSeekV4FlashTensor:
    return DeepSeekV4FlashTensor(
        name=name,
        dims=dims,
        tensor_type=GGML_TYPE_F16,
        offset=0,
        nbytes=0,
    )


def _make_hc_tensors(
    *,
    hc_mult: int,
    hidden_size: int,
    name_prefix: str = "hc",
) -> tuple[DeepSeekV4FlashHyperConnectionTensors, _FakeStore]:
    mix_count = 2 * hc_mult + hc_mult * hc_mult
    flat_size = hc_mult * hidden_size
    store = _FakeStore()
    fn_t = _tensor(f"{name_prefix}_fn.weight", (flat_size, mix_count))
    base_t = _tensor(f"{name_prefix}_base.weight", (mix_count,))
    scale_t = _tensor(f"{name_prefix}_scale.weight", (3,))
    store.matrices[fn_t.name] = torch.randn(
        flat_size, mix_count, dtype=torch.float32, device="cuda"
    )
    store.vectors[(base_t.name, torch.float32)] = torch.randn(
        mix_count, dtype=torch.float32, device="cuda"
    )
    store.vectors[(scale_t.name, torch.float32)] = torch.randn(
        3, dtype=torch.float32, device="cuda"
    )
    return (
        DeepSeekV4FlashHyperConnectionTensors(fn=fn_t, base=base_t, scale=scale_t),
        store,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_ensure_hyper_connection_streams_batched_expands_2d() -> None:
    hc, store = _make_hc_tensors(hc_mult=2, hidden_size=8)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    hidden = torch.randn(3, 8, dtype=torch.float32, device="cuda")
    streams = _ensure_hyper_connection_streams_batched(
        hidden, stager=stager, hyper_connection=hc
    )
    assert streams.shape == (3, 2, 8)
    torch.testing.assert_close(streams[0, 0], hidden[0])
    torch.testing.assert_close(streams[0, 1], hidden[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_hyper_connection_pre_post_batched_matches_loop() -> None:
    from vllm.model_executor.models.deepseek_v4_flash.gpu_layers import (
        _ensure_hyper_connection_streams,
        _hyper_connection_post_cuda,
        _hyper_connection_pre_cuda,
    )

    hc, store = _make_hc_tensors(hc_mult=2, hidden_size=8)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    hidden = torch.randn(3, 8, dtype=torch.float32, device="cuda")
    streams = _ensure_hyper_connection_streams_batched(
        hidden, stager=stager, hyper_connection=hc
    )

    state_b = _hyper_connection_pre_cuda_batched(
        streams, stager=stager, hyper_connection=hc
    )

    singles = [
        _hyper_connection_pre_cuda(
            _ensure_hyper_connection_streams(hidden[b], stager=stager, hyper_connection=hc),
            stager=stager,
            hyper_connection=hc,
        )
        for b in range(hidden.shape[0])
    ]

    torch.testing.assert_close(
        state_b.mixed, torch.stack([s.mixed for s in singles])
    )
    torch.testing.assert_close(
        state_b.post, torch.stack([s.post for s in singles])
    )
    torch.testing.assert_close(
        state_b.combine, torch.stack([s.combine for s in singles])
    )

    update = torch.randn(3, 8, dtype=torch.float32, device="cuda")
    out_b = _hyper_connection_post_cuda_batched(
        update, streams, state_b
    )
    singles_out = torch.stack(
        [
            _hyper_connection_post_cuda(
                update[b],
                _ensure_hyper_connection_streams(hidden[b], stager=stager, hyper_connection=hc),
                singles[b],
            )
            for b in range(hidden.shape[0])
        ]
    )
    torch.testing.assert_close(out_b, singles_out)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_batched_hyper_connections.py -v
```

Expected: `ModuleNotFoundError` / `ImportError` for the new helper names, or `AttributeError`.

- [ ] **Step 3: Implement the helpers**

In `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`, after `_GPUHyperConnectionState`, add:

```python
@dataclass(frozen=True)
class _GPUHyperConnectionStateBatched:
    mixed: torch.Tensor
    post: torch.Tensor
    combine: torch.Tensor
```

Add `_ensure_hyper_connection_streams_batched` after `_ensure_hyper_connection_streams`:

```python
def _ensure_hyper_connection_streams_batched(
    hidden: torch.Tensor,
    *,
    stager: DeepSeekV4FlashGPUWeightStager,
    hyper_connection: DeepSeekV4FlashHyperConnectionTensors,
) -> torch.Tensor:
    if not hidden.is_cuda:
        raise ValueError(
            "DeepSeek V4 Flash batched hyper-connection hidden must be a CUDA tensor"
        )
    base = stager.stage_vector(hyper_connection.base)
    hc_mult = _hyper_connection_stream_count(base)
    if hidden.ndim == 3:
        if hidden.shape[1] != hc_mult:
            raise ValueError(
                "batched mHC stream count must match hyper-connection tensors; "
                f"got {hidden.shape[1]} and {hc_mult}"
            )
        return hidden.to(torch.float32)
    if hidden.ndim != 2:
        raise ValueError(
            "batched hidden must be 2-D or stream-shaped 3-D; "
            f"got {hidden.ndim}-D"
        )
    batch, hidden_size = hidden.shape
    return (
        hidden.to(torch.float32)
        .reshape(batch, 1, hidden_size)
        .expand(batch, hc_mult, hidden_size)
        .clone()
    )
```

Add `_hyper_connection_pre_cuda_batched` after `_hyper_connection_pre_cuda`:

```python
def _hyper_connection_pre_cuda_batched(
    streams: torch.Tensor,
    *,
    stager: DeepSeekV4FlashGPUWeightStager,
    hyper_connection: DeepSeekV4FlashHyperConnectionTensors,
) -> _GPUHyperConnectionStateBatched:
    if not streams.is_cuda:
        raise ValueError(
            "DeepSeek V4 Flash batched mHC streams must be CUDA tensors"
        )
    if streams.ndim != 3:
        raise ValueError(f"batched mHC streams must be 3-D; got {streams.ndim}-D")

    fn_weight = stager.stage_matrix(hyper_connection.fn)
    base = stager.stage_vector(hyper_connection.base)
    scale = stager.stage_vector(hyper_connection.scale)
    batch, hc_mult, hidden_size = streams.shape
    mix_count = 2 * hc_mult + hc_mult * hc_mult
    flat_size = hc_mult * hidden_size
    if fn_weight.shape == (flat_size, mix_count):
        projection_weight = fn_weight.to(torch.float32).T
    elif fn_weight.shape == (mix_count, flat_size):
        projection_weight = fn_weight.to(torch.float32)
    else:
        raise ValueError(
            "hyper-connection fn tensor shape does not match batched streams; "
            f"got {tuple(fn_weight.shape)}, expected ({flat_size}, {mix_count})"
        )
    if base.shape != (mix_count,):
        raise ValueError(
            f"hyper-connection base shape must be ({mix_count},); "
            f"got {tuple(base.shape)}"
        )
    if scale.shape != (3,):
        raise ValueError(
            f"hyper-connection scale shape must be (3,); got {tuple(scale.shape)}"
        )

    flat = streams.reshape(batch, flat_size).to(torch.float32)
    mixes = torch.matmul(flat, projection_weight.T) * torch.rsqrt(
        flat.pow(2).mean(dim=-1, keepdim=True) + 1e-6
    )
    scale_f32 = scale.to(torch.float32)
    scaled_scale = torch.cat(
        [
            scale_f32[0:1].expand(hc_mult),
            scale_f32[1:2].expand(hc_mult),
            scale_f32[2:3].expand(hc_mult * hc_mult),
        ]
    )
    mixes = mixes * scaled_scale
    mixes = mixes + base.to(torch.float32)

    pre = torch.sigmoid(mixes[:, :hc_mult]) + 1e-6
    post = 2.0 * torch.sigmoid(mixes[:, hc_mult : 2 * hc_mult])
    combine_scores = mixes[:, 2 * hc_mult :].reshape(batch, hc_mult, hc_mult)
    combine = torch.softmax(combine_scores, dim=-1) + 1e-6
    combine = combine / (combine.sum(dim=1, keepdim=True) + 1e-6)
    for _ in range(1, 20):
        combine = combine / (combine.sum(dim=-1, keepdim=True) + 1e-6)
        combine = combine / (combine.sum(dim=1, keepdim=True) + 1e-6)

    mixed = (pre.unsqueeze(-1) * streams.to(torch.float32)).sum(dim=1)
    return _GPUHyperConnectionStateBatched(
        mixed=mixed, post=post, combine=combine
    )
```

Add `_hyper_connection_post_cuda_batched` after `_hyper_connection_post_cuda`:

```python
def _hyper_connection_post_cuda_batched(
    output: torch.Tensor,
    residual_streams: torch.Tensor,
    state: _GPUHyperConnectionStateBatched,
) -> torch.Tensor:
    if not output.is_cuda or not residual_streams.is_cuda:
        raise ValueError(
            "DeepSeek V4 Flash batched mHC post inputs must be CUDA tensors"
        )
    if output.ndim != 2:
        raise ValueError(f"batched mHC output must be 2-D; got {output.ndim}-D")
    if residual_streams.ndim != 3:
        raise ValueError(
            f"batched mHC residual streams must be 3-D; got {residual_streams.ndim}-D"
        )
    batch, hc_mult, hidden_size = residual_streams.shape
    if output.shape != (batch, hidden_size):
        raise ValueError(
            f"batched mHC output shape must be ({batch}, {hidden_size}); "
            f"got {tuple(output.shape)}"
        )
    if state.post.shape != (batch, hc_mult):
        raise ValueError(
            f"batched mHC post shape must be ({batch}, {hc_mult}); "
            f"got {tuple(state.post.shape)}"
        )
    if state.combine.shape != (batch, hc_mult, hc_mult):
        raise ValueError(
            "batched mHC combine shape must match residual streams; "
            f"got {tuple(state.combine.shape)}"
        )
    residual_mix = state.combine.to(torch.float32).transpose(-1, -2).matmul(
        residual_streams.to(torch.float32)
    )
    return (
        state.post.reshape(batch, hc_mult, 1).to(torch.float32)
        * output.to(torch.float32).unsqueeze(1)
        + residual_mix
    )
```

- [ ] **Step 4: Run the helper test**

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_batched_hyper_connections.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/deepseek_v4_flash/test_gpu_batched_hyper_connections.py \
        vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py
git commit -m "feat(deepseek): batched hyper-connection helpers"
```

---

## Task 2: Integrate Hyper-Connections into Batched Sliding Layer Forward

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
- Test: `tests/deepseek_v4_flash/test_gpu_batched_layers.py`

**Interfaces:**
- Consumes: `_ensure_hyper_connection_streams_batched`, `_hyper_connection_pre_cuda_batched`, `_hyper_connection_post_cuda_batched`.
- Produces: `deepseek_v4_flash_sliding_layer_forward_batched` returns `[B, hidden_size]` when no hyper-connection is present and `[B, hc_mult, hidden_size]` when hyper-connections are active.

- [ ] **Step 1: Add hyper-connection builders to the test fake layer store**

In `tests/deepseek_v4_flash/test_gpu_batched_layers.py`, add a helper that takes a layer/store tuple and injects fake hyper-connection tensors:

```python
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashHyperConnectionTensors,
)


def _add_hyper_connections_to_sliding_layer(
    layer: DeepSeekV4FlashLayerSemanticBindings,
    store: _FakeLayerStore,
    *,
    hc_mult: int = 2,
) -> DeepSeekV4FlashLayerSemanticBindings:
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    mix_count = 2 * hc_mult + hc_mult * hc_mult
    flat_size = hc_mult * hidden_size
    fn_t = _tensor("blk.0.attn_hc_fn.weight", (flat_size, mix_count))
    base_t = _tensor("blk.0.attn_hc_base.weight", (mix_count,))
    scale_t = _tensor("blk.0.attn_hc_scale.weight", (3,))
    ffn_fn_t = _tensor("blk.0.ffn_hc_fn.weight", (flat_size, mix_count))
    ffn_base_t = _tensor("blk.0.ffn_hc_base.weight", (mix_count,))
    ffn_scale_t = _tensor("blk.0.ffn_hc_scale.weight", (3,))

    torch.manual_seed(44)
    store.matrices[fn_t.name] = torch.randn(
        flat_size, mix_count, dtype=torch.float32, device="cuda"
    ) * 0.01
    store.vectors[(base_t.name, torch.float32)] = torch.randn(
        mix_count, dtype=torch.float32, device="cuda"
    )
    store.vectors[(scale_t.name, torch.float32)] = torch.randn(
        3, dtype=torch.float32, device="cuda"
    )
    store.matrices[ffn_fn_t.name] = torch.randn(
        flat_size, mix_count, dtype=torch.float32, device="cuda"
    ) * 0.01
    store.vectors[(ffn_base_t.name, torch.float32)] = torch.randn(
        mix_count, dtype=torch.float32, device="cuda"
    )
    store.vectors[(ffn_scale_t.name, torch.float32)] = torch.randn(
        3, dtype=torch.float32, device="cuda"
    )

    return dataclasses.replace(
        layer,
        attention_hyper_connection=DeepSeekV4FlashHyperConnectionTensors(
            fn=fn_t, base=base_t, scale=scale_t
        ),
        ffn_hyper_connection=DeepSeekV4FlashHyperConnectionTensors(
            fn=ffn_fn_t, base=ffn_base_t, scale=ffn_scale_t
        ),
    )
```

Add the test:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_sliding_layer_with_hyper_connections_matches_single_slot() -> None:
    layer, store = _make_real_sliding_layer()
    layer = _add_hyper_connections_to_sliding_layer(layer, store, hc_mult=2)
    backend = _NullAttentionBackend()
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden_a = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    hidden_b = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    hidden_batched = torch.stack([hidden_a, hidden_b])

    output_a = deepseek_v4_flash_sliding_layer_forward(
        hidden_a,
        layer=layer,
        stager=stager,
        backend=backend,
        state=_make_state(),
        token_idx=0,
        router_top_k=1,
    )
    output_b = deepseek_v4_flash_sliding_layer_forward(
        hidden_b,
        layer=layer,
        stager=stager,
        backend=backend,
        state=_make_state(),
        token_idx=0,
        router_top_k=1,
    )

    output_batched = deepseek_v4_flash_sliding_layer_forward_batched(
        hidden_batched,
        layer=layer,
        stager=stager,
        backend=backend,
        states=[_make_state(), _make_state()],
        token_indices=[0, 0],
        router_top_k=1,
    )

    # With hyper-connections the layer returns stream-shaped hidden.
    assert output_batched.shape == (2, 2, hidden_size)
    assert output_batched.device.type == "cuda"
    torch.testing.assert_close(output_batched[0], output_a)
    torch.testing.assert_close(output_batched[1], output_b)
```

- [ ] **Step 2: Run the new test to verify it fails**

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_batched_layers.py::test_batched_sliding_layer_with_hyper_connections_matches_single_slot -v
```

Expected: FAIL with `NotImplementedError: batched sliding layer does not support ...`.

- [ ] **Step 3: Modify `deepseek_v4_flash_sliding_layer_forward_batched`**

In `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`, replace the rejection block:

```python
    if layer.attention_hyper_connection is not None:
        raise NotImplementedError(
            "batched sliding layer does not support attention hyper-connections"
        )
    if layer.ffn_hyper_connection is not None:
        raise NotImplementedError(
            "batched sliding layer does not support ffn hyper-connections"
        )
```

with the batched hyper-connection setup:

```python
    uses_hyper_connection = (
        layer.attention_hyper_connection is not None
        or layer.ffn_hyper_connection is not None
    )
    attention_residual_streams: torch.Tensor | None = None
    attention_hc_state: _GPUHyperConnectionStateBatched | None = None
    if uses_hyper_connection:
        stream_hc = layer.attention_hyper_connection or layer.ffn_hyper_connection
        if stream_hc is None:
            raise AssertionError("hyper-connection state was not validated")
        attention_residual_streams = _ensure_hyper_connection_streams_batched(
            hidden,
            stager=stager,
            hyper_connection=stream_hc,
        )
        if layer.attention_hyper_connection is None:
            attn_source = attention_residual_streams.mean(dim=1).to(torch.float32)
        else:
            attention_hc_state = _hyper_connection_pre_cuda_batched(
                attention_residual_streams,
                stager=stager,
                hyper_connection=layer.attention_hyper_connection,
            )
            attn_source = attention_hc_state.mixed
    else:
        attn_source = hidden
```

Change the validation at the top to accept 3-D stream input:

```python
    if hidden.ndim == 2:
        batch = hidden.shape[0]
    elif hidden.ndim == 3:
        batch = hidden.shape[0]
    else:
        raise ValueError(
            f"batched sliding layer expects 2-D or stream-shaped 3-D hidden; "
            f"got {hidden.ndim}-D"
        )
```

After the attention block, replace:

```python
    hidden_after_attn = deepseek_v4_flash_residual_hyper_connection(
        hidden,
        attn_update,
        stager=stager,
        hyper_connection=None,
    )
```

with:

```python
    if uses_hyper_connection:
        if attention_residual_streams is None:
            raise AssertionError("attention residual streams were not initialized")
        if attention_hc_state is None:
            hidden_after_attn = attention_residual_streams + attn_update.unsqueeze(1)
        else:
            hidden_after_attn = _hyper_connection_post_cuda_batched(
                attn_update,
                attention_residual_streams,
                attention_hc_state,
            )
    else:
        hidden_after_attn = deepseek_v4_flash_residual_hyper_connection(
            hidden,
            attn_update,
            stager=stager,
            hyper_connection=None,
        )
```

After `ffn_input` is built, add the FFN hyper-connection setup before the MoE loop:

```python
    if uses_hyper_connection:
        ffn_residual_streams = hidden_after_attn
        if layer.ffn_hyper_connection is None:
            ffn_source = ffn_residual_streams.mean(dim=1).to(torch.float32)
            ffn_hc_state = None
        else:
            ffn_hc_state = _hyper_connection_pre_cuda_batched(
                ffn_residual_streams,
                stager=stager,
                hyper_connection=layer.ffn_hyper_connection,
            )
            ffn_source = ffn_hc_state.mixed
    else:
        ffn_source = hidden_after_attn
        ffn_residual_streams = None
        ffn_hc_state = None
```

Use `ffn_source` in the MoE loop: replace `ffn_input[b]` with `ffn_source[b]` inside the loop.

Finally replace the return:

```python
    return deepseek_v4_flash_residual_hyper_connection(
        hidden_after_attn,
        moe_update,
        stager=stager,
        hyper_connection=None,
    )
```

with:

```python
    if uses_hyper_connection:
        if ffn_residual_streams is None:
            raise AssertionError("FFN residual streams were not initialized")
        if ffn_hc_state is None:
            return ffn_residual_streams + moe_update.unsqueeze(1)
        return _hyper_connection_post_cuda_batched(
            moe_update,
            ffn_residual_streams,
            ffn_hc_state,
        )
    return deepseek_v4_flash_residual_hyper_connection(
        hidden_after_attn,
        moe_update,
        stager=stager,
        hyper_connection=None,
    )
```

- [ ] **Step 4: Run the sliding layer tests**

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_batched_layers.py -v
```

Expected: PASS, including the new hyper-connection test.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py \
        tests/deepseek_v4_flash/test_gpu_batched_layers.py
git commit -m "feat(deepseek): hyper-connections in batched sliding layer"
```

---

## Task 3: Integrate Hyper-Connections into Batched Compressed Layer Forward

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
- Test: `tests/deepseek_v4_flash/test_gpu_batched_compressed_layers.py`

**Interfaces:**
- Consumes: same batched helpers as Task 2.
- Produces: `deepseek_v4_flash_compressed_layer_forward_batched` supports hyper-connections and matches the single-slot compressed layer output.

- [ ] **Step 1: Add hyper-connection builders to the compressed test fake layer store**

In `tests/deepseek_v4_flash/test_gpu_batched_compressed_layers.py`, add an injection helper analogous to Task 2, using the layer index (default `2`) for tensor names and `dataclasses.replace` to set `attention_hyper_connection` and `ffn_hyper_connection`.

Add the test:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_compressed_layer_with_hyper_connections_matches_single_slot() -> None:
    store, layer = _make_real_compressed_layer_binding(layer_index=2)
    layer = _add_hyper_connections_to_compressed_layer(store, layer, hc_mult=2)
    backend = _RecordingCompressedBackend()
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden_a = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    hidden_b = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    hidden_batched = torch.stack([hidden_a, hidden_b])

    output_a = deepseek_v4_flash_compressed_layer_forward(
        hidden_a,
        layer=layer,
        stager=stager,
        backend=backend,
        state=_make_request_state(),
        token_idx=0,
        router_top_k=1,
    )
    output_b = deepseek_v4_flash_compressed_layer_forward(
        hidden_b,
        layer=layer,
        stager=stager,
        backend=backend,
        state=_make_request_state(),
        token_idx=0,
        router_top_k=1,
    )

    output_batched = deepseek_v4_flash_compressed_layer_forward_batched(
        hidden_batched,
        layer=layer,
        stager=stager,
        backend=backend,
        states=[_make_request_state(), _make_request_state()],
        token_indices=[0, 0],
        router_top_k=1,
    )

    assert output_batched.shape == (2, 2, hidden_size)
    assert output_batched.device.type == "cuda"
    torch.testing.assert_close(output_batched[0], output_a)
    torch.testing.assert_close(output_batched[1], output_b)
```

- [ ] **Step 2: Run the new test to verify it fails**

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_batched_compressed_layers.py::test_batched_compressed_layer_with_hyper_connections_matches_single_slot -v
```

Expected: FAIL with `NotImplementedError: batched compressed layer does not support ...`.

- [ ] **Step 3: Modify `deepseek_v4_flash_compressed_layer_forward_batched`**

Apply the same pattern as Task 2:

1. Allow 3-D hidden input in validation.
2. Remove the `NotImplementedError` block.
3. Add `uses_hyper_connection` setup, `attention_residual_streams`, `attention_hc_state`, and derive `attn_source`.
4. Replace `attn_update` residual with hyper-connection post or simple residual.
5. Derive `ffn_source`/`ffn_residual_streams`/`ffn_hc_state`.
6. Use `ffn_source[b]` inside the MoE loop.
7. Return hyper-connection post or simple residual.

The single-slot compressed layer already contains the same hyper-connection logic, so mirror it exactly. Key code blocks are the same as Task 2 except the function name and surrounding compressed-attention code.

- [ ] **Step 4: Run the compressed layer tests**

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_batched_compressed_layers.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py \
        tests/deepseek_v4_flash/test_gpu_batched_compressed_layers.py
git commit -m "feat(deepseek): hyper-connections in batched compressed layer"
```

---

## Task 4: Remove Model-Level Batched Generation Rejection

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`

**Interfaces:**
- Consumes: working batched layer forwards from Tasks 2 and 3.
- Produces: `generate_greedy_kernel_batched` no longer rejects models with hyper-connections.

- [ ] **Step 1: Delete the rejection call**

In `_generate_greedy_kernel_batched_impl`, remove:

```python
        self._reject_batched_generation_if_hyper_connections()
```

Keep the helper method `_reject_batched_generation_if_hyper_connections` in place (it is harmless and may be useful for diagnostics), or delete it if `ruff`/`mypy` complains about unused code. Defer the deletion decision until after lint.

- [ ] **Step 2: Run the existing batched generation test**

```bash
uv run pytest tests/deepseek_v4_flash/test_model_kernel_generate_batched.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/model.py
git commit -m "feat(deepseek): allow batched generation with hyper-connections"
```

---

## Task 5: Add/Update Unit Tests and Run GPU Test Suite

**Files:**
- Test: `tests/deepseek_v4_flash/test_gpu_batched_hyper_connections.py`
- Test: `tests/deepseek_v4_flash/test_gpu_batched_layers.py`
- Test: `tests/deepseek_v4_flash/test_gpu_batched_compressed_layers.py`
- Test: `tests/deepseek_v4_flash/test_model_kernel_generate_batched.py`

- [ ] **Step 1: Add edge-case tests for batched helpers**

In `tests/deepseek_v4_flash/test_gpu_batched_hyper_connections.py`, add:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_ensure_hyper_connection_streams_batched_preserves_3d() -> None:
    hc, store = _make_hc_tensors(hc_mult=2, hidden_size=8)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    streams = torch.randn(3, 2, 8, dtype=torch.float32, device="cuda")
    out = _ensure_hyper_connection_streams_batched(
        streams, stager=stager, hyper_connection=hc
    )
    assert out is not streams  # clones
    torch.testing.assert_close(out, streams)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_hyper_connection_post_batched_shape_validation() -> None:
    hc, store = _make_hc_tensors(hc_mult=2, hidden_size=8)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    streams = torch.randn(3, 2, 8, dtype=torch.float32, device="cuda")
    state = _hyper_connection_pre_cuda_batched(streams, stager=stager, hyper_connection=hc)
    bad_output = torch.randn(3, 4, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="batched mHC output shape must be"):
        _hyper_connection_post_cuda_batched(bad_output, streams, state)
```

- [ ] **Step 2: Run the focused DeepSeek test subset**

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_batched_hyper_connections.py \
              tests/deepseek_v4_flash/test_gpu_batched_layers.py \
              tests/deepseek_v4_flash/test_gpu_batched_compressed_layers.py \
              tests/deepseek_v4_flash/test_model_kernel_generate_batched.py -v
```

Expected: PASS.

- [ ] **Step 3: Lint the touched files**

```bash
uv run ruff check vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py \
                   vllm/model_executor/models/deepseek_v4_flash/model.py \
                   tests/deepseek_v4_flash/test_gpu_batched_hyper_connections.py \
                   tests/deepseek_v4_flash/test_gpu_batched_layers.py \
                   tests/deepseek_v4_flash/test_gpu_batched_compressed_layers.py
uv run ruff format vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py \
                   vllm/model_executor/models/deepseek_v4_flash/model.py \
                   tests/deepseek_v4_flash/test_gpu_batched_hyper_connections.py \
                   tests/deepseek_v4_flash/test_gpu_batched_layers.py \
                   tests/deepseek_v4_flash/test_gpu_batched_compressed_layers.py
```

Fix any lint errors and re-run tests.

- [ ] **Step 4: Commit**

```bash
git add tests/deepseek_v4_flash/test_gpu_batched_hyper_connections.py \
        tests/deepseek_v4_flash/test_gpu_batched_layers.py \
        tests/deepseek_v4_flash/test_gpu_batched_compressed_layers.py
git commit -m "test(deepseek): hyper-connection edge cases and layer parity"
```

---

## Task 6: Run the Real Batched E2E Benchmark

**Files:**
- Run: `tests/e2e_full_benchmark.py` with `--deepseek-batched-engine`
- Run: `tests/tools/run_deepseek_v4_flash_gpu_smoke_batched.py`

- [ ] **Step 1: Run the direct batched smoke tool**

```bash
uv run python tests/tools/run_deepseek_v4_flash_gpu_smoke_batched.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-tokens 16 \
  --batch-size 2 \
  --prompt-length 1 \
  --repeat 1 \
  --profile-json /tmp/deepseek_batched_hc.json
```

Expected: exits 0 and writes a JSON profile. If it still fails with a hyper-connection error, revisit Tasks 2–4.

- [ ] **Step 2: Run the e2e benchmark with the batched-engine workload**

```bash
uv run python tests/e2e_full_benchmark.py \
  --models deepseek_v4_flash_q2_gguf \
  --deepseek-batched-engine \
  --deepseek-concurrent 2 \
  --json-out /tmp/e2e_deepseek_batched_hc.json
```

Expected: no longer skips; reports aggregate/decode TPS.

- [ ] **Step 3: Report the numbers**

Print the summary from the JSON:

```bash
python3 - <<'PY'
import json
with open('/tmp/e2e_deepseek_batched_hc.json') as f:
    data = json.load(f)
s = data['summary']['deepseek_v4_flash_q2_gguf']
print(f"aggregate_tps={s['aggregate_tps']:.3f}")
print(f"decode_tps_aggregate={s['decode_tps_aggregate']:.3f}")
print(f"decode_p50_ms={s['decode_p50_ms']:.1f}")
PY
```

- [ ] **Step 4: Commit if any benchmark harness tweaks were needed**

If no changes are needed, no commit. If the e2e harness required a fix, commit it separately.

---

## Self-Review

1. **Spec coverage:**
   - Batched ensure/pre/post helpers → Task 1.
   - Batched sliding layer hyper-connections → Task 2.
   - Batched compressed layer hyper-connections → Task 3.
   - Model-level batched generation no longer rejects hyper-connections → Task 4.
   - Unit tests and real-GPU e2e measurement → Tasks 5 and 6.
2. **Placeholder scan:** No `TBD`, `TODO`, or vague "handle edge cases" steps. Every step contains code or an exact command.
3. **Type consistency:** `_GPUHyperConnectionStateBatched` fields are `torch.Tensor` with documented batch shapes; layer forwards use `attention_hc_state: _GPUHyperConnectionStateBatched | None` consistently.
