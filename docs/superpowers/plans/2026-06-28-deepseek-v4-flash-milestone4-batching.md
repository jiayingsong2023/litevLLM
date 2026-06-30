# DeepSeek V4 Flash Decode Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add static decode batching to the DeepSeek V4 Flash direct path so 2–4 concurrent decode positions run together, improving device occupancy and aggregate decode throughput while keeping per-request state isolated.

**Architecture:** Keep each request in its own `DeepSeekV4FlashGPURequestState` (single-slot KV cache, workspace, compressor state). Batching happens at the layer level: norms and matrix projections run on `[batch, hidden_size]`, while attention, MoE payload staging, and compressor updates loop per-slot using each request's own state. This avoids a risky rewrite of the KV-cache layout and unblocks throughput gains immediately.

**Tech Stack:** Python 3.12, PyTorch/ROCm, existing `vllm/model_executor/models/deepseek_v4_flash/*` modules, `uv`, `pytest`.

## Global Constraints

- Python 3.12 only; use `uv` for all Python commands.
- The batched path must remain opt-in; existing `generate_greedy_kernel(batch=1)` behavior and all current tests must keep passing.
- Per-request KV cache, compressor state, and MoE workspace must stay isolated.
- CUDA/HIP graph capture from Milestone 3 is explicitly disabled for batch size > 1 in this milestone. Do not extend graph capture to batched decode.
- All new code must pass `uv run ruff check .` and `uv run ruff format .` for touched files.
- New kernel paths require a PyTorch-reference correctness test.
- Before PR: `bash tests/run_regression_suite.sh` and `bash tests/run_deepseek_v4_flash_real_smoke.sh` must pass.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py` | Batched layer forwards, batched projection helpers, per-slot attention/MoE loops. |
| `vllm/model_executor/models/deepseek_v4_flash/model.py` | Batched token embedding, batched `_forward_kernel_token_step`, `generate_greedy_kernel_batched`, batched output streams/argmax. |
| `vllm/model_executor/models/deepseek_v4_flash/attention.py` | Per-token RoPE application already exists; no changes required for the per-slot loop approach. |
| `vllm/engine/lite_engine.py` | New `generate_deepseek_v4_flash_greedy_batched` entry point and `max_active_requests > 1` for the direct DeepSeek path. |
| `tests/deepseek_v4_flash/test_gpu_batched_projections.py` | Unit tests for batched Q8/staged matrix projections and RMSNorm. |
| `tests/deepseek_v4_flash/test_gpu_batched_layers.py` | Unit tests for batched sliding/compressed layer forwards with isolated fake states. |
| `tests/deepseek_v4_flash/test_model_kernel_generate_batched.py` | Unit tests for `generate_greedy_kernel_batched` with mocked/fake weights. |
| `tests/tools/deepseek_v4_flash_quality_smoke.py` | New `--batch-size N` flag and aggregate throughput reporting. |

---

## Task 1: Batched Matrix Projection Helpers

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py:111-235`
- Test: `tests/deepseek_v4_flash/test_gpu_batched_projections.py` (create)

**Interfaces:**
- Consumes: existing `deepseek_v4_flash_rms_norm`, `deepseek_v4_flash_staged_matrix_projection`, `_project_tensor`, `deepseek_v4_flash_q8_0_tensor_projection`.
- Produces: `deepseek_v4_flash_rms_norm` and projection helpers accept `hidden` of shape `[hidden_size]` or `[batch, hidden_size]` and return the same rank. `_project_tensor` returns `[batch, out_features]` when input is 2-D.

- [ ] **Step 1: Write the failing test for batched staged projection**

```python
# tests/deepseek_v4_flash/test_gpu_batched_projections.py
import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.gpu_layers import (
    deepseek_v4_flash_staged_matrix_projection,
)


def test_batched_staged_matrix_projection():
    device = torch.device("cuda")
    hidden_size = 16
    out_features = 8
    batch = 3
    hidden = torch.randn(batch, hidden_size, device=device, dtype=torch.float32)
    weight = torch.randn(out_features, hidden_size, device=device, dtype=torch.float32)
    out = deepseek_v4_flash_staged_matrix_projection(hidden, weight)
    expected = torch.matmul(hidden, weight.T)
    assert out.shape == (batch, out_features)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/jack/work/FastInference
uv run pytest tests/deepseek_v4_flash/test_gpu_batched_projections.py::test_batched_staged_matrix_projection -v
```

Expected: FAIL with `ValueError: hidden must be 1-D for batch=1; got 2-D`.

- [ ] **Step 3: Implement batched staged projection**

Edit `deepseek_v4_flash_staged_matrix_projection` in `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`:

```python
def deepseek_v4_flash_staged_matrix_projection(
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not hidden.is_cuda or not weight.is_cuda:
        raise ValueError("DeepSeek V4 Flash projection inputs must be CUDA tensors")
    if hidden.ndim not in (1, 2):
        raise ValueError(f"hidden must be 1-D or 2-D; got {hidden.ndim}-D")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D; got {weight.ndim}-D")

    hidden_f32 = hidden.to(torch.float32)
    weight_f32 = weight.to(torch.float32)
    if weight.shape[1] == hidden.shape[-1]:
        out = torch.matmul(hidden_f32, weight_f32.T)
    elif weight.shape[0] == hidden.shape[-1]:
        out = torch.matmul(hidden_f32, weight_f32)
    else:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} incompatible with hidden {tuple(hidden.shape)}"
        )
    return out
```

- [ ] **Step 4: Verify RMSNorm is already batched and add a test**

```python
def test_batched_rms_norm():
    device = torch.device("cuda")
    hidden_size = 16
    batch = 3
    hidden = torch.randn(batch, hidden_size, device=device, dtype=torch.float16)
    weight = torch.randn(hidden_size, device=device, dtype=torch.float16)
    from vllm.model_executor.models.deepseek_v4_flash.gpu_layers import (
        deepseek_v4_flash_rms_norm,
    )
    out = deepseek_v4_flash_rms_norm(hidden, weight)
    assert out.shape == (batch, hidden_size)
```

Run: `uv run pytest tests/deepseek_v4_flash/test_gpu_batched_projections.py -v` — PASS.

- [ ] **Step 5: Extend `_project_tensor` and Q8 projection to 2-D**

For the first milestone, `_project_tensor` may fall back to looping over the batch dimension when it encounters a Q8_0 tensor with 2-D hidden:

```python
def _project_tensor(
    hidden: torch.Tensor,
    tensor: DeepSeekV4FlashTensor,
    stager: DeepSeekV4FlashGPUWeightStager,
) -> torch.Tensor:
    if hidden.ndim == 2 and _can_project_q8_0(hidden, tensor):
        # Batch loop until a batched Q8 kernel is added.
        return torch.stack(
            [
                deepseek_v4_flash_q8_0_tensor_projection(
                    hidden[b], tensor, stager
                )
                for b in range(hidden.shape[0])
            ]
        )
    if hidden.ndim == 2:
        return torch.stack(
            [
                deepseek_v4_flash_staged_matrix_projection(
                    hidden[b], stager.stage_matrix(tensor)
                )
                for b in range(hidden.shape[0])
            ]
        )
    if _can_project_q8_0(hidden, tensor):
        return deepseek_v4_flash_q8_0_tensor_projection(hidden, tensor, stager)
    return deepseek_v4_flash_staged_matrix_projection(
        hidden, stager.stage_matrix(tensor)
    )
```

Add a test with a fake Q8 tensor (or reuse existing Q8 fixture) and a 2-D hidden input.

- [ ] **Step 6: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_gpu_batched_projections.py
uv run ruff check vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_gpu_batched_projections.py
uv run ruff format vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_gpu_batched_projections.py
git commit -m "feat(deepseek): batched linear projections for decode batching"
```

---

## Task 2: Batched Sliding Layer Forward

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py:529-790`
- Test: `tests/deepseek_v4_flash/test_gpu_batched_layers.py` (create)

**Interfaces:**
- Consumes: `deepseek_v4_flash_rms_norm`, `_project_tensor`, `_run_real_sliding_attention`, `_run_sliding_moe`, hyper-connection helpers.
- Produces: `deepseek_v4_flash_sliding_layer_forward_batched(hidden: [B, H], states: list[State], token_indices: list[int], token_id_tensors: torch.Tensor | None, ...)` returning `[B, H]`.

- [ ] **Step 1: Write failing test for batched sliding layer**

Create `tests/deepseek_v4_flash/test_gpu_batched_layers.py` with a test that builds two independent fake single-slot states, runs the batched sliding layer on `[2, hidden_size]`, and verifies each output matches the single-slot path.

- [ ] **Step 2: Implement `deepseek_v4_flash_sliding_layer_forward_batched`**

Strategy inside the function:

```python
def deepseek_v4_flash_sliding_layer_forward_batched(
    hidden: torch.Tensor,
    *,
    layer: DeepSeekV4FlashLayerSemanticBindings,
    stager: DeepSeekV4FlashGPUWeightStager,
    backend: DeepSeekV4FlashGPUBackend | _SlidingLayerBackend,
    states: list[DeepSeekV4FlashGPURequestState],
    token_indices: list[int],
    token_id_tensors: torch.Tensor | None = None,
    router_top_k: int = DEEPSEEK_V4_FLASH_SHAPE.num_experts_per_tok,
    use_reference_rope: bool = False,
) -> torch.Tensor:
    if hidden.ndim != 2:
        raise ValueError(f"batched sliding layer expects 2-D hidden; got {hidden.ndim}-D")
    batch = hidden.shape[0]
    if len(states) != batch or len(token_indices) != batch:
        raise ValueError("states and token_indices must match batch size")

    # Norms and attention projections are batched.
    attention_norm = stager.stage_vector(_required_tensor(layer.attention_norm, "attention_norm"))
    normed = deepseek_v4_flash_rms_norm(hidden, attention_norm)
    # ... compute q, k, v projections using _project_tensor (now 2-D capable)

    # Attention and MoE are per-slot because KV windows and expert payloads differ.
    outputs = []
    for b in range(batch):
        out_b = _run_sliding_attention_for_slot(
            q[b], k[b], v[b],
            state=states[b],
            token_idx=token_indices[b],
            layer=layer,
            backend=backend,
            use_reference_rope=use_reference_rope,
        )
        outputs.append(out_b)
    attn_out = torch.stack(outputs)

    # Batched output projection and residual addition.
    # ...

    # Batched FFN norm, per-slot MoE, batched output projection.
    ffn_out = torch.stack([
        _run_sliding_moe(
            ffn_normed[b],
            layer=layer,
            stager=stager,
            backend=backend,
            state=states[b],
            token_idx=token_indices[b],
            token_id_tensor=token_id_tensors[b] if token_id_tensors is not None else None,
            router_top_k=router_top_k,
        )
        for b in range(batch)
    ])

    # ... combine attention residual and ffn output, return [B, H]
    return hidden + ffn_out  # simplified; follow existing residual logic
```

The exact residual/attention-output logic must mirror the existing `deepseek_v4_flash_sliding_layer_forward`. Factor out `_run_sliding_attention_for_slot` by extracting the attention body from the single-slot function so both paths share code.

- [ ] **Step 3: Run layer tests**

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_batched_layers.py -v
```

- [ ] **Step 4: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_gpu_batched_layers.py
uv run ruff check vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_gpu_batched_layers.py
uv run ruff format vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_gpu_batched_layers.py
git commit -m "feat(deepseek): batched sliding layer forward with per-slot attention/moe"
```

---

## Task 3: Batched Compressed Layer Forward

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py:792-1130`
- Test: extend `tests/deepseek_v4_flash/test_gpu_batched_layers.py`

**Interfaces:**
- Consumes: batched projections, `_update_compressor_state`, `_write_compressed_runtime_row`, `_run_real_sliding_attention` for compressed layers with sliding tensors, `_run_sliding_moe`.
- Produces: `deepseek_v4_flash_compressed_layer_forward_batched(hidden: [B, H], states: list[State], token_indices: list[int], ...)` returning `[B, H]`.

- [ ] **Step 1: Write failing test for batched compressed layer**

Add a test that runs the compressed layer on two independent fake states and compares against the single-slot path.

- [ ] **Step 2: Implement `deepseek_v4_flash_compressed_layer_forward_batched`**

Same pattern as Task 2: batched norms/projections, per-slot compressor update, per-slot attention (when the layer has real sliding tensors), per-slot MoE. Reuse the existing single-slot helper functions by calling them inside the loop with `hidden[b]` and `states[b]`.

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_batched_layers.py -v
```

- [ ] **Step 4: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_gpu_batched_layers.py
uv run ruff check vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_gpu_batched_layers.py
uv run ruff format vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_gpu_batched_layers.py
git commit -m "feat(deepseek): batched compressed layer forward with per-slot compressor/attention/moe"
```

---

## Task 4: Batched Model Kernel and Greedy Generation

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py:1352-1700`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py:385-620`
- Test: `tests/deepseek_v4_flash/test_model_kernel_generate_batched.py` (create)

**Interfaces:**
- Consumes: batched layer forwards, batched projections, `_output_logits_chunked_cuda`, `_output_token_argmax_chunked_cuda`.
- Produces: `_stage_token_embedding_tensor_cuda` supports `[batch]` token ids; `_kernel_output_streams` supports `[batch, hidden_size]`; `_output_token_argmax_chunked_cuda` returns `[batch]`; `generate_greedy_kernel_batched(input_ids_list, max_tokens)` returns list of output tensors.

- [ ] **Step 1: Write failing test for batched token embedding and argmax**

```python
def test_batched_token_embedding_and_argmax(fake_model):
    device = torch.device("cuda")
    batch = 2
    token_ids = torch.tensor([100, 200], dtype=torch.long, device=device)
    # exercise fake_model._stage_token_embedding_tensor_cuda
    hidden = fake_model._stage_token_embedding_tensor_cuda(
        fake_model._require_weight_store(),
        fake_model._get_gpu_weight_stager(device),
        token_ids,
        device=device,
    )
    assert hidden.shape == (batch, fake_model.shape.hidden_size)
```

- [ ] **Step 2: Implement batched token embedding**

Change `_stage_token_embedding_tensor_cuda` to accept a 1-D tensor of arbitrary length:

```python
if token_id_tensor.ndim != 1:
    raise ValueError(f"token_id_tensor must be 1-D; got {token_id_tensor.ndim}-D")
hidden = tensor_f16.index_select(1, token_id_tensor).T  # [batch, hidden_size]
return hidden
```

- [ ] **Step 3: Implement batched output streams and argmax**

Update `_kernel_output_streams`:

```python
if hidden.ndim == 1:
    return hidden.to(torch.float32).reshape(1, -1).expand(4, -1).clone()
if hidden.ndim == 2:
    # [batch, hidden] -> [batch, 4, hidden]
    return hidden.to(torch.float32).unsqueeze(1).expand(-1, 4, -1).clone()
```

Update `_output_token_argmax_chunked_cuda` to return `[batch]` next token ids by looping over the batch dimension for the chunked Q8 output projection (or using a batched projection if already available). Keep the existing single-token path unchanged.

- [ ] **Step 4: Add `_forward_kernel_token_step_batched`**

```python
def _forward_kernel_token_step_batched(
    self,
    *,
    token_id_tensor: torch.Tensor,  # [batch]
    states: list[DeepSeekV4FlashGPURequestState],
    token_indices: list[int],
    device: torch.device,
    advance_state: bool = True,
) -> torch.Tensor:
    store = self._require_weight_store()
    self.gpu_backend.require_ready()
    stager = self._get_gpu_weight_stager(device)
    hidden = self._stage_token_embedding_tensor_cuda(
        store, stager, token_id_tensor, device=device
    )
    layers = list(getattr(store.bindings, "layers", []))
    for layer_offset, layer in enumerate(layers):
        if layer.layer_index < 2:
            hidden = deepseek_v4_flash_sliding_layer_forward_batched(
                hidden,
                layer=layer,
                stager=stager,
                backend=self.gpu_backend,
                states=states,
                token_indices=token_indices,
                token_id_tensors=token_id_tensor,
            )
        else:
            hidden = deepseek_v4_flash_compressed_layer_forward_batched(
                hidden,
                layer=layer,
                stager=stager,
                backend=self.gpu_backend,
                states=states,
                token_indices=token_indices,
                token_id_tensors=token_id_tensor,
            )
    streams = self._kernel_output_streams(hidden)
    next_tokens = self._output_token_argmax_chunked_cuda(
        store, stager=stager, streams=streams, device=device
    )
    if advance_state:
        for state in states:
            state.advance_token()
    return next_tokens
```

- [ ] **Step 5: Implement `generate_greedy_kernel_batched`**

```python
def generate_greedy_kernel_batched(
    self,
    input_ids_list: list[torch.Tensor],
    *,
    max_tokens: int = 1,
) -> list[torch.Tensor]:
    device = self._require_weight_store().device  # or derive from config
    states = []
    outputs = []
    batch = len(input_ids_list)
    for input_ids in input_ids_list:
        state = DeepSeekV4FlashGPURequestState(
            DeepSeekV4FlashGPUCacheConfig(
                context_length=self._kernel_context_length(),
                hidden_size=self.shape.hidden_size,
                batch_size=1,
                kv_width=self.shape.head_dim,
                device=device,
            )
        )
        states.append(state)
        output = input_ids.to(device=device, dtype=torch.long)
        outputs.append(output)
        # run prefix tokens one-by-one for this request
        for token_id in output[:-1].detach().cpu().tolist():
            self._forward_kernel_token_step_token_id(
                token_id=int(token_id),
                state=state,
                token_idx=state.token_position,
                device=device,
            )

    current_token_ids = torch.stack([
        outputs[i][states[i].token_position] for i in range(batch)
    ])
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    eos_token_id = self._eos_token_id()
    for _ in range(max_tokens):
        token_indices = [state.token_position for state in states]
        next_tokens = self._forward_kernel_token_step_batched(
            token_id_tensor=current_token_ids,
            states=states,
            token_indices=token_indices,
            device=device,
        )
        for i in range(batch):
            if finished[i]:
                continue
            outputs[i] = torch.cat([outputs[i], next_tokens[i].reshape(1)])
            current_token_ids[i] = next_tokens[i]
            if eos_token_id is not None and int(next_tokens[i].item()) == eos_token_id:
                finished[i] = True
        if finished.all():
            break
    return outputs
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/deepseek_v4_flash/test_model_kernel_generate_batched.py -v
```

- [ ] **Step 7: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/model.py tests/deepseek_v4_flash/test_model_kernel_generate_batched.py
uv run ruff check vllm/model_executor/models/deepseek_v4_flash/model.py tests/deepseek_v4_flash/test_model_kernel_generate_batched.py
uv run ruff format vllm/model_executor/models/deepseek_v4_flash/model.py tests/deepseek_v4_flash/test_model_kernel_generate_batched.py
git commit -m "feat(deepseek): batched token embedding, output, and greedy generation"
```

---

## Task 5: Engine Integration and Smoke Test

**Files:**
- Modify: `vllm/engine/lite_engine.py`
- Modify: `tests/tools/deepseek_v4_flash_quality_smoke.py`

**Interfaces:**
- Consumes: `generate_greedy_kernel_batched`.
- Produces: `LiteEngine.generate_deepseek_v4_flash_greedy_batched(requests, max_tokens)`; smoke script reports aggregate `decode_tps` for the batch.

- [ ] **Step 1: Add engine batched entry point**

In `LiteEngine._install_deepseek_v4_flash_direct_runtime`, set `self.max_active_requests = 4` (or keep the existing value but allow batching when multiple requests are submitted). Add:

```python
def generate_deepseek_v4_flash_greedy_batched(
    self,
    requests: list[tuple[str, str, SamplingParams]],
    *,
    max_tokens: int,
) -> dict[str, torch.Tensor]:
    input_ids_list = [
        torch.tensor(
            self._tokenizer.encode(prompt, add_special_tokens=True),
            dtype=torch.long,
        )
        for _, prompt, _ in requests
    ]
    outputs = self.model.generate_greedy_kernel_batched(
        input_ids_list,
        max_tokens=max_tokens,
    )
    return {request_id: outputs[i] for i, (request_id, _, _) in enumerate(requests)}
```

- [ ] **Step 2: Add `--batch-size` to quality smoke**

Modify `tests/tools/deepseek_v4_flash_quality_smoke.py` to optionally duplicate the prompt `batch_size` times and call the batched engine method. Report per-slot and aggregate decode_tps.

- [ ] **Step 3: Run smoke with batch size 2**

```bash
uv run python tests/tools/deepseek_v4_flash_quality_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --batch-size 2 --max-tokens 8
```

- [ ] **Step 4: Commit**

```bash
git add vllm/engine/lite_engine.py tests/tools/deepseek_v4_flash_quality_smoke.py
uv run ruff check vllm/engine/lite_engine.py tests/tools/deepseek_v4_flash_quality_smoke.py
uv run ruff format vllm/engine/lite_engine.py tests/tools/deepseek_v4_flash_quality_smoke.py
git commit -m "feat(deepseek): engine batched entry point and smoke benchmark"
```

---

## Task 6: Regression and Final Verification

- [ ] **Step 1: Run targeted tests**

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_batched_projections.py \
  tests/deepseek_v4_flash/test_gpu_batched_layers.py \
  tests/deepseek_v4_flash/test_model_kernel_generate_batched.py -q
```

Expected: PASS.

- [ ] **Step 2: Run existing deepseek tests**

```bash
uv run pytest tests/deepseek_v4_flash/test_attention_reference.py \
  tests/deepseek_v4_flash/test_gpu_sliding_layer.py \
  tests/deepseek_v4_flash/test_gpu_moe_path.py \
  tests/deepseek_v4_flash/test_model_kernel_generate.py -q
```

Expected: PASS (no regressions).

- [ ] **Step 3: Run regression suites**

```bash
bash tests/run_regression_suite.sh
bash tests/run_deepseek_v4_flash_real_smoke.sh
```

Expected: PASS.

- [ ] **Step 4: Final commit**

```bash
git commit --allow-empty -m "test(deepseek): Milestone 4 regression verification complete"
```

---

## Self-Review

**Spec coverage:**
- Batched projections: Task 1.
- Batched sliding layer: Task 2.
- Batched compressed layer: Task 3.
- Batched model kernel and greedy generation: Task 4.
- Engine integration and benchmark: Task 5.
- Regression verification: Task 6.

**Placeholder scan:**
- No `TBD` or `TODO`.
- Code snippets show exact interfaces and representative implementations.
- Commands include expected outcomes.

**Type consistency:**
- `hidden` is `[B, H]` in all batched functions.
- `states` is `list[DeepSeekV4FlashGPURequestState]`.
- `token_indices` is `list[int]`.
- `token_id_tensors` is `[B]` or per-element slice.

**Execution handoff:**
Plan complete and saved to `docs/superpowers/plans/2026-06-28-deepseek-v4-flash-milestone4-batching.md`.

**Execution options:**
1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks.
2. **Inline Execution** — execute tasks in this session using `executing-plans` with checkpoints.

Which approach would you like to use?
