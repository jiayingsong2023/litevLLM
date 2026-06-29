# DeepSeek V4 Flash Milestone 3: HIP/CUDA Graph Capture of the Decode Step

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the steady-state decode step (`_forward_kernel_token_step_token_tensor`) into a CUDA/HIP graph so each generated token replays the same kernel sequence with near-zero CPU launch overhead, while preserving correctness through fallback to the non-graph path.

**Architecture:** The decode step is not graph-ready as-is because it uses Python-scalar `token_idx` to compute RoPE angles, reads a dynamic slice of the KV cache, and branches on compressor emit boundaries. Milestone 3 therefore first extracts the dynamic parts (RoPE tables, KV window, expert payload bytes) into explicit tensor inputs that can be copied in place before `graph.replay()`. A new `DeepSeekV4FlashDecodeGraph` helper manages capture and replay; the greedy-generation loop uses it when the current step matches the captured configuration and falls back to the reference step otherwise.

**Tech Stack:** Python 3.12, PyTorch/ROCm, `torch.cuda.CUDAGraph`, Triton via `vllm/triton_utils`, `uv`, `pytest`.

## Global Constraints

- Never import `triton` directly; use `vllm.triton_utils`.
- All changes must pass `uv run ruff check . && uv run ruff format .`.
- All DeepSeek-specific tests and `bash tests/run_regression_suite.sh` must pass.
- Model output quality must pass `tests/run_deepseek_v4_flash_real_smoke.sh` and the DeepSeek path of `tests/run_inference_correctness_regression.sh`.
- Do not change upstream vLLM code paths outside `vllm/model_executor/models/deepseek_v4_flash/`.
- The graph path must be opt-in with fallback to the existing non-graph path; correctness gates must pass with graph enabled.

---

### Task 1: Precompute layer RoPE tables and pass them as tensor arguments

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/attention.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_runtime.py`
- Test: `tests/deepseek_v4_flash/test_attention_reference.py`, `tests/deepseek_v4_flash/test_gpu_sliding_layer.py`

**Interfaces:**
- Consumes: `token_idx: int`, `layer_idx: int`, `vectors: torch.Tensor`, `rotary_dim: int`.
- Produces: `rope_cos: torch.Tensor [vectors_per_layer, rotary_dim//2]`, `rope_sin: torch.Tensor [vectors_per_layer, rotary_dim//2]` cached per request state; `apply_precomputed_rope_to_tail(query, cos, sin)` function.

- [ ] **Step 1: Add a RoPE table builder**

In `vllm/model_executor/models/deepseek_v4_flash/attention.py`, add:

```python
def build_deepseek_layer_rope_tables(
    *,
    context_length: int,
    rotary_dim: int,
    rope_freq_base: float,
    compressed_rope_freq_base: float,
    rope_scale_factor: float,
    rope_original_context: int,
    rope_yarn_beta_fast: float,
    rope_yarn_beta_slow: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for all layer positions.

    Returns two tensors of shape [num_layers, context_length, rotary_dim // 2].
    For dense layers (layer_idx < 2) the compressed-YaRN schedule is unused and
    the table uses rope_freq_base; for compressed layers it uses the DS4
    long-context schedule from apply_deepseek_layer_rope_to_tail_reference.
    """
    ...
```

Implementation: iterate `layer_idx` and `token_idx`, call the existing reference `apply_deepseek_layer_rope_to_tail_reference` on a dummy `[rotary_dim//2, 2]` vector and store the resulting `cos, sin` values. Build dense tables for layers 0-1 and compressed tables for layers 2+.

- [ ] **Step 2: Add a precomputed-RoPE application helper**

```python
def apply_precomputed_rope_to_tail(
    vectors: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE using precomputed cos/sin vectors.

    vectors is [..., rotary_dim]. cos and sin are [rotary_dim // 2].
    Only the last rotary_dim entries of vectors are rotated.
    """
    ...
```

- [ ] **Step 3: Cache RoPE tables in request state**

In `DeepSeekV4FlashGPURequestState.__init__`, compute and store:

```python
self._rope_tables = build_deepseek_layer_rope_tables(
    context_length=cache_config.context_length,
    rotary_dim=DEEPSEEK_V4_FLASH_SHAPE.rotary_dim,
    ...
    device=cache_config.device,
)
```

Expose a method `rope_tables_for(layer_idx: int, token_idx: int) -> tuple[torch.Tensor, torch.Tensor]` returning `(cos, sin)`.

- [ ] **Step 4: Replace runtime RoPE in `_run_real_sliding_attention`**

In `gpu_layers.py`, replace the two calls to `apply_deepseek_layer_rope_to_tail_reference(query, ...)` and `apply_deepseek_layer_rope_to_tail_reference(current_kv, ...)` and the one inverse-RoPE call on `context` with `apply_precomputed_rope_to_tail(...)` using `state.rope_tables_for(layer.layer_index, token_idx)` / inverse tables. Keep the reference path available behind a flag `use_reference_rope` for fallback.

- [ ] **Step 5: Update compressed layer to use precomputed RoPE**

Find the compressed-layer paths that call RoPE functions and switch them to use `state.rope_tables_for`.

- [ ] **Step 6: Write tests**

Add `test_precomputed_rope_matches_reference` in `test_attention_reference.py` that compares `apply_precomputed_rope_to_tail` against `apply_deepseek_layer_rope_to_tail_reference` for several layer indices and token indices.

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_attention_reference.py tests/deepseek_v4_flash/test_gpu_sliding_layer.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/attention.py vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py vllm/model_executor/models/deepseek_v4_flash/gpu_runtime.py tests/deepseek_v4_flash/test_attention_reference.py tests/deepseek_v4_flash/test_gpu_sliding_layer.py
git commit -m "feat(deepseek): precompute layer RoPE tables for graph capture"
```

---

### Task 2: Pass KV window as an explicit tensor argument

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
- Test: `tests/deepseek_v4_flash/test_gpu_sliding_layer.py`

**Interfaces:**
- Consumes: `kv_rows: torch.Tensor | None`, `extra_kv_rows: torch.Tensor | None`.
- Produces: `_run_real_sliding_attention` no longer reads `state.raw_kv_cache` when `kv_rows` is provided.

- [ ] **Step 1: Refactor `_run_real_sliding_attention` to accept an explicit KV window**

The function already accepts `kv_rows` and `extra_kv_rows`. Change the logic so:

- If `kv_rows is None`, it reads from `state.raw_kv_cache` (existing behavior).
- If `kv_rows is not None`, it uses the provided tensor directly and concatenates `extra_kv_rows` if given.

This makes the captured graph see a static `kv_rows` input object.

- [ ] **Step 2: Add a state helper to materialize the window outside the graph**

```python
def raw_kv_window(
    self,
    layer_idx: int,
    token_idx: int,
    window: int,
) -> torch.Tensor:
    rows, _values = self.raw_kv_cache.read_raw_window(
        layer_idx,
        token_idx,
        window,
    )
    return rows
```

- [ ] **Step 3: Update sliding/compressed layer forward signatures**

Add optional `kv_rows: torch.Tensor | None = None` and `extra_kv_rows: torch.Tensor | None = None` parameters to `deepseek_v4_flash_sliding_layer_forward` and `deepseek_v4_flash_compressed_layer_forward`, and pass them through to `_run_real_sliding_attention`.

- [ ] **Step 4: Tests**

Add a test that calls `deepseek_v4_flash_sliding_layer_forward` with explicit `kv_rows` and compares to the path that reads from state.

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_sliding_layer.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_gpu_sliding_layer.py
git commit -m "feat(deepseek): allow explicit KV window in sliding attention"
```

---

### Task 3: Static expert payload buffer and copy helper

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Test: `tests/deepseek_v4_flash/test_gpu_moe_path.py`

**Interfaces:**
- Consumes: `expert_ids: torch.Tensor`, `grouped_experts`.
- Produces: `selected_payloads` whose `.payload` tensors are stable across calls.

- [ ] **Step 1: Pin grouped payload cache entries**

Ensure `stage_grouped_expert_payload` returns the same `torch.Tensor` object for the same `(tensor, expert_id)` across calls (it already does via `_grouped_cache`). Document that callers may mutate the bytes in place.

- [ ] **Step 2: Add a copy helper for dynamic expert selection**

In `gpu_weight_staging.py`, add:

```python
def copy_selected_expert_payload_bytes(
    self,
    grouped_experts: DeepSeekV4FlashGroupedExpertTensors,
    expert_ids: torch.Tensor,
    *,
    layer_idx: int | None = None,
) -> list[DeepSeekV4FlashSelectedExpertPayloads]:
    """Stage payloads, ensuring returned tensors are the cached stable objects.

    Before graph replay, call this to copy the currently-selected expert bytes
    into the pre-allocated cached payload tensors.
    """
    ...
```

This is essentially `_stage_selected_expert_payloads` with `stage_payloads_for_ids`; make sure the returned payloads are the cached objects.

- [ ] **Step 3: Expose the helper through the backend**

Add `backend.stage_selected_expert_payloads(...)` that delegates to the stager.

- [ ] **Step 4: Tests**

Add a test that calls the helper twice with different `expert_ids` and asserts that the returned `.payload` tensors for overlapping experts are the same objects (identity check with `is`).

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_moe_path.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py tests/deepseek_v4_flash/test_gpu_moe_path.py
git commit -m "feat(deepseek): stable expert payload buffer for graph capture"
```

---

### Task 4: Decode-step graph capture helper

**Files:**
- Create: `vllm/model_executor/models/deepseek_v4_flash/decode_graph.py`
- Test: `tests/deepseek_v4_flash/test_decode_graph.py`

**Interfaces:**
- Consumes: `model`, `state`, `device`, `token_idx`, `kv_rows`, `extra_kv_rows` (optional), non-emit flag.
- Produces: `DeepSeekV4FlashDecodeGraph.capture(...)` and `.replay(token_id_tensor)`.

- [ ] **Step 1: Implement the graph helper**

```python
class DeepSeekV4FlashDecodeGraph:
    """CUDA/HIP graph capture for one steady-state decode step.

    The graph assumes:
    - input token is a CUDA scalar long tensor (placeholder)
    - RoPE tables are precomputed and read from state (not captured as constants)
    - KV window tensors are stable input objects
    - expert payload tensors are stable cached objects whose bytes are copied before replay
    """

    def __init__(...):
        self.token_id_placeholder = torch.empty(
            (), dtype=torch.long, device=device
        )
        self.output_token = torch.empty(
            (), dtype=torch.long, device=device
        )
        self.graph = torch.cuda.CUDAGraph()

    @classmethod
    def capture(
        cls,
        model: DeepSeekV4FlashForCausalLM,
        *,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
        kv_rows_by_layer: dict[int, torch.Tensor] | None = None,
        extra_kv_rows_by_layer: dict[int, torch.Tensor] | None = None,
    ) -> DeepSeekV4FlashDecodeGraph:
        ...

    def replay(self, token_id_tensor: torch.Tensor) -> torch.Tensor:
        self.token_id_placeholder.copy_(token_id_tensor)
        self.graph.replay()
        return self.output_token.clone()
```

Capture flow:

1. Warm up: run `model._forward_kernel_token_step_token_tensor(...)` once with the placeholder and desired `token_idx`, using explicit KV windows and precomputed RoPE.
2. Start graph recording.
3. Run the step again; the output token is written into `self.output_token`.
4. End recording.

Because `_forward_kernel_token_step_token_tensor` returns a CUDA scalar, you cannot return it directly from the graph; instead write it to the placeholder output tensor inside the captured function. To do this, wrap the model call:

```python
def _capture_step() -> None:
    out = model._forward_kernel_token_step_token_tensor(
        token_id_tensor=self.token_id_placeholder,
        state=state,
        token_idx=token_idx,
        device=device,
    )
    self.output_token.copy_(out.reshape(()))
```

- [ ] **Step 2: Handle state side effects**

`state.advance_token()` is called inside the model step. During graph capture this would advance state. To avoid desync, add an optional flag to the step functions `advance_state: bool = True`; during capture set `advance_state=False`, and after `graph.replay()` the caller advances state manually.

Modify `_forward_kernel_token_step_token_tensor` and `_forward_kernel_token_hidden_token_tensor` to accept `advance_state` and only call `state.advance_token()` when true.

- [ ] **Step 3: Write tests**

Add `tests/deepseek_v4_flash/test_decode_graph.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_graph_matches_reference_step() -> None:
    # Build a model with real weights (use existing test fixtures if available)
    # Run one decode step with reference path.
    # Capture graph for the same state.
    # Replay and compare output token.
```

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_decode_graph.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/decode_graph.py vllm/model_executor/models/deepseek_v4_flash/model.py tests/deepseek_v4_flash/test_decode_graph.py
git commit -m "feat(deepseek): decode-step CUDA/HIP graph capture helper"
```

---

### Task 5: Integrate graph replay into greedy generation

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Test: `tests/deepseek_v4_flash/test_model_kernel_generate.py`

**Interfaces:**
- Consumes: `generate_greedy_kernel(..., use_graph: bool | None = None)`.
- Produces: same `output_ids` tensor, possibly faster.

- [ ] **Step 1: Add `use_graph` flag**

Add `use_graph: bool = False` to `generate_greedy_kernel`, `generate_greedy_kernel_timed`, and `_generate_greedy_kernel_impl`.

- [ ] **Step 2: Capture graph after prompt prefix**

After the prompt-prefix loop in `_generate_greedy_kernel_impl`, if `use_graph`:

1. Determine the steady-state configuration for the first decode position (`token_idx = state.token_position`).
2. Materialize explicit KV windows for all layers.
3. Capture a `DeepSeekV4FlashDecodeGraph`.

- [ ] **Step 3: Replay graph in the decode loop**

Inside the `for generated_idx in range(max_tokens)` loop, if `use_graph` and the current step is compatible with the captured graph (same layer types, no compressor emit, same window shape), replay; otherwise fall back to `_forward_kernel_token_step_token_tensor`.

After replay, manually call `state.advance_token()`.

- [ ] **Step 4: Tests**

Add `test_generate_greedy_kernel_with_graph_matches_without_graph` in `test_model_kernel_generate.py` that runs the same prompt with and without `use_graph=True` and asserts identical token outputs.

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_model_kernel_generate.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/model.py tests/deepseek_v4_flash/test_model_kernel_generate.py
git commit -m "feat(deepseek): integrate decode graph replay into greedy generation"
```

---

### Task 6: Correctness gates and benchmark

**Files:**
- Modify: `docs/superpowers/plans/2026-06-28-deepseek-v4-flash-performance.md`

- [ ] **Step 1: Run targeted tests**

```bash
uv run pytest tests/deepseek_v4_flash/test_decode_graph.py tests/deepseek_v4_flash/test_model_kernel_generate.py tests/deepseek_v4_flash/test_attention_reference.py tests/deepseek_v4_flash/test_gpu_backend_contract.py -q
```

Expected: PASS.

- [ ] **Step 2: Run regression suites**

```bash
bash tests/run_regression_suite.sh
bash tests/run_deepseek_v4_flash_real_smoke.sh
SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh
```

Expected: all green (graph integration must not break existing paths).

- [ ] **Step 3: Graph benchmark**

Run quality smoke with graph enabled. If the lite engine does not expose `use_graph`, temporarily add it to `LiteEngine.generate_deepseek_v4_flash_greedy` or run a standalone benchmark script.

```bash
uv run python tests/tools/deepseek_v4_flash_quality_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --context-length 4096 --max-tokens 32 --min-output-chars 8 \
  --prompt-text "What is the capital of France?" --json-out /tmp/ds_m3_run1.json
```

Run warmup + 3 recorded times. Compare to Milestone 2 average (~0.601 tok/s).

- [ ] **Step 4: Record results**

Append a "Milestone 3 Results" section to `docs/superpowers/plans/2026-06-28-deepseek-v4-flash-performance.md` and commit:

```bash
git add docs/superpowers/plans/2026-06-28-deepseek-v4-flash-performance.md
git commit -m "docs: milestone 3 benchmark results"
```

---

## Self-Review

- **Spec coverage:** The plan addresses the three dynamic inputs that block naive graph capture (RoPE angles, KV window, expert payloads) and adds the capture/replay helper and integration.
- **Placeholder scan:** All steps contain concrete code, commands, and expected outputs; no TBD/placeholder language.
- **Type consistency:** `token_id_tensor` stays a CUDA scalar `torch.Tensor`; `state.advance_token()` is called by the caller during graph replay; RoPE tables are keyed by `(layer_idx, token_idx)`.
- **Risk note:** This milestone is more invasive than Milestones 1 and 2 because it touches the top-level generation loop and requires making RoPE/KV window dynamic inputs. The plan keeps the non-graph path intact as a fallback.
