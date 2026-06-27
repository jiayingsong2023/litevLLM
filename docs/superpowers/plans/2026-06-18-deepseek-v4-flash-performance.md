# DeepSeek V4 Flash Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve DeepSeek V4 Flash real GGUF batch=1 greedy decode performance while preserving readable, coherent GPU output.

**Architecture:** Keep `LiteEngine` and the REST/control plane model-agnostic. Optimizations stay inside the DeepSeek V4 Flash model adapter, GPU staging layer, and Triton kernels. Every stage must preserve the current quality smoke baseline: `What is the capital of France?` generates `The capital of France is Paris.` or another clearly correct readable answer.

**Tech Stack:** Python 3.12 via `uv`, PyTorch ROCm tensors, Triton kernels imported through `vllm/triton_utils/`, GGUF weight staging, pytest.

---

## Current Baseline

Current validated real GGUF smoke:

```bash
timeout --foreground --kill-after=60s 1200s uv run --no-sync python tests/tools/deepseek_v4_flash_quality_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-tokens 8 --min-output-chars 8 --top-k 8 \
  --json-out /tmp/ds4_quality_perf_baseline.json \
  --dump-step-json /tmp/ds4_quality_perf_baseline_steps.json
```

Expected quality:

```text
readability.passed = true
readability.text = "The capital of France is Paris."
gpu_backend.q2_iq2_reference_fallback_calls = 0
gpu_backend.iq2_xxs_gate_up_fused_calls > 0
gpu_backend.q2_k_triton_calls > 0
```

Known likely bottlenecks from the current code:

- `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py::_run_staged_routed_experts` loops over selected experts in Python, staging and launching one expert at a time.
- `gpu_layers.py::_materialize_selected_expert_ids` copies selected expert ids from GPU to CPU each routed layer.
- `gpu_backend.py::quantized_expert_gemm` calls Q8 roundtrip and separate IQ2 gate/up plus Q2 down kernels per expert.
- `gpu_weight_staging.py` has prefetch support, but request scheduling and cache-hit validation need stronger real-run instrumentation.
- `tests/tools/deepseek_v4_flash_quality_smoke.py` records quality and backend counters, but not enough wall-clock/per-token/per-section performance data for regression gating.

## File Structure

Modify these files:

- `tests/tools/deepseek_v4_flash_quality_smoke.py`: add timing, warmup, repeat, and profiler JSON output.
- `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py`: unit tests for new quality smoke output schema and regression thresholds.
- `vllm/model_executor/models/deepseek_v4_flash/profiler.py`: add aggregate helpers without changing existing event schema.
- `tests/deepseek_v4_flash/test_profiler.py`: tests for profiler aggregation.
- `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`: remove per-layer CPU expert-id materialization, batch selected expert payload staging, reduce Python hot-loop overhead.
- `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`: expose expert payload bundle/prefetch stats and avoid repeated cache bookkeeping.
- `tests/deepseek_v4_flash/test_gpu_layer_forward.py`: tests for zero CPU materialization path and batched payload staging.
- `tests/deepseek_v4_flash/test_expert_prefetch.py`: tests for next-layer prefetch and cache-hit behavior.
- `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`: add fused routed-expert API boundaries and counters.
- `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`: add grouped selected-expert fused kernel entry point.
- `tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py`: correctness tests for grouped fused routed expert path.
- `tests/deepseek_v4_flash/test_deepseek_q2_iq2_kernel_profile.py`: profile test for grouped fused kernel throughput.
- `vllm/model_executor/models/deepseek_v4_flash/model.py`: schedule prefetch earlier and report performance metrics through smoke path.

Do not modify these unless a task explicitly requires it:

- `vllm/engine/*`: performance work must remain model-local.
- REST API files: only validate after model-local performance improves.
- Generic vLLM compatibility layers.

---

### Task 1: Add A Stable Performance Baseline To Quality Smoke

**Files:**
- Modify: `tests/tools/deepseek_v4_flash_quality_smoke.py`
- Modify: `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/profiler.py`
- Modify: `tests/deepseek_v4_flash/test_profiler.py`

- [x] **Step 1: Write profiler aggregation tests**

Add tests to `tests/deepseek_v4_flash/test_profiler.py`:

```python
def test_profiler_snapshot_includes_aggregate_by_name() -> None:
    profiler = DeepSeekV4FlashProfiler(enabled=True)

    with profiler.section("router_expert_stage", layer_idx=1):
        pass
    with profiler.section("router_expert_stage", layer_idx=1):
        pass
    with profiler.section("router_expert_kernel", layer_idx=1):
        pass

    data = profiler.snapshot()
    aggregate = data["aggregate_by_name"]

    assert aggregate["router_expert_stage"]["count"] == 2
    assert aggregate["router_expert_stage"]["total_ms"] >= 0.0
    assert aggregate["router_expert_stage"]["avg_ms"] >= 0.0
    assert aggregate["router_expert_kernel"]["count"] == 1
```

- [x] **Step 2: Implement profiler aggregation**

Update `DeepSeekV4FlashProfiler.snapshot()` in `vllm/model_executor/models/deepseek_v4_flash/profiler.py` so it returns:

```python
def _aggregate_by_name(self) -> dict[str, dict[str, float | int]]:
    aggregate: dict[str, dict[str, float | int]] = {}
    for event in self._events:
        current = aggregate.setdefault(
            event.name,
            {"count": 0, "total_ms": 0.0, "avg_ms": 0.0, "max_ms": 0.0},
        )
        current["count"] = int(current["count"]) + 1
        current["total_ms"] = float(current["total_ms"]) + event.elapsed_ms
        current["max_ms"] = max(float(current["max_ms"]), event.elapsed_ms)
        current["avg_ms"] = float(current["total_ms"]) / int(current["count"])
    return aggregate
```

and include `"aggregate_by_name": self._aggregate_by_name()` in `snapshot()`.

- [x] **Step 3: Add quality smoke timing CLI contract tests**

Add a CLI schema test in `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py` using the existing fake model/tool helpers. The expected payload must include:

```python
assert payload["performance"]["generated_tokens"] == len(payload["generated_token_ids"])
assert payload["performance"]["total_elapsed_ms"] >= 0.0
assert payload["performance"]["decode_tokens_per_second"] >= 0.0
assert "gpu_backend" in payload
```

- [x] **Step 4: Implement timing fields in quality smoke**

In `tests/tools/deepseek_v4_flash_quality_smoke.py`, wrap generation with `time.perf_counter()` and add:

```python
"performance": {
    "generated_tokens": len(generated_ids),
    "total_elapsed_ms": total_elapsed_ms,
    "decode_tokens_per_second": (
        len(generated_ids) / (total_elapsed_ms / 1000.0)
        if total_elapsed_ms > 0.0
        else 0.0
    ),
}
```

If `DeepSeekV4FlashForCausalLM` exposes a profiler snapshot, include it under `"profiler"`.

- [x] **Step 5: Verify**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_profiler.py tests/deepseek_v4_flash/test_real_smoke_tool_cli.py -q
uv run --no-sync ruff check tests/tools/deepseek_v4_flash_quality_smoke.py vllm/model_executor/models/deepseek_v4_flash/profiler.py tests/deepseek_v4_flash/test_profiler.py tests/deepseek_v4_flash/test_real_smoke_tool_cli.py
```

Expected: all tests pass and ruff reports no issues.

---

### Task 2: Establish Real GGUF Performance Gate

**Files:**
- Modify: `tests/tools/deepseek_v4_flash_quality_smoke.py`
- Modify: `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py`

- [x] **Step 1: Add CLI arguments for performance gating**

Add arguments:

```python
parser.add_argument("--warmup-tokens", type=int, default=0)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--min-decode-tps", type=float, default=0.0)
parser.add_argument("--max-total-elapsed-ms", type=float, default=0.0)
```

- [x] **Step 2: Implement repeat summary**

For `--repeat N`, run the same prompt N times after optional warmup. Emit:

```python
"performance_summary": {
    "repeat": args.repeat,
    "decode_tps_values": [...],
    "decode_tps_min": min(values),
    "decode_tps_median": statistics.median(values),
    "decode_tps_max": max(values),
}
```

The existing single-run fields must remain for backward compatibility.

- [x] **Step 3: Fail on explicit performance gates only**

The tool must return non-zero if:

```python
args.min_decode_tps > 0.0 and decode_tps_min < args.min_decode_tps
args.max_total_elapsed_ms > 0.0 and total_elapsed_ms > args.max_total_elapsed_ms
```

Quality failures must still return non-zero.

- [x] **Step 4: Verify real model baseline**

Run:

```bash
timeout --foreground --kill-after=60s 1200s uv run --no-sync python tests/tools/deepseek_v4_flash_quality_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-tokens 8 --warmup-tokens 1 --repeat 3 --top-k 8 \
  --json-out /tmp/ds4_perf_gate_baseline.json
```

Expected:

```text
readability.passed = true
performance_summary.decode_tps_min > 0
gpu_backend.q2_iq2_reference_fallback_calls = 0
```

Do not set a hard TPS threshold yet; this task records the current baseline.

---

### Task 3: Remove GPU-to-CPU Expert ID Materialization From The Hot Path

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`
- Modify: `tests/deepseek_v4_flash/test_gpu_layer_forward.py`

- [x] **Step 1: Write failing test for no CPU materialization when payloads are cached**

Add a test in `tests/deepseek_v4_flash/test_gpu_layer_forward.py` that:

1. Creates CUDA `expert_ids`.
2. Uses a fake stager with `stage_grouped_expert_payloads_for_ids(expert_ids=...)`.
3. Asserts `_run_staged_routed_experts` does not increment `routed_expert_id_materializations`.

Expected assertion:

```python
assert stager.cache_stats().get("routed_expert_id_materializations", 0) == 0
```

- [x] **Step 2: Add batched payload staging interface**

In `DeepSeekV4FlashGPUWeightStager`, add:

```python
def stage_grouped_expert_payloads_for_ids(
    self,
    tensors: DeepSeekV4FlashGroupedExpertTensors,
    expert_ids: torch.Tensor,
    *,
    layer_idx: int | None = None,
) -> list[tuple[int, DeepSeekV4FlashStagedQuantizedExpertPayload, DeepSeekV4FlashStagedQuantizedExpertPayload, DeepSeekV4FlashStagedQuantizedExpertPayload]]:
```

Implementation may materialize ids on CPU inside the stager initially, but it centralizes the sync and allows later replacement with a GPU gather/cache path. Record a counter named `"batched_payload_stage_calls"`.

- [x] **Step 3: Use batched staging from `_run_staged_routed_experts`**

In `gpu_layers.py::_run_staged_routed_experts`, prefer:

```python
stage_payloads_for_ids = getattr(stager, "stage_grouped_expert_payloads_for_ids", None)
if callable(stage_payloads_for_ids):
    payloads = stage_payloads_for_ids(grouped_experts, expert_ids, layer_idx=layer_idx)
    for index, (expert_id, gate_payload, up_payload, down_payload) in enumerate(payloads):
        expert_weight = expert_weights.reshape(-1)[index]
        ...
```

Keep the existing path as fallback for old fake stagers.

- [x] **Step 4: Verify**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_layer_forward.py tests/deepseek_v4_flash/test_gpu_weight_staging.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py tests/deepseek_v4_flash/test_gpu_layer_forward.py
```

Expected: existing quality behavior unchanged, new counter visible in stager stats.

---

### Task 4: Make Expert Prefetch Measurable And Earlier

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`
- Modify: `tests/deepseek_v4_flash/test_expert_prefetch.py`
- Modify: `tests/deepseek_v4_flash/test_model_kernel_generate.py`

- [x] **Step 1: Write tests for next-layer prefetch ordering**

In `tests/deepseek_v4_flash/test_model_kernel_generate.py`, add a fake stager that records prefetch requests. Verify that after layer `L` router ids are known, the model schedules layer `L+1` prefetch before entering layer `L+1` expert execution.

Expected recorded request:

```python
assert fake_stager.prefetch_requests[0].layer_idx == next_layer_idx
assert fake_stager.prefetch_requests[0].expert_ids
```

- [x] **Step 2: Add prefetch hit/miss counters per layer**

In `gpu_weight_staging.py`, extend `_record_prefetch_delta()` output counters:

```python
"prefetch_payload_hits"
"prefetch_payload_misses"
"prefetch_payload_streamed_bytes"
```

Keep existing counters intact.

- [x] **Step 3: Schedule prefetch as soon as selected expert ids are known**

In `model.py`, move or add `_schedule_next_layer_expert_prefetch(...)` immediately after router/indexer selection, before output projection and residual work for the current layer where possible. Use the existing `torch.cuda.Stream` path if available.

- [x] **Step 4: Verify real prefetch effect**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_expert_prefetch.py tests/deepseek_v4_flash/test_model_kernel_generate.py -q
timeout --foreground --kill-after=60s 1200s uv run --no-sync python tests/tools/deepseek_v4_flash_quality_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-tokens 8 --warmup-tokens 1 --repeat 3 --top-k 8 \
  --json-out /tmp/ds4_perf_prefetch.json
```

Expected:

```text
readability.passed = true
prefetch counters are present
decode_tps_min is not lower than baseline by more than 5%
```

---

### Task 5: Fuse Selected Expert Execution At Backend API Boundary

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
- Modify: `tests/deepseek_v4_flash/test_gpu_backend_contract.py`
- Modify: `tests/deepseek_v4_flash/test_gpu_layer_forward.py`

- [x] **Step 1: Add backend contract test**

In `tests/deepseek_v4_flash/test_gpu_backend_contract.py`, add a test for:

```python
backend.quantized_selected_experts_gemm(
    hidden=hidden,
    expert_weights=expert_weights,
    payloads=[(expert_id, gate_payload, up_payload, down_payload), ...],
)
```

Expected result must equal the current Python loop:

```python
expected = sum(
    weight * backend.quantized_expert_gemm(
        hidden=hidden,
        gate_payload=gate,
        up_payload=up,
        down_payload=down,
    ).to(torch.float32)
    for weight, (_, gate, up, down) in zip(expert_weights, payloads, strict=True)
)
torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
```

- [x] **Step 2: Implement backend method with existing kernels**

Add `DeepSeekV4FlashGPUBackend.quantized_selected_experts_gemm(...)`. First implementation may still loop internally, but it must:

- allocate the output once;
- keep accumulation on GPU;
- increment `"selected_expert_fused_api_calls"`;
- avoid re-entering `_run_staged_routed_experts` Python accumulation.

- [x] **Step 3: Use backend selected-expert API from `gpu_layers.py`**

In `_run_staged_routed_experts`, after payload staging, call:

```python
selected_gemm = getattr(backend, "quantized_selected_experts_gemm", None)
if callable(selected_gemm):
    return selected_gemm(
        hidden=hidden,
        expert_weights=expert_weights.reshape(-1),
        payloads=payloads,
    ).to(torch.float32)
```

Fallback to the old loop for fake backends.

- [x] **Step 4: Verify quality and performance**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_backend_contract.py tests/deepseek_v4_flash/test_gpu_layer_forward.py -q
timeout --foreground --kill-after=60s 1200s uv run --no-sync python tests/tools/deepseek_v4_flash_quality_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-tokens 8 --warmup-tokens 1 --repeat 3 --top-k 8 \
  --json-out /tmp/ds4_perf_selected_api.json
```

Expected:

```text
readability.passed = true
gpu_backend.selected_expert_fused_api_calls > 0
gpu_backend.q2_iq2_reference_fallback_calls = 0
```

---

### Task 6: Add A True Grouped Selected-Expert Triton Kernel

**Files:**
- Modify: `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Modify: `tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py`
- Modify: `tests/deepseek_v4_flash/test_deepseek_q2_iq2_kernel_profile.py`

- [x] **Step 1: Write grouped kernel correctness test**

Add a test using two or more selected experts with small synthetic payloads. The grouped kernel must match:

```python
expected = sum(
    expert_weights[i] * backend.quantized_expert_gemm(
        hidden=hidden,
        gate_payload=gate,
        up_payload=up,
        down_payload=down,
    )
    for i, (_, gate, up, down) in enumerate(payloads)
)
```

Use relaxed quantized tolerances:

```python
torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
```

- [x] **Step 2: Add kernel input dataclass and wrapper**

Historical note: this grouped selected-expert wrapper was later removed because
local A/B runs did not show a stable performance win over the simpler
selected-expert API loop.

Every new Triton kernel must include ASCII comments describing:

- GGUF IQ2_XXS gate/up row layout;
- GGUF Q2_K down row layout;
- selected expert dimension;
- output accumulation layout.

- [x] **Step 3: Use grouped kernel from backend**

In `gpu_backend.py::quantized_selected_experts_gemm`, detect all payloads are:

```python
gate/up = GGML_TYPE_IQ2_XXS
down = GGML_TYPE_Q2_K
same rows/columns across selected experts
```

Historical note: the backend now always uses the Task 5 API loop for this
path; the grouped-kernel capability and counter were deleted to reduce
maintenance cost.

- [ ] **Step 4: Profile kernel throughput**

Extend `tests/deepseek_v4_flash/test_deepseek_q2_iq2_kernel_profile.py` to print or assert GB/s for grouped selected-expert execution. The profile test should be skip-safe when CUDA/ROCm is unavailable.

- [x] **Step 5: Verify**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py tests/deepseek_v4_flash/test_deepseek_q2_iq2_kernel_profile.py -q
timeout --foreground --kill-after=60s 1200s uv run --no-sync python tests/tools/deepseek_v4_flash_quality_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-tokens 8 --warmup-tokens 1 --repeat 3 --top-k 8 \
  --json-out /tmp/ds4_perf_grouped_kernel.json
```

Expected:

```text
readability.passed = true
q2_iq2_reference_fallback_calls == 0
decode_tps_min does not regress from the current wrapper baseline
```

---

### Task 7: Reduce Shared Expert And Output Projection Overheads

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Modify: `vllm/kernels/triton/deepseek_v4_flash/q8_linear.py`
- Modify: `tests/deepseek_v4_flash/test_gpu_layer_forward.py`
- Modify: `tests/deepseek_v4_flash/test_triton_q8_linear.py`

- [x] **Step 1: Add tests for Q8 raw payload reuse**

Write tests asserting shared expert and output projection do not call `stage_matrix()` when Q8 raw payload path is available.

Expected:

```python
assert fake_stager.stage_matrix_calls == 0
assert fake_stager.stage_q8_raw_payload_calls > 0
```

- [ ] **Step 2: Add fused Q8 gate/up activation path for shared expert**

For shared experts where gate/up/down are Q8-compatible, add a backend API:

```python
q8_shared_expert_gemm(hidden, gate_tensor, up_tensor, down_tensor, stager)
```

It should use raw Q8 payload staging and avoid decoded dense F16/F32 matrices.

- [x] **Step 3: Add chunked output projection benchmark guard**

In `test_triton_q8_linear.py`, add a skip-safe GPU test that compares chunked output projection with the existing output path and records elapsed time.

- [x] **Step 4: Verify**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_layer_forward.py tests/deepseek_v4_flash/test_triton_q8_linear.py -q
timeout --foreground --kill-after=60s 1200s uv run --no-sync python tests/tools/deepseek_v4_flash_quality_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-tokens 8 --warmup-tokens 1 --repeat 3 --top-k 8 \
  --json-out /tmp/ds4_perf_q8_projection.json
```

Expected: readable output and no regression against Task 2 baseline.

---

### Task 8: Add Performance Regression Targets

**Files:**
- Modify: `tests/tools/deepseek_v4_flash_quality_smoke.py`
- Create: `tests/tools/fixtures/deepseek_v4_flash_perf_baseline.json`
- Modify: `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py`
- Modify: `tests/run_inference_correctness_regression.sh` only if runtime is acceptable.

- [ ] **Step 1: Save baseline fixture**

After Tasks 1-7, save the best stable local run:

```bash
cp /tmp/ds4_perf_grouped_kernel.json tests/tools/fixtures/deepseek_v4_flash_perf_baseline.json
```

The fixture must include:

```json
{
  "quality_text": "The capital of France is Paris.",
  "min_decode_tps": <measured_min_decode_tps * 0.75>,
  "required_backend_counters": {
    "q2_iq2_reference_fallback_calls": 0
  }
}
```

- [ ] **Step 2: Add fixture-driven gate option**

Add `--perf-baseline-json` to quality smoke. When present, validate required counters and minimum TPS with 25% tolerance.

- [ ] **Step 3: Decide regression-suite integration**

Do not add DeepSeek V4 Flash to the default correctness regression unless local runtime is acceptable. If it exceeds the current full regression budget, add a separate opt-in command:

```bash
RUN_DEEPSEEK_V4_FLASH=1 bash tests/run_inference_correctness_regression.sh
```

- [ ] **Step 4: Verify**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_real_smoke_tool_cli.py -q
timeout --foreground --kill-after=60s 1200s uv run --no-sync python tests/tools/deepseek_v4_flash_quality_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-tokens 8 --warmup-tokens 1 --repeat 3 --top-k 8 \
  --perf-baseline-json tests/tools/fixtures/deepseek_v4_flash_perf_baseline.json
```

Expected: command exits 0 and output remains readable.

---

## Final Verification

Run before claiming performance work complete:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash --ignore=tests/deepseek_v4_flash/test_direct_decode_real_smoke.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash vllm/kernels/triton/deepseek_v4_flash tests/deepseek_v4_flash tests/tools/deepseek_v4_flash_quality_smoke.py
timeout --foreground --kill-after=60s 1200s uv run --no-sync python tests/tools/deepseek_v4_flash_quality_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-tokens 8 --warmup-tokens 1 --repeat 3 --top-k 8 \
  --json-out /tmp/ds4_perf_final.json
```

Success criteria:

- Output is readable and coherent.
- `q2_iq2_reference_fallback_calls = 0`.
- Grouped selected expert fused path is hit after Task 6.
- Decode TPS improves over Task 2 baseline.
- No engine or REST API generic layer has DeepSeek-specific branching.

## Execution Recommendation

Use Subagent-Driven execution with one fresh subagent per task and review between tasks. Task 1 and Task 2 are mandatory before kernel work because they make performance regressions visible. Task 6 is the main expected speedup; Tasks 3-5 reduce Python and synchronization overhead so Task 6 can actually show up in end-to-end throughput.
