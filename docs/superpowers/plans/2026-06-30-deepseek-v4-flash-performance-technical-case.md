# DeepSeek V4 Flash Performance Technical Case Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve DeepSeek V4 Flash Q2 decode throughput without changing model math, router top-k, quantization, or output quality.

**Architecture:** Do not pursue graph/capture again. The remaining credible work is to measure the current decode hot path, remove per-layer/per-token staging overhead that does not change semantics, then tune only the Q2/IQ2 selected-expert Triton kernels that dominate the measured profile.

**Tech Stack:** Python 3.12, `uv`, PyTorch, ROCm, Triton via `vllm.triton_utils`, existing DeepSeek V4 Flash lite runtime.

---

## Decision Summary

Graph/capture is out of scope. It has already been attempted several times and should not consume more engineering time.

Full expert GPU tables are out of scope. They were semantically equivalent but had worse cold start, higher VRAM use, and no useful throughput gain.

Router top-k reduction is out of scope. It changes model behavior and can reduce answer quality.

The remaining non-quality-sacrificing candidates are:

1. Measure real decode time with profiler counters plus ROCm kernel timing.
2. Remove `torch.stack()` allocation/copy churn in selected expert kernels by reusing request-local payload workspaces.
3. Reduce Python/GPU synchronization around selected expert staging where measurable.
4. Tune the existing selected-expert Q2/IQ2 Triton kernels in place.

Expected outcome is incremental, not magical: first target is a verified 20-50% improvement over the current direct GGUF decode path. A 5 tok/s target should be treated as aspirational until profiling proves kernel time, not memory bandwidth or ROCm launch overhead, has enough headroom.

## Current Evidence

Current selected routed expert flow:

- `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py::_run_staged_routed_experts`
- `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py::stage_grouped_expert_payloads_for_ids`
- `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py::fused_quantized_selected_experts_gemm`
- `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py::deepseek_v4_iq2_xxs_selected_experts_activation`
- `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py::deepseek_v4_q2_k_selected_experts_down_projection`

Known costs in the current path:

- `stage_grouped_expert_payloads_for_ids()` converts `expert_ids` to CPU with `.to(device="cpu").tolist()`. That is a synchronization point.
- `deepseek_v4_iq2_xxs_selected_experts_activation()` stacks gate/up payloads with `torch.stack(...).contiguous()` every call.
- `deepseek_v4_q2_k_selected_experts_down_projection()` stacks down payloads with `torch.stack(...).contiguous()` every call.
- The fused selected path launches at least one IQ2 gate/up activation kernel and one Q2 down projection kernel per routed layer.

## Files

- Modify: `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`
  Add compact decode hot-path metrics and optional ROCm profiler command hints.
- Modify: `vllm/model_executor/models/deepseek_v4_flash/profiler.py`
  Only if existing summaries cannot report per-section totals sorted by elapsed time.
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_runtime.py`
  Add request-local reusable uint8 payload workspaces.
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
  Pass request-local payload workspaces into the fused selected-expert backend.
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
  Accept optional preallocated selected payload stacks and avoid `torch.stack()` allocation inside the hot path.
- Modify: `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`
  Add workspace-aware selected-expert wrapper functions and tune existing kernels only after measurements justify it.
- Test: `tests/deepseek_v4_flash/test_gpu_backend_contract.py`
- Test: `tests/deepseek_v4_flash/test_gpu_layer_forward.py`
- Test: `tests/deepseek_v4_flash/test_gpu_weight_staging.py`
- Test: `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py`

## Phase 0: Measurement Gate

Do this before optimizing. If this does not show selected-expert staging or Q2/IQ2 kernels dominating decode, stop and write a new plan for the measured bottleneck.

### Task 1: Add A Decode Hot-Path Summary

**Files:**
- Modify: `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`
- Test: `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py`

- [ ] **Step 1: Write the failing CLI summary test**

Add an assertion to the existing smoke CLI test that the JSON summary includes these integer fields:

```python
assert "decode_profile_top_sections" in result
assert isinstance(result["decode_profile_top_sections"], list)
assert "q2_k_triton_calls" in result
assert "iq2_xxs_gate_up_fused_calls" in result
assert "fused_selected_expert_api_calls" in result
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_real_smoke_tool_cli.py -q
```

Expected: FAIL because `decode_profile_top_sections` is missing.

- [ ] **Step 3: Implement the smallest summary**

In `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`, reuse the existing model profiler and backend stats. Add only a compact sorted list:

```python
profile = profiler.compact_summary() if profiler is not None else {}
sections = profile.get("sections", {})
top_sections = sorted(
    (
        {"name": name, "ms": float(value.get("elapsed_ms", 0.0))}
        for name, value in sections.items()
    ),
    key=lambda item: item["ms"],
    reverse=True,
)[:12]
result["decode_profile_top_sections"] = top_sections
```

- [ ] **Step 4: Run the test and verify it passes**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_real_smoke_tool_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Run the real baseline**

Run:

```bash
uv run python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-new-tokens 32 \
  --profile \
  --json
```

Expected: JSON includes decode TPS, backend counters, cache stats, and top sections.

### Task 2: Capture ROCm Kernel Timing For One Decode

**Files:**
- Create: `docs/superpowers/plans/2026-06-30-deepseek-v4-flash-performance-profile.md`

- [ ] **Step 1: Run a ROCm profile around the same smoke command**

Run:

```bash
rocprofv3 --stats --output-format csv --output-file /tmp/deepseek_v4_flash_decode \
  uv run python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
    --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
    --max-new-tokens 32 \
    --profile \
    --json
```

Expected: `/tmp/deepseek_v4_flash_decode*.csv` exists and lists Triton kernel durations.

- [ ] **Step 2: Write the measured bottleneck note**

Create `docs/superpowers/plans/2026-06-30-deepseek-v4-flash-performance-profile.md` with:

```markdown
# DeepSeek V4 Flash Decode Profile

## Baseline

- Command: `rocprofv3 --stats --output-format csv --output-file /tmp/deepseek_v4_flash_decode uv run python tests/tools/run_deepseek_v4_flash_gpu_smoke.py --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --max-new-tokens 32 --profile --json`
- Decode TPS: record the `decode_tps` value printed by the command.
- TTFT: record the `ttft_ms` value printed by the command.
- Backend counters: paste the `gpu_backend` counters object from the JSON output.
- Top Python profiler sections: paste the first 12 `decode_profile_top_sections` entries.
- Top ROCm kernels: paste the 12 highest-duration Triton kernels from `/tmp/deepseek_v4_flash_decode*.csv`.

## Decision

- Continue with Phase 1 if selected expert staging or Q2/IQ2 selected kernels dominate.
- Stop and re-plan if another section dominates.
```

- [ ] **Step 3: Commit**

```bash
git add tests/tools/run_deepseek_v4_flash_gpu_smoke.py tests/deepseek_v4_flash/test_real_smoke_tool_cli.py docs/superpowers/plans/2026-06-30-deepseek-v4-flash-performance-profile.md
git commit -m "test: add deepseek v4 flash decode profile summary"
```

## Phase 1: Remove Selected-Expert Stack Allocation Churn

This keeps exact expert IDs, exact weights, exact Q2/IQ2 payload bytes, and exact kernels. It only replaces repeated allocation with reusable request-local buffers.

### Task 3: Add Request-Local Payload Stack Workspaces

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_runtime.py`
- Test: `tests/deepseek_v4_flash/test_gpu_layer_forward.py`

- [ ] **Step 1: Write the failing request-state test**

Add a test that asks for the same uint8 workspace twice and verifies tensor identity:

```python
def test_request_state_reuses_moe_payload_workspace() -> None:
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=8,
            hidden_size=4,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    )

    first = state.moe_payload_workspace(
        name="gate",
        num_experts=6,
        payload_bytes=1024,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    second = state.moe_payload_workspace(
        name="gate",
        num_experts=6,
        payload_bytes=1024,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    assert first.data_ptr() == second.data_ptr()
    assert first.shape == (6, 1024)
    assert first.dtype == torch.uint8
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_layer_forward.py::test_request_state_reuses_moe_payload_workspace -q
```

Expected: FAIL because `moe_payload_workspace` does not exist.

- [ ] **Step 3: Add the minimal workspace cache**

In `DeepSeekV4FlashGPURequestState.__init__` add:

```python
self._moe_payload_workspace: dict[
    tuple[str, int, int, torch.device],
    torch.Tensor,
] = {}
```

Add:

```python
def moe_payload_workspace(
    self,
    *,
    name: str,
    num_experts: int,
    payload_bytes: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = self.config.device
    if device is None:
        device = torch.device("cuda")
    key = (name, num_experts, payload_bytes, device)
    cached = self._moe_payload_workspace.get(key)
    if cached is not None:
        return cached
    workspace = torch.empty(
        (num_experts, payload_bytes),
        dtype=torch.uint8,
        device=device,
    )
    self._moe_payload_workspace[key] = workspace
    return workspace
```

- [ ] **Step 4: Run the test and verify it passes**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_layer_forward.py::test_request_state_reuses_moe_payload_workspace -q
```

Expected: PASS.

### Task 4: Pass Payload Workspaces To The Fused Selected Backend

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Test: `tests/deepseek_v4_flash/test_gpu_layer_forward.py`

- [ ] **Step 1: Extend the fake backend test**

In `test_staged_routed_experts_prefers_fused_selected_path_without_state`, add a second test with `state` present and assert the backend receives three uint8 stack buffers:

```python
assert backend.payload_stack_shapes == {
    "gate": (2, 1),
    "up": (2, 1),
    "down": (2, 1),
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_layer_forward.py::test_staged_routed_experts_prefers_fused_selected_path_with_payload_workspaces -q
```

Expected: FAIL because payload stack buffers are not passed.

- [ ] **Step 3: Add optional backend parameters**

In `DeepSeekV4FlashGPUBackend.fused_quantized_selected_experts_gemm`, add optional keyword-only parameters:

```python
gate_stack: torch.Tensor | None = None
up_stack: torch.Tensor | None = None
down_stack: torch.Tensor | None = None
```

In `_run_staged_routed_experts`, when `state is not None`, allocate:

```python
gate_bytes = int(selected_payloads[0][1].payload.numel())
up_bytes = int(selected_payloads[0][2].payload.numel())
down_bytes = int(selected_payloads[0][3].payload.numel())
gate_stack = state.moe_payload_workspace(
    name="gate",
    num_experts=len(selected_payloads),
    payload_bytes=gate_bytes,
    device=hidden.device,
)
up_stack = state.moe_payload_workspace(
    name="up",
    num_experts=len(selected_payloads),
    payload_bytes=up_bytes,
    device=hidden.device,
)
down_stack = state.moe_payload_workspace(
    name="down",
    num_experts=len(selected_payloads),
    payload_bytes=down_bytes,
    device=hidden.device,
)
```

Pass those buffers to `selected_gemm(...)`.

- [ ] **Step 4: Run the test and verify it passes**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_layer_forward.py::test_staged_routed_experts_prefers_fused_selected_path_with_payload_workspaces -q
```

Expected: PASS.

### Task 5: Replace `torch.stack()` With Workspace Copies

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Modify: `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`
- Test: `tests/deepseek_v4_flash/test_gpu_backend_contract.py`

- [ ] **Step 1: Write a correctness test for workspace-backed selected GEMM**

Duplicate the existing fused selected GEMM correctness test and call the backend with preallocated stacks:

```python
gate_stack = torch.empty((6, len(payloads[0][1].payload)), dtype=torch.uint8, device="cuda")
up_stack = torch.empty((6, len(payloads[0][2].payload)), dtype=torch.uint8, device="cuda")
down_stack = torch.empty((6, len(payloads[0][3].payload)), dtype=torch.uint8, device="cuda")
actual = backend.fused_quantized_selected_experts_gemm(
    hidden=hidden,
    expert_weights=expert_weights,
    payloads=payloads,
    workspace=torch.empty((6, rows), dtype=torch.float32, device="cuda"),
    gate_stack=gate_stack,
    up_stack=up_stack,
    down_stack=down_stack,
)
torch.testing.assert_close(actual, expected, rtol=5.0e-3, atol=1.0e-4)
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_backend_contract.py::test_fused_quantized_selected_experts_gemm_accepts_payload_workspaces -q
```

Expected: FAIL because optional stack parameters are not wired into the kernels.

- [ ] **Step 3: Implement workspace copy in backend**

In `fused_quantized_selected_experts_gemm`, copy each selected payload into the matching row:

```python
for payload_index, (_expert_id, gate, up, down) in enumerate(payloads):
    gate_stack[payload_index].copy_(gate.payload.reshape(-1), non_blocking=True)
    up_stack[payload_index].copy_(up.payload.reshape(-1), non_blocking=True)
    down_stack[payload_index].copy_(down.payload.reshape(-1), non_blocking=True)
```

Then call new q2/iq2 helpers that accept already-stacked tensors. Keep the current list-based helpers as fallback wrappers.

- [ ] **Step 4: Run correctness tests**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_backend_contract.py::test_fused_quantized_selected_experts_gemm_accepts_payload_workspaces tests/deepseek_v4_flash/test_gpu_backend_contract.py::test_fused_quantized_selected_experts_gemm_matches_loop -q
```

Expected: PASS.

- [ ] **Step 5: Benchmark before continuing**

Run the same smoke benchmark from Phase 0.

Expected: continue only if decode TPS or profiler sections improve. If no measurable improvement, revert Task 3-5 and stop.

## Phase 2: Kernel Tuning Only If Profile Justifies It

Do not tune blindly. ROCm Triton may be part of the ceiling, but the proof has to be kernel timing, not suspicion.

### Task 6: Tune Selected IQ2 Gate/Up Rows Per Program

**Files:**
- Modify: `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`
- Test: `tests/deepseek_v4_flash/test_gpu_backend_contract.py`

- [ ] **Step 1: Add a local constant for selected activation rows per block**

Replace the inline `rows_per_block = 4` in `deepseek_v4_iq2_xxs_selected_experts_activation` with:

```python
_SELECTED_IQ2_ROWS_PER_BLOCK = 4
```

- [ ] **Step 2: Benchmark 2, 4, and 8 manually**

For each value, run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_backend_contract.py::test_fused_quantized_selected_experts_gemm_matches_loop -q
uv run python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-new-tokens 32 \
  --profile \
  --json
```

Expected: keep the fastest value only if correctness passes and decode TPS improves.

- [ ] **Step 3: Document register pressure if value increases**

If the chosen value raises Triton register pressure above roughly 64 registers per thread, add a short ASCII comment above the kernel launch:

```python
# ROCm note: ROWS_PER_BLOCK=8 improves decode on AI Max+395 but increases
# register pressure; keep 4 if occupancy regresses on larger GPUs.
```

### Task 7: Tune Selected Q2 Down Projection Rows Per Program

**Files:**
- Modify: `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`
- Test: `tests/deepseek_v4_flash/test_gpu_backend_contract.py`

- [ ] **Step 1: Add a local constant for selected down rows per block**

Replace the inline `rows_per_block = 4` in `deepseek_v4_q2_k_selected_experts_down_projection` with:

```python
_SELECTED_Q2_ROWS_PER_BLOCK = 4
```

- [ ] **Step 2: Benchmark 2, 4, and 8 manually**

Use the same commands as Task 6.

Expected: keep only the fastest correct value.

## Phase 3: CPU Synchronization Reduction

This is lower priority than stack churn because selected payload staging still needs Python-visible expert IDs unless experts are fully resident. Do not build a complicated async prefetcher without profile proof.

### Task 8: Count Expert-ID CPU Syncs

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`
- Test: `tests/deepseek_v4_flash/test_gpu_weight_staging.py`

- [ ] **Step 1: Add a counter test**

In a staging test that calls `stage_grouped_expert_payloads_for_ids()` with CUDA expert IDs, assert:

```python
assert stager.cache_stats()["expert_id_cpu_syncs"] == 1
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_weight_staging.py::test_stage_grouped_expert_payloads_counts_expert_id_cpu_sync -q
```

Expected: FAIL because the counter is missing.

- [ ] **Step 3: Add the counter**

Initialize:

```python
"expert_id_cpu_syncs": 0,
```

Increment before `flattened.to(device="cpu", dtype=torch.int64).tolist()` when `flattened.is_cuda`.

- [ ] **Step 4: Run the test**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_weight_staging.py::test_stage_grouped_expert_payloads_counts_expert_id_cpu_sync -q
```

Expected: PASS.

- [ ] **Step 5: Decide**

If the counter is high but profile time is low, stop. If profile time is high, write a separate plan for request-local predicted/prefetched selected experts. Do not mix it into this plan.

## Final Verification

Run:

```bash
uv run ruff check vllm/model_executor/models/deepseek_v4_flash tests/deepseek_v4_flash tests/tools/run_deepseek_v4_flash_gpu_smoke.py
uv run pytest tests/deepseek_v4_flash/test_gpu_layer_forward.py tests/deepseek_v4_flash/test_gpu_backend_contract.py tests/deepseek_v4_flash/test_gpu_weight_staging.py tests/deepseek_v4_flash/test_real_smoke_tool_cli.py -q
bash tests/run_regression_suite.sh
```

Expected:

- Ruff passes.
- Targeted tests pass.
- Regression suite passes.
- Smoke benchmark shows non-regressed correctness and improved decode TPS.

## Stop Conditions

Stop immediately if:

- A proposed change alters router top-k, expert weights, payload bytes, quantization, or logits.
- The benchmark improvement is within noise after three runs.
- ROCm profiling shows time dominated by launch overhead or memory bandwidth with no local code churn left to remove.
- A change requires resurrecting graph/capture or full expert tables.

## Self-Review

Spec coverage:

- Graph/capture is explicitly abandoned.
- Full expert tables stay removed.
- Quality-sacrificing top-k reduction is excluded.
- Remaining work is measurable, incremental, and reversible.

Placeholder scan:

- No task contains unresolved placeholder instructions.

Type consistency:

- Request state owns reusable tensors.
- GPU layers allocate request-local workspaces.
- Backend accepts optional payload stack tensors.
- Kernels remain the only place that launch Triton selected-expert work.
