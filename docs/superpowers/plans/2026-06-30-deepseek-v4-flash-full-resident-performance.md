# DeepSeek V4 Flash Full-Resident Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move DeepSeek V4 Flash Q2 inference from staging/streaming bring-up toward a DS4-style full-resident fast path.

**Architecture:** Keep the existing adapter-owned direct runtime, but add an opt-in full-resident mode that stages raw GGUF payloads once and then keeps decode on GPU-facing kernels. Phase 1 removes repeated payload staging, Phase 2 forces the fused selected-expert MoE path when shapes allow it, and Phase 3 adds focused profiling gates so future work optimizes the remaining hot kernels instead of cache plumbing.

**Tech Stack:** Python 3.12, PyTorch ROCm, Triton via `vllm/triton_utils`, existing DeepSeek V4 Flash GGUF loader and direct runtime.

---

### Task 1: Full-Resident Raw Payload Mode

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Test: `tests/deepseek_v4_flash/test_gpu_weight_staging.py`
- Test: `tests/deepseek_v4_flash/test_single_request_direct_tuning.py`

- [ ] **Step 1: Write failing tests**

Add tests proving a stager can enter full-resident mode, that grouped expert payloads become cached/pinned, and that the model honors an opt-in environment flag.

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_weight_staging.py::test_full_resident_grouped_payloads_are_pinned -q
uv run pytest tests/deepseek_v4_flash/test_single_request_direct_tuning.py::test_full_resident_env_removes_staging_budget_cap -q
```

Expected: both fail because full-resident mode does not exist yet.

- [ ] **Step 2: Implement minimal full-resident staging**

Add `enable_full_resident_mode()` and `full_resident_enabled` to `DeepSeekV4FlashGPUWeightStager`. In full-resident mode, grouped payload inserts ignore LRU eviction and are pinned. Do not decode dense matrices or add a new cache layer.

- [ ] **Step 3: Wire the env flag**

Add `FASTINFERENCE_DEEPSEEK_V4_FLASH_FULL_RESIDENT=1` handling in the DeepSeek model setup. When enabled, remove the staging budget cap for payloads and call the stager full-resident method. Keep the old staging budget path as the default.

- [ ] **Step 4: Verify Task 1**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_weight_staging.py tests/deepseek_v4_flash/test_single_request_direct_tuning.py -q
```

Expected: pass.

### Task 2: Fused Selected-Expert MoE as the Fast Path

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
- Test: `tests/deepseek_v4_flash/test_gpu_backend_contract.py`
- Test: `tests/deepseek_v4_flash/test_gpu_layer_forward.py`

- [ ] **Step 1: Write failing tests**

Add tests that selected expert execution uses `fused_quantized_selected_experts_gemm()` when all selected experts have compatible IQ2 gate/up and Q2 down payloads, and falls back only for incompatible payload types.

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_backend_contract.py::test_selected_experts_prefers_fused_path_for_q2_iq2_payloads -q
```

Expected: fail because the unfused loop is still reachable first in some paths.

- [ ] **Step 2: Implement minimal fast-path selection**

Keep one helper that checks payload compatibility. If compatible, call the fused backend once with the existing workspace. If not compatible, use the current loop. Do not add a new dispatcher abstraction.

- [ ] **Step 3: Verify Task 2**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_backend_contract.py tests/deepseek_v4_flash/test_gpu_layer_forward.py -q
```

Expected: pass.

### Task 3: Decode Hot-Path Profiling Gates

**Files:**
- Modify: `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`
- Modify: `tests/e2e_full_benchmark.py`
- Test: `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py`

- [ ] **Step 1: Write failing tests**

Add a smoke-tool test that reports `full_resident_enabled`, `fused_selected_expert_api_calls`, `streamed_bytes`, and `q2_iq2_reference_fallback_calls` in the compact metrics.

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_real_smoke_tool_cli.py -q
```

Expected: fail because the metrics are incomplete.

- [ ] **Step 2: Add metric gates**

Extend the existing JSON metrics only. The fast-path success condition is `full_resident_enabled=1`, `streamed_bytes=0` after warmup, and `q2_iq2_reference_fallback_calls=0`.

- [ ] **Step 3: Verify Task 3**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_real_smoke_tool_cli.py -q
bash tests/run_regression_suite.sh
```

Expected: pass.

### Task 4: Real-Model Measurement

**Files:**
- No code changes unless a bug is exposed.

- [ ] **Step 1: Run focused real smoke when the GGUF exists**

Run:

```bash
FASTINFERENCE_DEEPSEEK_V4_FLASH_FULL_RESIDENT=1 \
FASTINFERENCE_KV_TYPE=fp16 \
FASTINFERENCE_BLOCK_SIZE=32 \
uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --context-length 4096 \
  --max-tokens 8 \
  --repeat 1 \
  --min-steady-decode-tps 5
```

Expected: pass on a configured AI Max+ 395 with the target GGUF. If it fails, inspect the emitted JSON counters before changing code.
