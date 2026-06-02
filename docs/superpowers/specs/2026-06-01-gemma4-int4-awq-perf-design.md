# Gemma4 INT4/AWQ Performance Optimization Design

**Date**: 2026-06-02
**Status**: approved
**Scope**: Gemma4 26B MoE + 31B dense, INT4/AWQ main path only
**Target hardware**: AMD Ryzen AI Max+ 395 (Radeon 8060S 40CU, 128GB LPDDR5x-8000, 256 GB/s theoretical BW)

## Goals

| Model       | Current decode | Phase 1 target | Phase 2 target |
|-------------|---------------|----------------|----------------|
| Gemma4 26B  | ~12 tok/s     | 14 tok/s       | 16 tok/s       |
| Gemma4 31B  | ~3.7 tok/s    | 6 tok/s        | 8+ tok/s       |

Gate: no regression on `run_regression_suite.sh` or `run_inference_correctness_regression.sh` (A-lite tier).

## Roofline

31B decode at INT4: theoretical ~13-16 tok/s. Current 3.7 tok/s ~= 67 GB/s effective read BW out of 256 GB/s theoretical. The gap is kernel/data-movement, not scheduling.

## Architecture (5 phases, chained branches)

```
main
  └── perf/awq-audit-instrumentation          # P0
       └── perf/31b-downproj-splitk-gemv      # P1a
            └── perf/31b-mlp-streaming-fusion  # P1c
                 └── perf/26b-moe-strategy      # P2a (+ prefill)
```

Each phase: implement → unit tests → regression gate → benchmark compare.

## Phase 0: AWQ Fast-Path Audit + QKV Fusion Wiring

### P0a: Audit instrumentation

**Files**: `tests/e2e_full_benchmark.py`, `vllm/model_executor/layers/quantization/tensor.py`, `vllm/model_executor/models/gemma4/moe.py`

1. Benchmark output: after `AsyncLLM.generate()`, call `get_awq_runtime_prefix_stats()` and include in JSON output. Mark prefixes with `dense_build`/`decision_dense`/`cache_miss` as fallback paths.

2. Per-layer first-fallback trace: in `tensor.py` `_dense_fallback()`, on the first fallback per prefix, log shape / dtype / decision path. Subsequent same-prefix fallbacks count silently.

3. MoE fallback logging: remove `profile_enabled` guard on fallback-reason tracking in `moe.py:561` so it always records into stats.

**Acceptance**: benchmark JSON shows no `dense_build` or `decision_dense` on 31B MLP prefixes; no `moe_int4_decode_fallback:*` on 26B decode.

### P0b: QKV fusion wiring (quick win)

**Files**: `vllm/model_executor/models/gemma4/attention.py`

Kernel `packed_int4_symmetric_fused_qkv_m1_safe` already exists at `vllm/kernels/triton/awq_fused_gemm.py:2579`. Wire it into `attention.py` `forward()` for decode (`seqlen==1`, prefill path excluded).

Before: 3 × GEMV kernel launch per layer.
After: 1 × fused QKV GEMV per layer.

## Phase 1a: 31B down_proj Split-K GEMV

### Problem

31B down_proj shape: `m=1, n=5376, k=21504`. Current `_packed_int4_symmetric_group32_gemv_m1` (line 871) does single-grid GEMV; large K dimension limits parallelism.

### Solution

New kernel `_packed_int4_symmetric_group32_gemv_m1_splitk`:

- Split K into `SPLIT_K` parts (4 or 8)
- Each program: 1 split × BLOCK_N rows, accumulate in fp32
- Reduce across splits via `tl.atomic_add` on output tensor
- RDNA 3.5 adaptation: BLOCK_N aligned to wave64 (64 or 128), group_size=32 scale reuse preserved

**File**: `vllm/kernels/triton/awq_fused_gemm.py` (new kernel + launcher)
**Dispatch**: `tensor.py` matmul detects `m=1, k>=16384` and routes to split-K.

**Target**: down_proj per-layer time -15-25%.

## Phase 1c: 31B MLP Streaming Fusion

### Problem

Current mlp.py flow:
1. Fused gate/up GEMV → write 21504-dim h to global memory (~42KB fp16)
2. Read h back for down_proj GEMV

This intermediate read/write is wasted bandwidth for M=1 decode.

### Solution

New kernel `packed_int4_symmetric_fused_gate_up_silu_down_m1`:

- Input: x (5376), gate/up qweight+scales, down qweight+scales
- Compute: gate/up GEMV → activations → silu(gate)*up → down_proj GEMV
- Intermediate h lives in registers only, never global memory

### Risk mitigation

- PyTorch eager reference: compare per-layer hidden_states, require max_diff < 1e-3
- Fallback plan: if numerics fail A-lite gate, fuse only gate/up→silu (no down_proj), keeping intermediate in registers but still writing to global before down_proj. Moderate risk, moderate reward.

**Files**: `vllm/kernels/triton/awq_fused_gemm.py` (kernel), `vllm/model_executor/models/gemma4/mlp.py` (wiring)

**Target**: MLP total time -20-35%.

## Phase 2a: 26B MoE INT4 Strategy Table

### Problem

Current kernel selection is a flat if/else chain in `moe.py:518-538`. It picks one global strategy regardless of token count or expert distribution.

### Solution

Shape-aware policy table:

| Condition | Strategy |
|-----------|----------|
| n_tokens=1, decode | `two_stage` |
| n_tokens=2-8, unique_experts ≤ 4 | `batched_chunked` |
| n_tokens=2-8, unique_experts > 4 | `batched_grouped_streaming` |
| n_tokens ≥ 17, prefill | `grouped_prefill` |
| other decode | `batched_chunked_pair` |

### Additional changes

1. **Default enable grouped prefill** for 26B: flip `gemma4.py:124` default from `False` to `True`.
2. **Expert cache constraint**: max 8 dense expert copies, LRU eviction, per-step reset.

**Files**: `vllm/model_executor/models/gemma4/moe.py`, `vllm/adapters/gemma4.py`

## Prefill optimization (distributed across phases)

- P0 audit: verify prefill uses SDPA (not degraded to batch matmul).
- P2a: 26B prefill defaults to grouped.
- Targets: 31B prompt=256 TPS ≥ 50; 26B prompt=256 TPS ≥ 80.

## Verification per phase

```bash
# Unit tests
uv run pytest tests/test_awq_gemm_m1_specialization.py -q
uv run pytest tests/test_gemma4_moe_int4_kernel.py -q
uv run pytest tests/test_gemma4_moe_perf_policy.py -q

# Regression
bash tests/run_regression_suite.sh

# Correctness (manual gate, strict tier skipped)
SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh

# Performance
uv run python tests/e2e_full_benchmark.py \
  --models gemma4_26b_a4b,gemma4_31b_q4 \
  --model-process-isolation
```

## Non-goals (explicitly out of scope)

- HIP graph / CUDA graph capture for static-shape decode (future work)
- KV cache rework (current FP8/INT4 KV + 16-token blocks are sufficient for 512 context)
- Scheduler tuning (current bottleneck is kernel, not scheduling)
- FP16 weight paths (31B must be INT4/AWQ for this hardware)
