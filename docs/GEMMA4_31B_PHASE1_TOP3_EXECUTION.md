# Gemma4-31B Phase 1 Top3 Execution Checklist

This checklist is the Phase 1 execution baseline for current Gemma4-31B bottlenecks.
Scope excludes speculative decoding and multi-card parallelism.

## Baseline Bottleneck Share (from layer profile)

| Rank | Bottleneck | Share |
|---|---|---:|
| 1 | `layer_dense_mlp` | 51.29% |
| 2 | `layer_self_attn` | 41.80% |
| 3 | `attn_local_decode` | 3.30% |

## Phase 1 Task List (Task -> Expected Gain -> Gate)

| Task ID | Task | Code Position | Expected Gain (E2E) | Risk | Gate Threshold |
|---|---|---|---|---|---|
| P1-T1 | MLP fused GEMM decode bucket tuning (`n=5376,k=21504,m<=2`) | `vllm/kernels/triton/awq_fused_gemm.py` (`_select_fused_gemm_blocks`) | +3% to +8% decode TPS in C1 hot shapes | Kernel config overfits one shape and regresses others | `p256/p384 x d32/d64` decode TPS >= +5% vs baseline, quality unchanged |
| P1-T2 | Self-attention decode tile refinement (local/global split buckets) | `vllm/kernels/triton/paged_attention.py`, scheduler decode path | +4% to +10% decode TPS | Layer mix mismatch; local/global config interference | Same matrix: no cell worse than -2%, mean >= +5% |
| P1-T3 | Persistent small-M autotune/profile reuse (remove first-run jitter) | `vllm/kernels/triton/awq_fused_profile.json`, profile loader | +2% to +6% steady-state, better p95 | Stale profile causing hardware/version mismatch | Cold vs warm run delta < 8%; no accuracy drift |

## Validation Protocol

1. Accuracy gate first:
   - `bash tests/run_inference_correctness_regression.sh`
   - Must pass before throughput acceptance.
2. Throughput gate (fixed decode length):
   - enforce `ignore_eos=True`, `min_tokens=max_new_tokens`
   - report `prefill_tps`, `decode_tps`, `ttft_ms` separately
3. Required matrix:
   - prompt: `256, 384`
   - decode: `32, 64`
   - concurrency: `1` (C1 primary), optional `2`
4. Acceptance:
   - quality: no new failures
   - performance: C1 decode TPS mean >= +5%
   - stability: p95 latency regression <= 3%

## Current Status

- P1-T1: in progress (MLP decode bucket tuning landed in heuristic path).
- P1-T2: pending.
- P1-T3: pending (profile materialization not yet promoted to default rollout).
