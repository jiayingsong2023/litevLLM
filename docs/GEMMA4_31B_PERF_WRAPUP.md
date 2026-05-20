# Gemma4-31B Performance Optimization Wrap-up

## Scope

This document summarizes the completed optimization work for `gemma4_31b_q4`, what produced durable value, what did not, the recommended default configuration, and what direction remains if performance work is resumed later.

The optimization program explicitly prioritized:

- inference accuracy must not regress
- single-node, single-GPU execution first
- no speculative decoding
- no multi-GPU parallelism

## What Was Done

### 1. Profiling and measurement cleanup

We first fixed the benchmark methodology so later numbers were usable:

- fixed decode length benchmarking via `ignore_eos=True` and `min_tokens=max_new_tokens`
- separated `prefill_tps` and `decode_tps`
- added layer-level profiling for Gemma4-31B
- added matrix benchmarking across prompt/decode shapes
- added startup memory audit logging
- added CUDA tensor memory attribution by type and category

This established a usable measurement baseline and avoided false improvements caused by early EOS or mixed prefill/decode timing.

### 2. Accuracy recovery and gated rollout

Before continuing optimization, accuracy regressions were repaired and kept under regression gating. The fused-path work was then reintroduced incrementally:

- attention path first
- MLP fused path second
- spot checks first, then broader regression

This prevented the optimization work from drifting away from a correct inference baseline.

### 3. Fused GEMM / MLP kernel tuning

We tuned AWQ fused GEMM for Gemma4-31B real shapes, especially decode-hot small-`M` MLP shapes.

Durable changes include:

- shape-aware heuristic tuning for MLP down-proj decode cases
- regression coverage for fused GEMM heuristics and numerics

This produced shape-local gains, especially on `p384`-class workloads, but did not translate into a large global average gain.

### 4. Attention/local decode optimization

We profiled the decode attention path and specifically targeted the local decode hot path:

- local decode Triton on/off A/B runs
- local/global launch parameter sweeps
- layer-level before/after profiling

This produced the clearest durable end-to-end benefit in the whole project.

### 5. Scheduler and bucket routing experiments

We evaluated several runtime profiles:

- `baseline`
- `decode_bias`
- `catchup_prefill`

Then we tested:

- single-cutoff prompt bucket routing
- prompt-length auto cutoff search
- a two-dimensional bucket policy using `prompt_tokens x decode_tokens`

The routing experiments improved over the original bucket policy but still failed to produce a strong enough global mean gain to justify changing the default scheduler policy.

### 6. Memory audit and startup footprint analysis

We added startup audit logging and traced memory consumers.

Observed CUDA startup breakdown on the audited run:

- model params: about `17.97 GiB`
- params[`int32`]: about `13.64 GiB`
- params[`bfloat16`]: about `4.33 GiB`
- model buffers: about `8.75 GiB` (`bfloat16`)
- KV pool delta: about `0.24 GiB`
- total after KV: about `27.28 GiB`

This confirmed that the dominant steady GPU footprint is not the KV cache. It is mostly model parameters and large BF16 buffers.

## Current E2E Baseline

Latest benchmark command:

```bash
uv run python tests/e2e_full_benchmark.py --models gemma4_26b_a4b,gemma4_31b_q4 --model-process-isolation
```

Latest measurement date: `2026-05-19` for the current 26B and 31B single-model default-profile reports.

Runtime profile:

- benchmark recommended profile now installed by default through runtime/model adapter policy
- per-model process isolation enabled automatically for Gemma4 26B + 31B
- `BS=1`
- `prompt_tokens~384`
- `max_new_tokens=24`
- `FASTINFERENCE_KV_MAX_MODEL_LEN=512`
- Gemma4 accuracy guard forced runtime KV cache to `fp8`

Current report summary:

| Model key | Aggregate TPS | TTFT p50 | E2E p50 | Prefill TPS agg | Decode TPS agg |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `gemma4_26b_a4b` | `4.86` | `2582.0ms` | `4940.0ms` | `152.60` | `9.75` |
| `gemma4_31b_q4` | `1.42` | `9335.7ms` | `16873.4ms` | `42.20` | `3.05` |

Interpretation:

- The recommended 26B MoE default is now `FASTINFERENCE_GEMMA4_MOE_INT4_KERNEL_STRATEGY=two_stage`.
- The 26B MoE path now has materially better prefill and decode throughput than 31B under this default shape.
- 31B remains TTFT/prefill-bound for this conservative default benchmark.
- `batched_chunked` and `batched_grouped_streaming` remain opt-in 26B MoE experiments; the latest default-entry verification measured `9.75` decode TPS.
- The 2026-05-20 31B single-model run measured `3.05` decode TPS and should be treated as the current conservative baseline.
- These numbers supersede older README headline Gemma4 TPS claims that were measured under different or less clearly pinned profiles.

## Confirmed Gains

### 1. Local decode Triton path

This is the strongest confirmed end-to-end win.

Reference: archived phase-2 local decode on/off gate report from 2026-04-24.

On `p384`:

- `d32`: decode TPS `1.2174 -> 1.2817` (`+5.28%`)
- `d64`: decode TPS `1.2057 -> 1.2752` (`+5.76%`)

TTFT remained acceptable in the same gate:

- `d32`: slightly improved
- `d64`: small regression, still within acceptable range for that gate

This is the only change that clearly crossed a practical per-shape acceptance threshold.

### 2. MLP fused kernel tuning

Reference: archived phase-1 GPU bottleneck report from 2026-04-24.

On a tuned `p384,d64,c1` case:

- decode TPS `1.2810 -> 1.3471`

This is meaningful, but it remained shape-local and did not generalize well enough to justify a broad default change by itself.

### 3. Better scheduling understanding

The profiling and matrix work successfully established the real policy behavior:

- short prompt behavior is not explained by prompt length alone
- `p64/p128` can split by decode length
- `p256/p384` generally prefer `baseline`

This is valuable operational knowledge even though it did not produce a strong enough default bucket policy.

## What Did Not Pay Off

### 1. Original prompt-only bucket default

Reference: archived sprint-2 matrix leaderboard from 2026-04-24.

The original bucket strategy:

- short prompt profile: `decode_bias`
- long prompt profile: `baseline`

did not hold up on the representative matrix. It underperformed pure `baseline` on average and failed the gate.

### 2. Two-dimensional bucket policy as default

Reference: archived phase-3 2D bucket gate report from 2026-04-24.

The improved 2D routing was:

- `prompt <= 128 and decode <= 32 -> catchup_prefill`
- `prompt <= 128 and decode > 32 -> decode_bias`
- `prompt > 128 -> baseline`

This improved over the old bucket policy, but only slightly:

- baseline decode TPS mean: `1.2869`
- 2D auto bucket decode TPS mean: `1.2908`
- mean decode gain: about `+0.30%`

This is too small to justify changing the default runtime policy.

### 3. KV write micro-optimizations

We tested micro-optimizations around KV write reshaping and unnecessary contiguous handling. These did not produce a consistent improvement and were reverted.

### 4. Fused GEMM modulo/masking rewrite

We tested a lower-level indexing change in the fused GEMM kernels. Numerical tests passed, but end-to-end performance regressed noticeably. The change was reverted.

### 5. More launch-tuning on already-optimized local/global attention configs

Additional local/global launch parameter sweeps did not yield a durable improvement beyond the current effective local decode path.

## Root Cause Summary

The project did identify the main bottlenecks clearly.

Reference: archived phase-1 p384 root-cause report from 2026-04-24.

The dominant runtime is still:

- MLP
- self-attention

The behavior is more consistent with a memory-access / small-shape efficiency problem than a simple scheduler-only problem.

The important implication is:

- the remaining work is not "find one more knob"
- the remaining work is "deeper kernel/data-movement redesign"

That is much higher effort and much lower certainty than the work already completed.

## Recommended Default Configuration

For production-like default behavior now:

- keep correctness-proven fused path changes already merged
- keep local decode Triton enabled
- keep scheduler default conservative

Recommended policy:

- `FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON=1` (default-on path)
- `FASTINFERENCE_AWQ_DECODE_GEMV=1` and `FASTINFERENCE_AWQ_FUSED_GATE_UP=1` installed by the Gemma4 adapter
- `FASTINFERENCE_AWQ_GROUP32_GEMV_ALL=1` and `FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ=1` installed for dense Gemma4
- `FASTINFERENCE_GEMMA4_MOE_INT4_KERNEL_STRATEGY=two_stage` installed as the 26B MoE default
- `FASTINFERENCE_GPU_GREEDY_*` benchmark profile enabled by default, with `FASTINFERENCE_GPU_GREEDY_IGNORE_EOS` still opt-in
- scheduler profile default: `baseline`
- bucket routing: keep available as an experiment, not as the default policy
- keep fixed decode length benchmarking for regression work
- keep current accuracy regression gate in front of any future performance change

Operationally, for Gemma4-31B on the current codebase, the safest default is still:

- local decode optimization on
- baseline scheduler on

## Why We Are Stopping Here

At this point, the work has entered diminishing returns:

- the clear wins have already been captured
- remaining scheduler gains are small and unstable
- remaining kernel opportunities are narrower and more expensive
- the next improvements are likely to require deeper kernel redesign, not incremental tuning

Under the current constraints, another large improvement from the same line of work is unlikely.

## If Work Resumes Later

The next phase should not continue as more small scheduler tweaks. It should change level and target one of the following:

### 1. Deeper MLP/self-attention data-movement work

Priority direction:

- reduce memory traffic in decode-hot shapes
- improve packed weight / scale access locality
- rework small-`M` kernel structure more aggressively

This is the most justified next technical direction if staying on one GPU.

### 2. More explicit per-shape policy table

If operational complexity is acceptable:

- use a tiny per-shape routing table instead of prompt-only or simple 2D cutoffs
- only route a few known shapes
- keep everything else on `baseline`

This is lower-risk than broad bucket policy changes, but the upside is still limited.

### 3. Architectural route change

If the business target remains dramatically higher throughput, the current constraints must change. The realistic next options would be:

- speculative decoding
- multi-GPU execution
- more aggressive custom attention/kernel rewrite
- different quantization/runtime format decisions

Without one of those shifts, another large gain should not be expected.

## Final Recommendation

Freeze the current performance line with:

- accurate benchmarking
- accuracy-preserving fused/local decode improvements
- conservative default scheduling

Treat this phase as complete. If performance work restarts, start from a new phase charter centered on deeper kernel/data-movement redesign or architectural changes, not another round of small scheduling tweaks.
