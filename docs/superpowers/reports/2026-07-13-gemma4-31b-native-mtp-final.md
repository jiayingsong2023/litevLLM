# Gemma4-31B Native MTP Final Report

## Verdict

**NO-GO.** Do not implement P2 engine integration and do not merge this experiment
to `main`.

P0 established that the official assistant can share the target's paged FP8 KV
correctly and fits with the target on the test GPU. P1 did not meet its required
combination of greedy bit-exactness and at least 1.20x target-only decode speed.

## What Worked

- Official 31B assistant loading, shared target KV, and no assistant KV cache.
- P0 same-process reference parity for the assistant's shared-KV attention.
- Target/assistant residency and the P0 single-token latency gate.
- Cached multi-step state machine: proposal, K+1 verification, acceptance,
  rollback, predictor-state replacement, and block ownership.
- P1 performance probes with K=6 showed potential throughput before the strict
  numerical gate was applied. A full K=6 run had aggregate speedup 1.576x and
  every context bucket exceeded 1.0x.

Those performance numbers are not a release result because `chat_zh_2048` was
not bit-exact.

## Why P1 Failed

Normal target decode is optimized for M=1. It takes decode-specialized AWQ GEMV
and projection paths. A K+1 verifier takes generic M>1 linear paths. These are
mathematically close but not numerically identical in BF16/FP8 execution.

The first full P1 run was stable across target-only repetitions but diverged on
`chat_zh_2048` at continuation token 32. The verifier therefore failed a real
numerical contract, not a stochastic ROCm comparison.

The diagnostic sequence isolated the issue:

| Comparison | Result |
| --- | --- |
| M=1 vs M=7 layer-0 Q/K/V before exact-row work | max errors 1.0 / 1.0 / 0.5 |
| Exact-row M=7 Q/K/V vs seven M=1 calls | bit-exact |
| M=1 vs M=7 layer-0 raw attention output | bit-exact |
| M=1 vs M=7 layer-0 o_proj output | max error 0.0625 |

Thus QKV is not the only M=1-specialized component. `o_proj`, MLP projections,
and possibly lm-head would also require exact-row implementations. The resulting
model-level verifier would need every numerically sensitive linear to preserve
the M=1 reduction path.

## Exact-Row Microbenchmark Gate

The proposed solution was an exact-row M-small QKV kernel: one M=7 launch with
one independent M=1-equivalent reduction per row. On real 31B layer-0 weights:

| Metric | Value |
| --- | ---: |
| Bit-exact versus seven M=1 calls | yes |
| M=7 single launch | 2.090 ms |
| Seven M=1 launches | 1.905 ms |
| Speedup | 0.912x |

The exact kernel was 8.8% slower. Expanding it to `o_proj`, MLP, and lm-head
cannot plausibly recover the P1 1.20x end-to-end requirement. Per the planned
microbenchmark gate, this ends the experiment.

## Lessons For Future Projects

1. Establish the target's M=1 versus verifier M>1 numerical contract before
   designing speculative state machinery.
2. A multi-token verifier can only be bit-exact when every sensitive target
   operator has an exact M-small path, not merely QKV and attention.
3. Kernel microbenchmarks must compare against the deployed M=1 fast path;
   comparison to generic GEMM or mathematical references is insufficient.
4. Treat a performance result as invalid until it also passes the full greedy
   token-ID matrix. Near-tied logits can surface only after long contexts and
   rollback sequences.
5. Native MTP is worthwhile only when the checkpoint/runtime was designed with
   a numerically compatible multi-token verifier path, or when a measured
   exact-row kernel family is already faster than repeated M=1 decode.

## Repository Decision

Keep the P0/P1 tools and results on `gemma4-e2b-e4b-support` as experimental
evidence. Do not merge these Native MTP changes to `main`; do not add P2 engine
integration. Future decode performance work should target the normal 31B decode
path independently.
