# Gemma4-31B P1 AWQ Kernel Tuning Wrap-up

Date: 2026-06-01

Branch: `feat/gemma4-awq-audit-prefill`

## Scope

P1 follows the P0 audit by tuning decode-hot AWQ kernels without introducing an
untested large fused MLP kernel. The accepted changes must keep these audit
counters clean:

```text
qkv_separate_decode = 0
qk_separate_decode = 0
audit:dense_fallback = 0
awq_dense_builds = 0
cache_bytes = 0
```

## Completed Items

### 1. QKV/QK fused decode tuning

Tool:

```bash
uv run python tests/tools/benchmark_awq_qkv_fused_m1.py \
  --warmup 16 \
  --repeat 256 \
  --block-n 128,256 \
  --block-groups 4,8 \
  --json-out /tmp/gemma31_qkv_sweep_confirm.json
```

Winner for both Gemma4-31B exact decode shapes:

```text
BLOCK_N=128
BLOCK_GROUPS=8
```

End-to-end gate:

| State | Decode mean |
|---|---:|
| QKV fused before tuning | `~386ms` |
| QKV fused after tuning | `349.825ms` |
| QKV fused after rerun | `349.814ms` |

### 2. MLP `down_proj` decode GEMV tuning

Tool:

```bash
uv run python tests/tools/benchmark_awq_down_proj_m1.py \
  --warmup 8 \
  --repeat 64 \
  --json-out /tmp/gemma31_down_proj_sweep.json
```

Best microbenchmark candidate:

```text
path=group32
BLOCK_N=128
BLOCK_GROUPS=8
```

This exact shape is:

```text
m=1, n=5376, k=21504, group_size=32
```

End-to-end gate after routing this exact shape through group32 GEMV:

```text
decode_step_mean_ms = 349.814
decode_step_p50_ms  = 349.692
```

The first post-change profile was noisy (`376.320ms` decode mean), but the
immediate rerun returned to the QKV-tuned baseline. The route is kept because it
does not create a stable regression and keeps the decode path on the specialized
group32 kernel.

## MLP Streaming Fusion Decision

Full `gate/up -> silu(gate)*up -> down_proj` streaming fusion is not landed in
P1.

Reason:

- The current safe fused path already fuses `gate_proj` and `up_proj` with the
  activation for M=1 decode.
- Naively adding `down_proj` to the same kernel would recompute gate/up values
  for each output tile, which is likely slower than the current two-stage path.
- A correct high-performance version needs a cooperative/persistent design that
  computes gate/up once, keeps or tiles the intermediate activation, and reduces
  through `down_proj` without global materialization.
- That design needs a separate numerical gate with layer hidden-state reference
  comparison, not only final logits.

P1 therefore stops at verified QKV/QK and `down_proj` GEMV tuning. Full MLP
streaming fusion is a P2 kernel project.

## Final P1 Gate

Final profile command:

```bash
uv run python tests/tools/profile_gemma4_layer_breakdown.py \
  --model models/gemma-4-31B-it-AWQ-4bit \
  --prompt-tokens 128 \
  --decode-steps 8 \
  --warmup-decode 2 \
  --max-new-tokens 24 \
  --max-num-batched-tokens 512 \
  --max-model-len 512 \
  --awq-prefix-limit 512 \
  --json-out /tmp/gemma31_downproj_tuned_rerun.json
```

Final audited counters:

```text
qkv_fused_decode = 1150
qk_fused_decode = 230
qkv_separate_decode = 0
qk_separate_decode = 0
awq_dense_builds = 0
cache_bytes = 0
```

## Verification

Commands run:

```bash
uv run pytest tests/test_awq_gemm_m1_specialization.py tests/test_awq_audit_summary.py -q
uv run ruff check ...
bash tests/run_regression_suite.sh
```

The full model correctness regression remains the final kernel/numerics gate
before merge.
