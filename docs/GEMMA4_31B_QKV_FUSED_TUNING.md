# Gemma4-31B QKV Fused Decode Tuning

Date: 2026-06-01

Branch: `feat/gemma4-awq-audit-prefill`

## Scope

This document records the P1 QKV/QK fused decode tile tuning that follows the
P0 AWQ audit. The goal is to keep Gemma4Attention decode on the fused QKV/QK
path while making it faster than the previous separate GEMV baseline.

## Microbenchmark

Tool:

```bash
uv run python tests/tools/benchmark_awq_qkv_fused_m1.py \
  --warmup 16 \
  --repeat 256 \
  --block-n 128,256 \
  --block-groups 4,8 \
  --json-out /tmp/gemma31_qkv_sweep_confirm.json
```

Measured exact shapes:

| Shape | Hidden K | Q | K/V | Fused output |
|---|---:|---:|---:|---:|
| `local_qkv` | `5376` | `8192` | `4096/4096` | `16384` |
| `global_qk` | `5376` | `16384` | `2048/0` | `18432` |

Winning config for both shapes:

```text
BLOCK_N=128
BLOCK_GROUPS=8
```

High-repeat confirmation:

| Shape | Old default | Winner | Fused ms | Separate ms | Speedup vs separate |
|---|---|---|---:|---:|---:|
| `local_qkv` | `256/4` | `128/8` | `0.331` | `1.545` | `4.67x` |
| `global_qk` | `256/4` | `128/8` | `0.440` | `1.955` | `4.44x` |

The broader sweep also showed that `BLOCK_N=192` and `BLOCK_N=384` trigger
kernel exceptions for this kernel, so they are not used in exact-shape defaults.

## End-to-End Gate

Profile command:

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
  --json-out /tmp/gemma31_qkv_tuned_rerun.json
```

Result:

| Run | Decode mean | Decode p50 | QKV separate | QK separate | Dense builds |
|---|---:|---:|---:|---:|---:|
| P0 pre-QKV fused reference | `~363ms` | unavailable | nonzero | nonzero | `0` |
| QKV fused before tuning | `~386ms` | `~386ms` | `0` | `0` | `0` |
| QKV fused after tuning | `349.825ms` | `348.603ms` | `0` | `0` | `0` |

Audit counters after tuning:

```text
qkv_fused_decode = 1150
qk_fused_decode  = 230
qkv_separate_decode = 0
qk_separate_decode  = 0
awq_dense_builds = 0
cache_bytes = 0
```

## Decision

Hardcode the two Gemma4-31B exact decode shapes to `BLOCK_N=128,
BLOCK_GROUPS=8`. Keep the tool override path intact so future ROCm/Triton
versions can re-sweep without changing code.
