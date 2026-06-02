# Gemma4-31B AWQ P0 Runtime Audit

Date: 2026-06-01

Branch: `feat/gemma4-awq-audit-prefill`

## Scope

P0 answers which runtime path is responsible for the current Gemma4-31B AWQ
decode and prefill slowdown before deeper kernel work starts.

The audit specifically tracks:

- `qkv_projection_paths`
- `first_fallbacks`
- `audit:dense_fallback`
- `audit:fused_runtime_fallback`
- per-prefix AWQ matmul counters

## Profile Command

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
  --json-out /tmp/gemma31_p0_audit.json
```

## Baseline Findings

Initial P0 audit showed two hidden regressions:

| Signal | Value | Meaning |
|---|---:|---|
| `audit:dense_fallback` | `60` | Prefill `mlp.down_proj` built dense weights |
| `awq_dense_builds` | `60` | Dense materialization happened once per layer |
| `cache_bytes` | `13,872,660,480` | Dense cache consumed about 12.9 GiB |
| `qk_separate_decode` | `230` | QK decode projections were not fused |
| `qkv_separate_decode` | `1150` | QKV decode projections were not fused |

The first fallback detail identified the shape:

```text
prefix=*.mlp.down_proj
m=141, n=5376, k=21504, group_size=32
reason=dense_fallback
```

## P0 Fix Results

After expanding the Gemma4-31B `down_proj` fused AWQ path to prefill shapes:

| Signal | Before | After |
|---|---:|---:|
| `audit:dense_fallback` | `60` | `0` |
| `awq_dense_builds` | `60` | `0` |
| `cache_bytes` | `13,872,660,480` | `0` |
| `mlp_down_proj` fused attempts | `1380` | `1440` |
| `mlp_down_proj` fused success rate | `1.0` | `1.0` |

After adding Gemma4Attention decode QKV/QK fused-path dispatch:

| Signal | Before | After |
|---|---:|---:|
| `qk_separate_decode` | `230` | `0` |
| `qkv_separate_decode` | `1150` | `0` |
| `qk_fused_decode` | `0` | `230` |
| `qkv_fused_decode` | `0` | `1150` |

## Performance Note

The QKV/QK fused path is functionally connected and removes the separate decode
counters, but the measured decode step regressed on this profile:

```text
prefill down_proj fixed, QKV separate: decode_step_mean_ms ~= 363
prefill down_proj fixed, QKV fused:    decode_step_mean_ms ~= 386
```

This means P0 proved that QKV fusion was wired correctly, but the initial
fused QKV launch config was not yet a performance win. The follow-up P1 tuning
recorded in `docs/GEMMA4_31B_QKV_FUSED_TUNING.md` hardcoded the exact-shape
`BLOCK_N=128, BLOCK_GROUPS=8` config and improved the same profile to about
`349.825ms` decode mean with separate QKV/QK counters still at zero.

## P0 Completion Criteria

- Runtime audit counters are available through `get_awq_runtime_audit_summary()`.
- Profile output includes the AWQ audit summary.
- Prefill `down_proj` dense fallback is eliminated for the audited 31B shape.
- Decode QKV/QK separate projection counters are eliminated.
- First fallback detail is retained once per fallback class and later repeats
  are counted.
- Targeted unit tests cover audit summary, prefill `down_proj`, and QKV/QK fused
  helper behavior.

## Out Of Scope For P0

- Tuning the QKV fused kernel so it beats separate exact GEMV.
- 31B `down_proj` split-K/GEMV tile tuning.
- MLP streaming fusion.
- 26B MoE expert-cache policy.
- Full prefill SDPA and grouped-prefill optimization.
