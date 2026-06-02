# Gemma4-31B P3a AWQ Interleaved Group32 Layout

Date: 2026-06-02

## Objective

Evaluate whether co-locating group32 AWQ qweight packs and scales improves M=1
GEMV global-read behavior on the AMD AI MAX / 8060S path.

Experimental layout:

```text
packed[n, group, 0:4] = four int32 qweight packs for 32 K values
packed[n, group, 4]   = fp16/bf16 scale bits stored in low 16 bits of int32
```

Raw storage per group changes from separate layout:

```text
separate:    4 * int32 + 1 * fp16 = 18 bytes/group
interleaved: 5 * int32            = 20 bytes/group
```

So this layout only wins if improved access coalescing beats the 11% raw-byte
padding overhead.

## Implementation

Experimental kernel/helper:

```text
pack_awq_group32_interleaved_qweight_scales
packed_int4_symmetric_group32_interleaved_gemv_m1_safe
```

Runtime hook:

```text
awq_group32_interleaved_down_proj(default=False)
```

The runtime path is policy-gated and limited to Gemma4-31B exact down_proj
decode:

```text
M=1, N=5376, K=21504, group_size=32, scales=float16/bfloat16
```

It caches one interleaved copy per `PackedInt4Weight` instance and records:

```text
awq_interleaved_builds
awq_interleaved_cache_bytes
interleaved_down_proj_success
interleaved_down_proj_fallback
```

Default behavior remains unchanged unless `awq_group32_interleaved_down_proj` is
set in the runtime kernel policy.

## Microbenchmark

Tool:

```bash
uv run python tests/tools/benchmark_awq_interleaved_group32_m1.py \
  --warmup 8 \
  --repeat 64 \
  --json-out /tmp/gemma31_awq_interleaved_group32_repeat64.json
```

31B down_proj result with fp16 scales:

```text
shape: M=1, N=5376, K=21504
current_group32     = 1.095891714 ms
interleaved_group32 = 0.798694193 ms
speedup             = 1.372x
```

31B down_proj result with bf16 scales, matching the observed model profile:

```text
shape: M=1, N=5376, K=21504
current_group32     = 1.122018218 ms
interleaved_group32 = 0.772554159 ms
speedup             = 1.452x
```

QKV/QK shape checks:

```text
local QKV-like:  M=1, N=16384, K=5376
current_group32     = 0.932257473 ms
interleaved_group32 = 2.475172281 ms
speedup             = 0.377x

global QK-like:  M=1, N=18432, K=5376
current_group32     = 1.005427599 ms
interleaved_group32 = 2.422262430 ms
speedup             = 0.415x
```

Conclusion: this layout is useful only for the deep-K/narrow-N down_proj shape.
It should not replace QKV/QK fused kernels.

## Memory Cost

For one 31B down_proj layer:

```text
N * (K / 32) * 5 * 4 bytes = 5376 * 672 * 20 = 72,253,440 bytes
```

Across the observed 60 down_proj layers in the local 31B profile this is:

```text
60 * 72,253,440 = 4,335,206,400 bytes (~4.04 GiB) additional cache
```

This is much smaller than the previous dense fallback cache failure mode, but it
is still large enough that the path must remain policy-gated until end-to-end
profile validates the tradeoff.

## End-to-End Gate

Short 31B profiles showed that the runtime branch could activate, but they were
not stable enough for a default-policy decision. The promotion gate is therefore
the serial long profile below.

Serial long profile, baseline:

```text
json: /tmp/gemma31_p3a_baseline_long_serial.json
decode_step_mean_ms       = 369.184
decode_step_p50_ms        = 370.498
awq_interleaved_builds    = 0
awq_dense_builds          = 0
mlp_down_proj fused calls = 2400 / 2400
```

Serial long profile, interleaved enabled:

```text
json: /tmp/gemma31_p3a_interleaved_long_serial.json
decode_step_mean_ms                = 402.169
decode_step_p50_ms                 = 403.087
awq_interleaved_builds             = 60
awq_interleaved_cache_bytes        = 4,335,206,400
audit:interleaved_down_proj_success = 2340
awq_dense_builds                   = 0
mlp_down_proj interleaved calls    = 2340 / 2400
```

Delta versus baseline:

```text
decode mean: +32.985 ms (+8.93%)
decode p50:  +32.589 ms (+8.80%)
extra cache: +4.04 GiB
```

Decision: do not default-enable `awq_group32_interleaved_down_proj` for
Gemma4-31B. The microbenchmark win does not survive whole-model decode, likely
because the larger interleaved cache footprint and global memory/cache behavior
outweigh the per-kernel read co-location benefit. Keep the path as an
experimental policy gate only.
