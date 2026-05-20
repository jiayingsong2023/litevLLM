# Gemma4-26B MoE Strict Profile Breakdown - 2026-05-20

Command:

```bash
uv run python tests/tools/profile_gemma4_layer_breakdown.py \
  --model models/gemma-4-26B-A4B-it-AWQ-4bit \
  --prompt-tokens 384 \
  --max-new-tokens 40 \
  --warmup-decode 4 \
  --decode-steps 24 \
  --max-num-batched-tokens 512 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.55 \
  --kv-type turbo_int4 \
  --torch-profiler \
  --json-out tests/reports/gemma4_26b_strict_profile_breakdown_20260520.json
```

The profiler synchronizes GPU work at span boundaries, so the absolute wall
times below are profiling overhead numbers. Use the shares and counts to locate
bottlenecks; use the normal benchmark reports for production latency.

## Key Finding

This report captures the pre-fix strict profile that exposed the activation
capability gap. Current code supports `gelu_pytorch_tanh` in the packed-int4
MoE decode path, and the default 26B decode strategy is now `two_stage`; see
`gemma4_26b_default_after_prefill_fused_profile_20260520.json` for the current
default profile.

The current Gemma4-26B A4B MoE path is not using the packed-int4 decode kernel.
The profile records `moe_int4_decode_fallback:unsupported_activation` for all
decode MoE attempts. The checkpoint config uses:

```json
"hidden_activation": "gelu_pytorch_tanh"
```

but the packed-int4 MoE decode validator accepted only `silu/swish` at the
time this report was captured. Therefore the then-experimental batched MoE
int4 decode strategy was not active for this 26B checkpoint; it fell back to
per-expert AWQ materialization and dense expert matmuls.

## Prefill Breakdown

Profiled wall: `5515.377 ms` for the prefill phase.

| Component | Time ms | Share | Count | Avg ms |
| --- | ---: | ---: | ---: | ---: |
| `layer_moe_sparse_experts` | 4300.893 | 77.98% | 30 | 143.363 |
| `moe_materialize_one_expert_awq` | 3108.872 | 56.37% | 2477 | 1.255 |
| `layer_self_attn` | 889.321 | 16.12% | 30 | 29.644 |
| `moe_sparse_expert_linear` | 799.766 | 14.50% | 2477 | 0.323 |
| `moe_sparse_expert_gate_up` | 360.781 | 6.54% | 2477 | 0.146 |
| `attn_q_proj` | 304.760 | 5.53% | 30 | 10.159 |
| `moe_sparse_expert_down_reduce` | 255.179 | 4.63% | 2477 | 0.103 |
| `attn_local_prefill` | 215.060 | 3.90% | 25 | 8.602 |

Runtime overhead is not the bottleneck:

| Component | Time ms | Share |
| --- | ---: | ---: |
| `prefill_executor.execute` | 5506.257 | 99.83% |
| `sampling.sample_next_token` | 4.369 | 0.08% |
| `scheduler.build_plan` | 0.211 | 0.00% |

PyTorch/ROCm profiler launch summary:

| Metric | Value |
| --- | ---: |
| CPU self time | 5017.670 ms |
| Kernel launch count | 67012 |
| Kernel launch CPU time | 648.408 ms |
| Device kernel time | 0.000 ms |

`device_kernel_ms` is zero because this ROCm profiler run did not expose device
event durations through PyTorch `key_averages`; CPU launch count/time is still
reported.

## Decode Breakdown

Profiled wall: `7533.077 ms` for 24 decode steps, `313.878 ms/step` under
instrumentation.

| Component | Time ms | Share | Count | Avg ms |
| --- | ---: | ---: | ---: | ---: |
| `layer_moe_sparse_experts` | 4204.596 | 55.82% | 720 | 5.840 |
| `moe_sparse_expert_linear` | 1846.581 | 24.51% | 5760 | 0.321 |
| `layer_self_attn` | 1793.528 | 23.81% | 720 | 2.491 |
| `moe_materialize_one_expert_awq` | 1397.658 | 18.55% | 5760 | 0.243 |
| `moe_sparse_expert_gate_up` | 851.200 | 11.30% | 5760 | 0.148 |
| `moe_sparse_expert_down_reduce` | 601.488 | 7.98% | 5760 | 0.104 |
| `layer_dense_mlp` | 366.638 | 4.87% | 720 | 0.509 |
| `attn_o_proj` | 340.261 | 4.52% | 720 | 0.473 |

Runtime overhead is also not the bottleneck in decode:

| Component | Time ms | Share |
| --- | ---: | ---: |
| `decode_executor.execute_sync_fast` | 7521.357 | 99.84% |
| `sampling.sample_next_token` | 4.650 | 0.06% |
| `scheduler.build_plan` | 0.886 | 0.01% |

PyTorch/ROCm profiler launch summary:

| Metric | Value |
| --- | ---: |
| CPU self time | 6595.821 ms |
| Kernel launch count | 160050 |
| Kernel launch CPU time | 1171.245 ms |
| Device kernel time | 0.000 ms |

## AWQ Projection Counters

Attention AWQ projections use fused packed-int4 GEMM successfully:

| Projection | Matmul calls | Fused attempts | Fused success | Success rate |
| --- | ---: | ---: | ---: | ---: |
| `attn_q_proj` | 1230 | 1200 | 1200 | 97.56% |
| `attn_k_proj` | 1230 | 1200 | 1200 | 97.56% |
| `attn_v_proj` | 1025 | 1000 | 1000 | 97.56% |
| `attn_o_proj` | 1230 | 1200 | 1200 | 97.56% |

This points away from attention projection policy as the primary issue. MoE is
the dominant cost.

## Decision

Do not continue blind tile/layout experiments yet. The next high-confidence
step is to make the packed-int4 MoE kernels support Gemma4's
`gelu_pytorch_tanh` activation and then rerun this exact profile. Until that is
fixed, decode optimization experiments are measuring the fallback path rather
than the intended packed-int4 MoE kernel path.

After GELU support lands, rerun:

1. This strict profile report.
2. `gemma4_26b_a4b` default benchmark.
3. Gemma4-26B strict correctness regression.
