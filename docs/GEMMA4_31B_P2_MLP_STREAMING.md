# Gemma4-31B P2 MLP Streaming Fusion

Date: 2026-06-02

## Objective

Evaluate full AWQ MLP streaming fusion for the Gemma4-31B decode-hot shape:

```text
M=1
hidden=5376
intermediate=21504
group_size=32
```

The target computation is:

```text
gate = x @ W_gate
up   = x @ W_up
h    = silu(gate) * up
out  = h @ W_down
```

P1 already provides the verified baseline:

```text
fused gate/up activation + tuned down_proj group32 GEMV
```

## P2a Guardrails Added

Runtime policy helper:

```text
awq_mlp_streaming_fusion_enabled(default=False)
```

Audit events:

```text
mlp_streaming_attempt
mlp_streaming_fallback
```

Runtime helper:

```text
try_fused_awq_mlp_streaming(...)
```

The helper is wired into `Gemma4MLP.forward` before the existing gate/up fused
path, but it is default-off and returns `None` unless the kernel policy enables
`awq_mlp_streaming_fusion`. With the default policy, existing runtime behavior is
unchanged.

## Reference Gate

Tool:

```bash
uv run python tests/tools/validate_gemma4_mlp_streaming_reference.py \
  --hidden 5376 \
  --intermediate 21504 \
  --dtype bfloat16 \
  --json-out /tmp/gemma31_mlp_reference_synthetic_scaled.json
```

Scaled synthetic AWQ result:

```text
cosine   = 0.9999992251396179
max_diff = 0.5
mean_diff = 0.02626691572368145
```

This validates the current two-stage fused path against dense fp32 dequant
reference under a controlled synthetic scale distribution. Real model layer
validation remains required before enabling a new streaming kernel by default.

## Baseline Microbenchmark

Tool:

```bash
uv run python tests/tools/benchmark_gemma4_mlp_streaming_m1.py \
  --warmup 8 \
  --repeat 64 \
  --json-out /tmp/gemma31_mlp_streaming_baseline_scaled.json
```

Result:

```text
current_two_stage = 2.408869981765747 ms/layer
```

A streaming candidate should beat this by at least 15%, so the acceptance target
is approximately:

```text
<= 2.05 ms/layer
```

## P2c Experimental Recompute Candidate

Implemented an isolated negative-control kernel:

```text
packed_int4_symmetric_mlp_streaming_m1_recompute_safe
```

This candidate computes a full MLP output in one Triton kernel, but it recomputes
`gate` and `up` for every `down_proj` output tile. It is intentionally not wired
as a production fast path.

Correctness smoke:

```bash
uv run pytest \
  tests/test_awq_gemm_m1_specialization.py::test_mlp_streaming_recompute_candidate_matches_dense_reference \
  -q
```

Result:

```text
1 passed
```

Short exact-shape benchmark:

```bash
uv run python tests/tools/benchmark_gemma4_mlp_streaming_m1.py \
  --warmup 2 \
  --repeat 8 \
  --json-out /tmp/gemma31_mlp_streaming_p2c_recompute_short.json
```

Result:

```text
current_two_stage              = 2.609935 ms/layer
candidate_streaming_recompute = 74.648613 ms/layer
```

Conclusion: naive one-kernel fusion is rejected. The slowdown is consistent with
the expected repeated gate/up work across down output tiles. Any production P2c
path must be cooperative or persistent, computing gate/up once and sharing the
intermediate across down reduction work.

## P2c Design Cost Model

Tool:

```bash
uv run python tests/tools/analyze_gemma4_mlp_streaming_design.py \
  --hidden 5376 \
  --intermediate 21504 \
  --block-n 64 \
  --block-i 16 \
  --json-out /tmp/gemma31_mlp_streaming_design_costs.json
```

Result:

```text
current_two_stage estimated_bytes              = 198,739,968
naive_recompute estimated_bytes                = 10,989,748,224
naive_recompute gate_up_read_multiplier        = 84
intermediate_tile_partial estimated_bytes      = 252,887,040
intermediate_tile_partial fp32 partial_bytes   = 28,901,376
true_shared_intermediate estimated_bytes       = 195,127,296
```

Decisions:

```text
naive_recompute_single_kernel       reject
intermediate_tile_partial_staging   reject
true_shared_intermediate_required   requires_cross_program_sharing
```

The current two-stage path writes only the 21,504-value activation vector
(~43 KB bf16). Even though down_proj rereads that vector for each output tile,
the total activation traffic is small relative to packed INT4 weight traffic.
Partial staging avoids activation rereads but introduces a ~28.9 MB fp32 partial
buffer plus a reduce pass, so it is not a better P2c target.

With plain Triton programs, there is no cross-program LDS/shared-memory exchange
for the gate/up intermediate across all down output tiles. Therefore the only
theoretically attractive design requires a different execution model: a
persistent/cooperative kernel with explicit cross-tile sharing, or a backend
primitive outside the current hand-written Triton pattern.

Runtime audit behavior now reports this explicitly when
`awq_mlp_streaming_fusion` is enabled but no viable production kernel exists:

```text
mlp_streaming_attempt = 1
mlp_streaming_fallback reason = requires_cross_program_sharing
```

## Kernel Decision

Do not implement a naive single-kernel `gate/up/down` fusion. If each
`down_proj` output tile recomputes `gate` and `up`, the gate/up GEMV work is
repeated across many output tiles and the kernel is expected to regress.

Acceptable future candidates must compute gate/up once per intermediate tile and
feed `down_proj` reduction without global materialization. That likely requires
a cooperative/persistent design or a carefully staged two-kernel design. Until
that exists, the current P1 two-stage path remains the runtime default.

## Verification

Targeted checks completed during P2a/P2b:

```bash
uv run python -m py_compile \
  vllm/kernels/triton/awq_fused_gemm.py \
  vllm/model_executor/models/_fused_awq_pair.py \
  vllm/model_executor/models/gemma4/mlp.py \
  tests/tools/validate_gemma4_mlp_streaming_reference.py \
  tests/tools/benchmark_gemma4_mlp_streaming_m1.py

uv run pytest \
  tests/test_awq_gemm_m1_specialization.py::test_mlp_streaming_fusion_default_disabled_and_policy_overridable \
  tests/test_awq_audit_summary.py::test_awq_audit_summary_tracks_mlp_streaming_events \
  tests/test_awq_audit_summary.py::test_mlp_streaming_helper_default_disabled_records_fallback \
  -q
```

