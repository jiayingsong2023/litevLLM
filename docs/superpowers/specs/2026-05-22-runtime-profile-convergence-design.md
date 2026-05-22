# Runtime Profile Convergence Design

## Goal

FastInference will make production inference reproducible by replacing scattered
`FASTINFERENCE_*` runtime branches with a small profile selection surface. The
production path will use fixed, reviewed profiles that preserve inference
accuracy first and then select the fastest validated branch. Benchmark and
diagnostic tools may keep broad tuning controls, but those controls must not
leak into default runtime behavior.

## Scope

This design covers production inference code paths under:

- `vllm/engine/`
- `vllm/model_executor/`
- `vllm/kernels/`
- `vllm/adapters/`
- offline and OpenAI-serving runtime config construction

This design does not remove tuning controls from:

- `tests/tools/`
- kernel microbenchmarks
- grid-search utilities
- diagnostic scripts
- one-off regression tools

Those tools remain the place to prove a branch is both accurate and fast before
it is promoted into a production profile.

## Production Configuration Surface

Production runtime will keep one primary FastInference-specific environment
entry point:

- `FASTINFERENCE_PROFILE`

Allowed values are initially:

- `auto`: choose the recommended stable profile from model capabilities and
  device class.
- `latency`: optimize time per output token and single-request serving.
- `throughput`: optimize aggregate tokens per second for supported batch shapes.
- `accuracy`: prefer higher-fidelity branches when a performance branch has not
  cleared model-specific accuracy gates.
- `benchmark`: use the repository's current benchmark-recommended profile for
  reproducible reports.

Standard deployment variables such as model path, tokenizer path, Hugging Face
cache variables, `CUDA_VISIBLE_DEVICES`, and ROCm/PyTorch process setup remain
outside this cleanup because they are deployment inputs rather than inference
strategy branches. `PYTORCH_ALLOC_CONF` and
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` may remain process initialization
defaults.

## Runtime Profile Model

Introduce a profile layer as the source of truth for runtime policy:

```text
RuntimeProfile
  name
  description
  kv_cache_policy
  scheduler_policy
  backend_policy
  model_policy
  kernel_policy
  verification_contract
```

`RuntimeConfig.from_vllm_config()` will resolve configuration through:

```text
VllmConfig + FASTINFERENCE_PROFILE
  -> RuntimeProfileRegistry.resolve(model capabilities, device class)
  -> RuntimeConfig
```

The registry owns defaults. Environment variables no longer act as implicit
fallbacks for production inference behavior.

## Model Policy Boundary

Model-specific decisions belong in adapters, not generic runtime config.

Examples:

- Gemma4 dense optimized branches such as dense down projection, AWQ GEMV,
  fused gate-up, and QKV/GQA-specific policy belong in `Gemma4Adapter`.
- Gemma4 MoE choices such as the validated int4 decode kernel strategy belong
  in Gemma4 model policy.
- Qwen3.5 stabilizer, SDPA prefill, and residual/input-cap policy belong in
  `Qwen35Adapter`.

The generic runtime may consume normalized policy objects, but it should not
carry model-family fields such as `gemma4_moe_int4_kernel_strategy`.

## Environment Variable Disposition

Every current `FASTINFERENCE_*` production branch will be classified into one
of three actions.

### Delete From Production

Delete or make unreachable in production runtime:

- legacy fallback branches
- force-reference branches
- disable-stabilizer branches
- manual Triton tile overrides
- A/B-only switches
- experimental selective-attention controls
- one-off diagnostic toggles

These may survive only in tools or tests when they are needed for comparison.

### Migrate Into Profile

Move stable, validated choices into profile data:

- KV cache dtype
- block size
- max KV active requests and context cap
- fusion level
- decode priority
- prefill chunking defaults
- GPU greedy sampling behavior
- validated model-specific fast paths
- stable kernel launch policies

### Keep In Tools

Keep broad tuning controls in benchmark and diagnostic utilities:

- autotune toggles
- kernel tile search controls
- grid-search scheduler knobs
- diagnostic profile switches
- temporary model comparison controls

Tool-level controls must not be read by default production runtime modules.

## Verification Contract

Each implementation step must produce an accuracy report and a performance
report.

Accuracy report requirements:

- exact command
- pass/fail result
- covered models or units
- any skipped model reason
- if model behavior changes, correctness regression or model-specific diagnostic
  result

Performance report requirements:

- profile name
- model
- prompt length
- max new tokens
- concurrency
- TTFT
- TPOT or equivalent decode latency
- prefill TPS
- decode TPS
- aggregate TPS
- comparison against the previous baseline for the same profile

Profile promotion rule:

> A branch can become production default only after it passes the relevant
> accuracy gate and is faster or operationally safer than the branch it replaces.

## Implementation Phases

### Phase 1: Profile Infrastructure

Add the profile registry and tests while preserving current behavior. The
runtime should report the resolved profile and policy values in stats and
benchmark output. No environment branches are deleted in this phase.

### Phase 2: Engine Config Convergence

Move `vllm/engine/runtime_config.py` and runtime assembly onto profiles. Keep
`FASTINFERENCE_PROFILE` as the only production FastInference policy selector.
Engine-level scheduler and backend env parsing is replaced by profile values.

### Phase 3: Model Policy Convergence

Move Gemma4 and Qwen3.5 production decisions into adapter-owned model policy.
Remove production reads of model-specific `FASTINFERENCE_*` switches. Retain
tool-level switches only for diagnostics and profile discovery.

### Phase 4: Kernel Policy Convergence

Replace production kernel toggle reads with profile-resolved kernel policy.
Validated launch choices and fused branches become fixed policy. Manual tile and
A/B controls stay in benchmark tools.

### Phase 5: Documentation And Guardrails

Update README, capability matrix, architecture docs, and governance tests so new
production `FASTINFERENCE_*` reads fail unless explicitly allowlisted. Add a
profile promotion checklist requiring accuracy and performance evidence.

## Risks And Mitigations

Risk: removing fallback branches can make debugging harder.

Mitigation: preserve comparison paths in tools and diagnostic scripts, but keep
them outside default runtime.

Risk: fixed profiles may regress on unsupported GPUs.

Mitigation: profile resolution includes device class. Unknown devices choose
accuracy or conservative auto profile until benchmarked.

Risk: benchmark tools and production runtime diverge.

Mitigation: benchmark reports include the resolved profile and all effective
policy values. Promotion requires copying proven settings into profile data and
adding a regression test.

Risk: current tests depend on env switches.

Mitigation: tests that assert production behavior should migrate to profile
fixtures. Tests that compare old and new branches should move under tools or
explicit diagnostic coverage.

## Success Criteria

- Production runtime reads no broad `FASTINFERENCE_*` strategy variables other
  than `FASTINFERENCE_PROFILE`.
- Profile resolution is deterministic and visible in runtime stats.
- Existing supported models keep or improve their accuracy gates.
- Benchmark reports are reproducible from a profile name alone.
- Governance tests prevent new production env branches from being added without
  explicit review.

