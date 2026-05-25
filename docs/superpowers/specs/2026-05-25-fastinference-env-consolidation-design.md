# FastInference Environment Variable Consolidation Design

## Goal

Reduce public `FASTINFERENCE_*` runtime environment variables to fewer than 10 without reducing inference correctness or inference performance for the existing Qwen3.5-9B and Gemma4-26B/31B regression targets.

## Current State

The repository currently contains 153 concrete `FASTINFERENCE_*` names after excluding prefix placeholders and the internal `_FASTINFERENCE_TUNING_ENV_PREFIX` constant. Eighty-four names appear in `vllm/` runtime code, and the rest appear only in tests, tools, or documentation.

The largest uncontrolled groups are:

- `FASTINFERENCE_AWQ_*`: kernel launch, fused GEMM, split-k, cache, and fallback tuning.
- `FASTINFERENCE_GEMMA4_*` and `FASTINFERENCE_GEMMA31B_*`: model policy, diagnostics, MoE, and benchmark tuning.
- `FASTINFERENCE_QWEN35_*`, `FASTINFERENCE_MOE_*`, and `FASTINFERENCE_DEEPSEEK_*`: model-specific accuracy and memory policy toggles.
- `FASTINFERENCE_LITE_*`, `FASTINFERENCE_BENCH_*`, and `FASTINFERENCE_GPU_GREEDY_*`: scheduler, benchmark, and sampling experiments.

The current bridge in `vllm/engine/runtime_config.py` captures every `FASTINFERENCE_*` environment variable into `RuntimeConfig.tuning_env`. That broad capture keeps older tuning paths working, but it also prevents real governance because any new variable automatically becomes available to lower layers.

## Target Public Interface

The final public runtime environment interface is:

```text
FASTINFERENCE_PROFILE
FASTINFERENCE_CONFIG
FASTINFERENCE_KV_TYPE
FASTINFERENCE_DEBUG
FASTINFERENCE_LOG_LEVEL
FASTINFERENCE_BENCH_PROFILE
FASTINFERENCE_ALLOW_LEGACY_ENV
```

`FASTINFERENCE_KV_FP8` remains as a short-term deprecated compatibility alias for `FASTINFERENCE_KV_TYPE=fp8`, but it is not part of the final public interface.

No other `FASTINFERENCE_*` variable may affect production inference. Test-only and tool-only variables are allowed only if their names are registered as non-runtime and only if they are read outside `vllm/` production paths.

## Architecture

Add a central environment registry in `vllm/engine/env_registry.py`. The registry owns the allowlist, deprecation state, scope, replacement guidance, and parsing behavior for every accepted public runtime variable. Production code may only read `FASTINFERENCE_*` through this registry or through `RuntimeProfileRegistry.resolve_from_env`.

Move model, scheduler, and kernel tuning into typed runtime/profile policy objects:

- Runtime profile fields for KV, scheduler, prefill, warmup, queue, and general runtime policy.
- Adapter-owned policy for model-specific behavior such as Gemma4 and Qwen3.5 accuracy/performance modes.
- Kernel policy fields for AWQ and paged-attention launch tuning.
- Benchmark CLI arguments or JSON configs for benchmark-only knobs.

Replace the broad `FASTINFERENCE_*` capture with a narrow compatibility bridge. The bridge reads deprecated variables only when `FASTINFERENCE_ALLOW_LEGACY_ENV=1` is set, maps supported legacy names to profile/config fields, and emits a warning that names the replacement. The default production path must ignore legacy variables once the first migration phase is complete.

## Migration Strategy

### Phase 1: Freeze and Register

Introduce the registry and governance tests before removing behavior. The tests must fail if a new `FASTINFERENCE_*` appears outside the registry. They must also fail if production files read `os.environ` or `getenv` for `FASTINFERENCE_*` directly.

The registry initially marks current public variables as one of:

- `public`: final public runtime interface.
- `deprecated`: accepted only through the compatibility bridge.
- `tool_only`: accepted in tests/tools, never in production inference.
- `removed`: documented historical name that must not be read.

### Phase 2: Replace Broad Tuning Capture

Remove broad captures from `RuntimeConfig` and `LiteInferenceConfig`. Replace them with a structured `legacy_tuning_overrides` object that is populated only when legacy env compatibility is explicitly enabled.

Keep exact behavior for profiles used by the correctness and performance regressions. The default profile must preserve current production defaults.

### Phase 3: Collapse AWQ and Kernel Env

Move `FASTINFERENCE_AWQ_*` launch tuning into kernel policy fields and persistent profile data. Runtime inference should consume policy values, not environment variables. Tool-only AWQ benchmark overrides move to command-line flags in `tests/tools/bench_awq_fused_gemm_ab.py` and related profiling scripts.

The migrated policy must preserve the current fused AWQ defaults used by Gemma4-26B/31B. If a tuning value is required for the benchmark profile, it belongs in the benchmark profile or an explicit config file rather than a standalone env variable.

### Phase 4: Collapse Model-Specific Env

Move Gemma4 and Qwen3.5 accuracy/performance switches into adapter policies. Accuracy profile must select HF-faithful settings for Qwen3.5 full attention and must not silently enable stabilizers that change strict prefill argmax behavior.

Gemma4 diagnostics such as layer profiling and ROCtx profiling become `FASTINFERENCE_DEBUG` submodes or tool CLI flags, not public model-specific env.

### Phase 5: Clean Tests, Docs, and Historical Plans

Update scripts and docs to use the final public variables, CLI arguments, or config files. Historical design/plan documents may keep old names as historical references, but governance tests should exclude archived docs from runtime allowlist checks.

## Correctness and Performance Gates

Every migration phase must pass:

```bash
bash tests/run_regression_suite.sh
uv run pytest tests/test_project_governance.py -q
bash tests/run_inference_correctness_regression.sh
```

For phases that touch model policy, KV cache, attention, AWQ, or scheduler behavior, also run the model-specific gates:

```bash
bash tests/run_inference_accuracy_regression.sh
uv run python tests/tools/profile_gemma4_layer_breakdown.py --model models/gemma-4-31B-it-AWQ-4bit --prompt-tokens 128 --decode-steps 8 --warmup-decode 2 --max-new-tokens 24 --max-num-batched-tokens 512 --max-model-len 512
```

Acceptance requires:

- Qwen3.5-9B strict prefill audit remains passing with cosine similarity at or above the current threshold and matching argmax.
- Gemma4-26B and Gemma4-31B Tier-B and A-lite correctness remain passing.
- Gemma4-26B A-strict remains passing.
- Gemma4-31B does not require A-strict prefill audit.
- Gemma4-26B/31B benchmark metrics do not regress materially from the captured baseline unless the regression is explained and approved.

## Compatibility Policy

Deprecated variables are supported only for one migration window. When `FASTINFERENCE_ALLOW_LEGACY_ENV=1` is set, the registry maps selected legacy variables into structured config and prints one warning per variable. Without that flag, legacy variables are ignored and listed in a startup warning.

Variables that only existed in tests, abandoned experiments, or historical docs should not receive compatibility aliases. They should be deleted from active tests/tools or moved to explicit CLI flags.

## Non-Goals

This work does not introduce new model features, new kernels, or new performance tuning behavior. It consolidates control surfaces and preserves current correctness/performance behavior.

This work does not remove support for runtime profiles. Profiles are the main replacement for most environment variables.

## Success Criteria

- Fewer than 10 final public `FASTINFERENCE_*` runtime variables.
- No broad `FASTINFERENCE_*` capture in production runtime paths.
- No direct production `os.environ.get("FASTINFERENCE_...")` outside the registry/profile resolver.
- Governance tests prevent newly introduced unmanaged variables.
- Existing Qwen3.5-9B and Gemma4-26B/31B correctness gates pass.
- Existing Gemma4 performance baselines do not regress.
