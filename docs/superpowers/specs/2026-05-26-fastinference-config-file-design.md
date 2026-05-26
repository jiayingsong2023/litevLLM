# FastInference Config File Migration Design

Date: 2026-05-26

## Goal

Move long-lived FastInference runtime controls out of environment variables and into explicit configuration files or API-provided configuration objects. Environment variables may remain only as short-term compatibility or process-bootstrap mechanisms.

The immediate target is:

- production inference behavior is derived from `FastInferenceConfig`, not from public `FASTINFERENCE_*` runtime variables;
- `FASTINFERENCE_CONFIG` remains temporarily as a config-file locator;
- CLI/API call sites can pass a config path or config object directly;
- legacy `FASTINFERENCE_*` tuning variables are ignored by default and only admitted through explicit config-based compatibility mode.

## Non-Goals

- Do not remove every legacy tuning knob in the first implementation stage.
- Do not change model math, kernel heuristics, scheduler behavior, or benchmark defaults unless required to preserve existing behavior through config.
- Do not add a dependency-heavy configuration framework.
- Do not make tests rely on ambient process environment for normal runtime behavior.

## Current State

The public runtime surface has been reduced to seven `FASTINFERENCE_*` names:

- `FASTINFERENCE_PROFILE`
- `FASTINFERENCE_CONFIG`
- `FASTINFERENCE_KV_TYPE`
- `FASTINFERENCE_DEBUG`
- `FASTINFERENCE_LOG_LEVEL`
- `FASTINFERENCE_BENCH_PROFILE`
- `FASTINFERENCE_ALLOW_LEGACY_ENV`

However, the execution path still contains many deprecated or tool-only names through legacy compatibility, AWQ/Gemma4/MoE tuning snapshots, diagnostics, and warmup helpers. The public surface is smaller, but the long-term control plane is still environment-shaped.

## Recommended Approach

Use a config-first architecture with a short transition path.

1. Add a strongly typed `FastInferenceConfig` dataclass.
2. Add a loader that accepts:
   - an explicit config object;
   - an explicit config file path;
   - temporary fallback from `FASTINFERENCE_CONFIG`;
   - built-in defaults.
3. Change production runtime construction to pass the resolved config into `RuntimeProfileRegistry` and `RuntimeConfig`.
4. Keep legacy env admission behind `legacy_env.enabled = true` in the config.
5. Keep `FASTINFERENCE_CONFIG` as a locator only. It must not directly change inference policy.

This gives an immediate clean production path while preserving a controlled escape hatch for existing benchmark and diagnostic flows.

## Configuration Format

Use TOML as the primary on-disk format. JSON can be supported later if needed, but TOML is easier for operators to read and edit.

Minimal example:

```toml
profile = "benchmark"
kv_type = "turbo_int4"
debug = false
log_level = "info"

[benchmark]
profile = "default"

[legacy_env]
enabled = false
```

Fields:

- `profile`: one of `auto`, `latency`, `throughput`, `accuracy`, `benchmark`.
- `kv_type`: one of `auto`, `fp16`, `fp8`, `turbo_int4`.
- `debug`: boolean diagnostic flag.
- `log_level`: logging level string.
- `benchmark.profile`: benchmark profile selector, replacing `FASTINFERENCE_BENCH_PROFILE`.
- `legacy_env.enabled`: when true, deprecated registered env variables may be collected into `RuntimeConfig.tuning_env`.

The initial schema should stay deliberately small. Model- and kernel-specific policies should live under typed policy fields later, not as arbitrary environment-variable maps.

## Resolution Order

Runtime config resolution must use this precedence:

1. Explicit API config object.
2. Explicit API/CLI config path.
3. Temporary `FASTINFERENCE_CONFIG` fallback path.
4. Built-in defaults.

Once a config is resolved, regular public env variables such as `FASTINFERENCE_PROFILE` and `FASTINFERENCE_KV_TYPE` must not override it. This is the main behavioral change.

`FASTINFERENCE_CONFIG` is a bootstrap locator only. It is allowed during transition because it does not encode runtime policy by itself.

## Runtime Integration

Add `vllm/engine/fastinference_config.py` containing:

- `FastInferenceConfig`
- `BenchmarkConfig`
- `LegacyEnvConfig`
- `load_fastinference_config(path: str | Path | None)`
- `resolve_fastinference_config(config: FastInferenceConfig | None, path: str | Path | None)`

Update `RuntimeProfileRegistry`:

- keep `resolve()` as the pure resolver;
- add or use `resolve_from_config(config, model_capabilities, gpu_total_gb)`;
- mark `resolve_from_env()` as legacy and remove it from production construction.

Update `RuntimeConfig.from_vllm_config()`:

- accept or discover a resolved `FastInferenceConfig`;
- read `profile` and `kv_type` from config;
- collect deprecated env only when `config.legacy_env.enabled` is true;
- still write the resolved requested profile into `tuning_env` for existing adapter/kernel snapshot compatibility during transition.

Update `build_vllm_config()`:

- accept `fastinference_config` and `fastinference_config_path` keyword arguments;
- attach the resolved config to `VllmConfig`;
- pass it into `RuntimeConfig.from_vllm_config()`.

## Legacy Compatibility

The previous public flag `FASTINFERENCE_ALLOW_LEGACY_ENV` becomes:

```toml
[legacy_env]
enabled = true
```

When disabled, ambient deprecated variables are ignored. When enabled, only registered deprecated variables are collected through `env_registry.collect_runtime_tuning_env()`.

This keeps old experiments recoverable without letting undeclared environment variables silently influence production inference.

## Error Handling

- Missing config path: fail fast with a clear file-not-found error.
- Invalid TOML: fail fast with parser context.
- Unknown top-level field: fail fast in strict mode. Initial implementation should be strict by default.
- Invalid `profile` or `kv_type`: fail fast rather than silently falling back.
- `FASTINFERENCE_CONFIG` pointing to a bad file should fail the same way as an explicit bad path.

Failing fast is preferable because a typo in config should not silently change model accuracy or performance.

## Tests

Add focused unit tests:

- default config resolves to current benchmark-compatible behavior;
- explicit config object wins over config path and env;
- explicit config path wins over `FASTINFERENCE_CONFIG`;
- `FASTINFERENCE_CONFIG` is honored only as a path fallback;
- `FASTINFERENCE_PROFILE` and `FASTINFERENCE_KV_TYPE` no longer affect production runtime config;
- `legacy_env.enabled = false` ignores deprecated env variables;
- `legacy_env.enabled = true` admits only registered deprecated variables;
- invalid profile, invalid KV type, unknown fields, and missing file fail clearly.

Run after implementation:

```bash
bash tests/run_regression_suite.sh
bash tests/run_inference_correctness_regression.sh
uv run python tests/e2e_full_benchmark.py --models tinyllama,qwen35_9b_awq,gemma4_26b_a4b,gemma4_31b_q4 --model-process-isolation
```

The full correctness and performance runs are required because recent fixes showed that environment/config changes can create false passes or performance shifts.

## Migration Plan

Stage 1: Add config model and loader.

Stage 2: Route production `RuntimeProfile` and `RuntimeConfig` through the resolved config.

Stage 3: Update CLI/API/test helpers to pass config path or config object explicitly.

Stage 4: Keep `FASTINFERENCE_CONFIG` as the only transitional public environment variable.

Stage 5: Move AWQ/Gemma4/MoE legacy tuning snapshots into typed config policy fields and reduce execution-path env reads.

Stage 6: Deprecate and later remove `FASTINFERENCE_CONFIG` once deployment entry points can always pass explicit config.

## Open Decisions

None for the first implementation stage. The chosen transition policy is:

- short-term `FASTINFERENCE_CONFIG` is allowed as a config-file locator;
- production deployment should prefer explicit config path or API config object;
- all other public inference controls should migrate into config fields.
