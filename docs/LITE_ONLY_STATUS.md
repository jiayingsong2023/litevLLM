# Lite-Only Architecture Status

## Delivery Snapshot
As of 2026-05-29, within the current Lite runtime scope:

- `P0` delivered — Gemma4 全局状态隔离（`Gemma4LayerConfig` dataclass, 消除模块级 mutable global）
- `P1` delivered — policy key 常量 + TypedDict + RuntimeAssemblyContext 类型强化 + StepScheduler 参数聚合
- `P2` delivered — 工具函数去重 + 异常日志细化
- Legacy cleanup: 删除 `vllm/worker/`、`vllm/core/`、`vllm/distributed/`、`vllm/third_party/`（~55k 行）
- Test cleanup: 删除 `tests/reports/`（2.4MB 历史性能数据）+ DeepSeek 诊断脚本
- Config migration: 所有运行时 tuning 参数从 `os.environ` 迁移到 TOML `[tuning_keyvals]`
- Explicit deferred items: `speculative decoding`、模型适配器拆分、`gemma4.py` 大文件拆分

For an external-facing DONE/KNOWN_GAPS list, see:
- `docs/DELIVERY_STATUS.md`

For the current supported / experimental / compatibility / unsupported surface,
use `docs/CAPABILITY_MATRIX.md` as the source of truth.

## Current Project Position
FastInference is now a `lite-only` single-GPU inference project. The repository no longer treats `Qwen3.5-35B` as an officially supported model. The maintained support surface is the lite runtime path plus the current regression targets:

- `TinyLlama-1.1B`
- `Qwen3.5-9B-AWQ`
- `Gemma4-26B-A4B-it-AWQ-4bit` (MoE, A-strict + A-lite + B)
- `Gemma4-31B-it-AWQ-4bit` (Dense, A-lite + B)

The project no longer contains `vllm/worker/`, `vllm/core/`, `vllm/distributed/`, or `vllm/third_party/`. These upstream vLLM subsystems were deleted as part of the lite-only cleanup.

## Official Runtime Path
The only official runtime path is:

- Offline sync: `vllm.entrypoints.llm.LLM` -> `vllm.engine.lite_engine.LiteEngine`
- Async/server path: `vllm.engine.async_llm.AsyncLLM` -> `vllm.engine.lite_engine.LiteEngine`

Internal decomposition:

- `vllm/serving/config_builder.py` — TOML config -> VllmConfig + RuntimeConfig
- `vllm/adapters/*` — model adapter with TypedDict policy keys
- `vllm/engine/step_scheduler.py` — step-level scheduling (Lora/MultiModal params aggregated)
- `vllm/engine/request_scheduler.py` — request/slot lifecycle
- `vllm/engine/prefill_executor.py` — prefill (hardware SDPA)
- `vllm/engine/decode_executor.py` — decode (Triton PagedAttention)
- `vllm/engine/sampling_driver.py` — sampling
- `vllm/engine/output_pipeline.py` — output assembly
- `vllm/engine/runtime_observer.py` — observability
- `vllm/engine/errors.py` — error semantics

## Key Architectural Changes (2026 Q2)

### Gemma4 Instance Isolation
Module-level global mutable state (`_GEMMA4_TUNING`, `_GEMMA4_PROFILE_ENABLED`, etc.) has been replaced with per-instance `Gemma4LayerConfig`. Model layers receive their configuration via constructor injection rather than reading module-level variables.

### Policy Typing
Policy keys (`model_policy`, `kernel_policy`) are now defined as named constants in `vllm/adapters/policy_keys.py` with `TypedDict` type constraints.

### StepScheduler Parameter Aggregation
LoRA and MultiModal scheduling constraints are aggregated into `LoraSchedulingParams` and `MultiModalSchedulingParams` frozen dataclasses, reducing the `StepScheduler.__init__` signature from ~40 to ~18 parameters.

### TOML Config
All runtime tuning parameters are now configured via TOML `[tuning_keyvals]` section. Environment variables are deprecated — the config file is the sole source of truth.

## What Was Removed (P0 + P1 + P2)
- `vllm/worker/` — GPU worker, model runner, CUDA graph, sampling pipeline
- `vllm/core/` — block pool, KV cache manager, scheduler
- `vllm/distributed/` — parallel state
- `vllm/third_party/` — flashmla, triton_kernels
- `vllm/model_executor/warmup/` — kernel warmup
- `vllm/model_executor/layers/mamba/` — Mamba layers
- `vllm/model_executor/layers/kda.py` — KDA module
- `tests/reports/` — 33 historical perf JSON files (2.4MB)
- DeepSeek diagnostic scripts (not a supported model)
- `stress_test_64k.py` — unreferenced stress test

## Regression Status
Validated after the current refactor set (2026-05-29):

- `bash tests/run_regression_suite.sh` — 113 pass, 1 pre-existing failure
- `SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh` — 4/4 models PASS
- `uv run python tests/e2e_full_benchmark.py` — both Gemma4 models PASS
- `uv run pytest tests/test_gemma4_global_state_isolation.py -v` — 3/3 PASS

## Adapter / Policy Onboarding Checklist

### Adapter Interfaces
Add a model adapter module under `vllm/adapters/`, with responsibilities:

- expose head counts, KV head counts, head dim, layer count, and max model length
- declare whether the model uses dense attention, hybrid attention, or MoE routing
- define attention metadata expectations for prefill and decode
- expose model capability flags

### Policy Interfaces
Only add model policy code if the model truly needs it. Keep it out of `LiteEngine`.

Recommended interfaces:

- `prompt_policy`: optional prompt wrapping or chat-template normalization
- `logits_policy`: optional logit bias or stop-token shaping
- `output_policy`: optional output cleanup
- `moe_policy`: expert residency, CPU offload, FP8 enablement, packed-weight compatibility

### Test Requirements
Before claiming support, add:

- load smoke test
- one-step prefill correctness test vs reference
- Tier-B generation smoke
- MoE routing / expert weight correctness test if applicable

### Design Rule
New model support should be implemented by adding adapter, loader, and policy modules. It should not reintroduce model-name-based conditionals inside `LiteEngine`, `step_scheduler.py`, executor modules, or generic loader hot paths.
