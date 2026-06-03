# Lite-Only Status

This page records the current architectural boundary. For model and feature
status, use [CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md).

## Current Position

FastInference is maintained as a lite-only, single-GPU inference runtime. The
official path is:

- Offline: `vllm.LLM` / `vllm.entrypoints.llm.LLM` -> `LiteEngine`
- Async/server: `AsyncLLM` -> `LiteEngine`
- HTTP: `vllm.entrypoints.openai.api_server`

The maintained regression targets are:

- `TinyLlama-1.1B`
- `Qwen3.5-9B-AWQ`
- `Gemma4-26B-A4B-it-AWQ-4bit`
- `Gemma4-31B-it-AWQ-4bit`

`Qwen3.5-35B` and the full upstream vLLM model surface are not official support
targets.

## Delivered Architecture Work

- Engine control plane is decomposed into scheduler, request lifecycle,
  prefill/decode executors, sampling, output assembly, observer, and errors.
- Runtime policy flows through `FastInferenceConfig`, runtime profiles, and
  `RuntimeConfig`.
- Model capability decisions are centralized under `vllm/adapters/`.
- Gemma4 model code has been split from a large monolithic module into the
  `vllm/model_executor/models/gemma4/` package.
- `vllm/worker/`, `vllm/core/`, and `vllm/distributed/` have been removed from
  the code tree.

## Still Present But Not A Second Runtime

The following code may exist for compatibility, vendored kernels, or
experimental feature surface:

- `vllm/model_executor/warmup/`
- LoRA and multimodal runtime hooks

Their support level is defined by the capability matrix, not by upstream vLLM
feature claims.

## Configuration Boundary

The public production configuration entrypoint is `FASTINFERENCE_CONFIG`,
pointing at a TOML file:

```toml
profile = "benchmark"
kv_type = "turbo_int4"

[tuning_keyvals]
FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS = "1"
FASTINFERENCE_KV_MAX_MODEL_LEN = "512"
```

Supported profiles are `auto`, `benchmark`, `latency`, `throughput`, and
`accuracy`. Deprecated or tool-only `FASTINFERENCE_*` names remain registered
for compatibility, but new production policy should be expressed through
config/profile fields.

## Removed Upstream Runtime Scope

- Worker-based GPU runtime
- Distributed tensor/pipeline/data parallel runtime
- Upstream block manager / scheduler core
- Full upstream model registry compatibility promise
- Upstream Transformers modeling backend wrappers
- Generic upstream asset downloader and multimodal audio/parser helpers
- C++/CUDA extension source in the maintained path
- Speculative decoding as a supported feature
- Upstream CLI, pooling, gRPC, executor, and vendored third-party Triton kernel
  compatibility paths

## Onboarding Rule

New model support should be implemented through adapter, loader, model, and
policy modules. Do not add model-name conditionals to `LiteEngine`,
`step_scheduler.py`, executor modules, or generic hot-path loader code.
