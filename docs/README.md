---
hide:
  - navigation
---

# FastInference Documentation

FastInference is a lite-only, single-GPU inference runtime derived from vLLM.
The maintained path is pure Python plus Triton, with AMD ROCm as the primary
tuning target and CUDA compatibility kept where the same path supports it.

This documentation covers the current repository, not upstream vLLM. For model
and feature status, use the capability matrix as the source of truth.

## Start Here

- [Quickstart](getting_started/quickstart.md) - local setup, offline inference,
  OpenAI-compatible serving, and regression commands.
- [Capability Matrix](CAPABILITY_MATRIX.md) - supported, experimental,
  compatibility, and unsupported areas.
- [Architecture](ARCHITECTURE_LITE.md) - current lite runtime control plane and
  execution path.
- [API Reference](API_REFERENCE.md) - maintained HTTP surface for the bundled
  OpenAI-compatible server.
- [Inference Accuracy](INFERENCE_ACCURACY.md) - correctness tiers and local
  model quality gates.

## Maintained Runtime Path

```text
LLM / AsyncLLM / OpenAI API Server
  -> vllm/serving/config_builder.py
  -> vllm/engine/lite_engine.py
  -> vllm/engine/step_scheduler.py
  -> vllm/engine/request_scheduler.py
  -> vllm/engine/prefill_executor.py + vllm/engine/decode_executor.py
  -> vllm/engine/sampling_driver.py
  -> vllm/engine/output_pipeline.py
```

The control plane lives in `vllm/engine/`; model capability policy belongs in
`vllm/adapters/`; hot kernels live under `vllm/kernels/triton/` and use
`vllm/triton_utils/` for Triton imports.

## Configuration

Production configuration is resolved through `FastInferenceConfig` and
`RuntimeConfig`. The public environment entrypoint is `FASTINFERENCE_CONFIG`,
which points at a TOML file:

```toml
profile = "benchmark"
kv_type = "turbo_int4"

[tuning_keyvals]
FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS = "1"
FASTINFERENCE_KV_MAX_MODEL_LEN = "512"
```

Supported profile names are `auto`, `benchmark`, `latency`, `throughput`, and
`accuracy`. Legacy `FASTINFERENCE_*` switches may still exist for tests,
benchmarks, or compatibility, but they are not the preferred production
configuration surface.

## Current Regression Targets

- `TinyLlama-1.1B`
- `Qwen3.5-9B-AWQ`
- `Gemma4-26B-A4B-it-AWQ-4bit`
- `Gemma4-31B-it-AWQ-4bit`
- `DeepSeek-V4-Flash` target GGUF, experimental and opt-in for correctness

Run the fast structural gate with:

```bash
bash tests/run_regression_suite.sh
```

Run model correctness gates with:

```bash
bash tests/run_inference_correctness_regression.sh
```

When the target GGUF exists at `MODEL_DEEPSEEK_V4_FLASH_GGUF` or the default
`models/DeepSeek-V4-Flash-ds4/...imatrix.gguf` path, the script runs the
experimental DeepSeek Tier-B smoke by default. Set
`RUN_DEEPSEEK_V4_FLASH_GPU_SMOKE=0` to skip it.

Run the default end-to-end benchmark set, including the DeepSeek direct-GGUF
smoke when the model file is present, with:

```bash
uv run python tests/e2e_full_benchmark.py
```
