# FastInference Quickstart

This guide covers the maintained lite-only path: Python 3.12, `uv`, one local
GPU, and the bundled `LLM` / OpenAI-compatible entrypoints.

## Prerequisites

- Linux
- Python 3.12
- One AMD ROCm or CUDA GPU
- `uv`
- Local model directories for model-loading tests and benchmarks

FastInference does not require a project C++ build step. Use `uv` for all
project commands.

## Install

```bash
uv sync
```

If the machine cannot reach Hugging Face directly, set the mirror before model
download or tokenizer access:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

For offline environments, download models ahead of time and pass local paths to
the engine.

## Offline Inference

```python
from vllm import LLM, SamplingParams

llm = LLM(model="models/TinyLlama-1.1B-Chat-v1.0")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64)

outputs = llm.generate(
    ["Hello, my name is", "The capital of France is"],
    sampling_params,
)
for output in outputs:
    print(output.outputs[0].text)
```

## OpenAI-Compatible Server

```bash
uv run python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3.5-9B-AWQ \
  --host 0.0.0.0 \
  --port 8000
```

Smoke test:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/Qwen3.5-9B-AWQ",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 32
  }'
```

## Runtime Profiles

Create a TOML file and point `FASTINFERENCE_CONFIG` at it:

```toml
profile = "balanced"
kv_type = "turbo_int4"

[tuning_keyvals]
FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS = "1"
FASTINFERENCE_KV_MAX_MODEL_LEN = "512"
```

```bash
FASTINFERENCE_CONFIG=configs/local-benchmark.toml \
uv run python -m vllm.entrypoints.openai.api_server \
  --model models/gemma-4-31B-it-AWQ-4bit
```

Production service profiles are `balanced`, `latency`, and `throughput`.
`auto` resolves to `balanced`; `benchmark` and `accuracy` are diagnostic
profiles for measurement and correctness work.

## Regression Commands

```bash
# Fast unit/smoke gate, no full model loads.
bash tests/run_regression_suite.sh

# Local-model quality and semantic gates.
bash tests/run_inference_correctness_regression.sh

# Gemma4 26B/31B end-to-end performance baseline.
uv run python tests/e2e_full_benchmark.py
```

On smaller GPUs, skip the heavier A-tier correctness checks:

```bash
SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh
```

## Supported Models

The maintained regression targets are listed in
[Supported Models](../models/supported_models.md). Do not assume upstream vLLM
model compatibility unless the model has a FastInference adapter, smoke test,
and correctness gate.
