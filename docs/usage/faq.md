# Frequently Asked Questions

## Why is this project called lite-only?

FastInference keeps a single-GPU pure Python + Triton runtime path and removes
the upstream worker/distributed runtime boundary from the maintained product
surface.

## Does FastInference require a C++ compiler?

No project C++ build step is required. Performance-critical maintained kernels
are Python/Triton. Use `uv sync` to install the pinned Python 3.12 environment.

## What is the supported configuration surface?

Use `FASTINFERENCE_CONFIG` to point at a TOML file. Runtime profiles resolve
into `RuntimeConfig`, which is then passed to the engine, scheduler, backend,
and model metadata. Deprecated `FASTINFERENCE_*` names may still exist for
compatibility and tools.

## Which models are supported?

Use [Capability Matrix](../CAPABILITY_MATRIX.md) and
[Supported Models](../models/supported_models.md). The maintained regression
targets are TinyLlama-1.1B, Qwen3.5-9B-AWQ, Gemma4-26B-A4B-it-AWQ-4bit, and
Gemma4-31B-it-AWQ-4bit.

## Can I rely on upstream vLLM docs?

No. Upstream docs are useful background, but this repository has a narrower
runtime, model, deployment, and kernel surface. Use the docs in this repository
for current behavior.

## Must inference match Hugging Face logits exactly?

Not always. Product-quality checks and strict semantic/logit checks serve
different purposes. See [INFERENCE_ACCURACY.md](../INFERENCE_ACCURACY.md) for
the current tiers and commands.
