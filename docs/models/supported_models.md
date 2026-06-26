# Supported Models (FastInference)

The source of truth for model and feature status is
[FastInference Capability Matrix](../CAPABILITY_MATRIX.md). This page is a
short model-focused view of that matrix.

## Supported Regression Targets

| Model / Family | Formats | Status | Gate |
| :--- | :--- | :--- | :--- |
| TinyLlama-1.1B | FP16/BF16 | Supported | Tier-B quality spotcheck and A-strict semantic integrity. |
| Qwen3.5-9B-AWQ | AWQ | Supported | Tier-B quality spotcheck and A-strict AWQ-vs-FP16 audit. |
| Gemma4-26B-A4B-it-AWQ-4bit | AWQ 4-bit | Supported | Tier-B, A-lite, and default A-strict audit unless locally disabled. |
| Gemma4-31B-it-AWQ-4bit | AWQ 4-bit | Supported | Tier-B and A-lite; A-strict remains manual/specialized. |

## Experimental / Compatibility Surface

| Area | Status | Notes |
| :--- | :--- | :--- |
| Llama-like fallback models | Experimental | Adapter fallback exists, but support should not be claimed without model-specific load, smoke, and correctness gates. |
| DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf | Experimental | Native DS4 GGUF target. Batch=1 greedy GPU direct path, LiteEngine/AsyncLLM routing, OpenAI REST smoke, compressed-paged KV contracts, automatic Tier-B smoke when the local GGUF exists, and e2e direct benchmark coverage are present. Default validation uses 4K context; 8K is a code cap, not a default regression promise. |
| LoRA runtime | Experimental | Runtime and tests exist; production policy tuning remains workload dependent. |
| Multimodal serving | Experimental | Lite serving paths exist, but need broader real-traffic hardening. |
| Legacy vLLM entrypoints | Compatibility | Kept for import and migration stability, not as a second runtime path. |

## Unsupported In Lite

- Full upstream vLLM model compatibility.
- Multi-GPU distributed runtime as an official execution path.
- TPU/XPU-specific kernels.
- C++/CUDA source extensions outside allowed third-party code.
- Speculative decoding.
