# Supported Models (FastInference)

The source of truth for model and feature status is
[FastInference Capability Matrix](../CAPABILITY_MATRIX.md). This page is a
short model-focused view of that matrix.

## Supported Regression Targets

| Model / Family | Formats | Status | Gate |
| :--- | :--- | :--- | :--- |
| TinyLlama-1.1B | FP16/BF16 | Supported | Tier-B quality spotcheck and A-strict semantic integrity. |
| Qwen3.5-9B-AWQ | AWQ | Supported | Tier-B quality spotcheck and A-strict AWQ-vs-FP16 audit. |
| Gemma4-26B-A4B-it-AWQ-4bit | AWQ 4-bit | Supported | Tier-B, A-lite, default A-strict audit unless locally disabled, and image multimodal quality spotcheck. |
| Gemma4-31B-it-AWQ-4bit | AWQ 4-bit | Supported | Tier-B, A-lite, and image multimodal quality spotcheck; A-strict remains manual/specialized. |
| Gemma4 image multimodal | AWQ 4-bit | Supported | 26B/31B image quality gate, image placeholder expansion, official Gemma4 image patch preprocessing, Gemma4 vision tower, placeholder replacement, multi-image requests, multi-request batching, and Gemma4 projector LoRA. |

## Experimental / Compatibility Surface

| Area | Status | Notes |
| :--- | :--- | :--- |
| Llama-like fallback models | Experimental | Adapter fallback exists, but support should not be claimed without model-specific load, smoke, and correctness gates. |
| DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf | Experimental | Native DS4 GGUF target. Adapter-owned executors and compressed-paged KV contracts run through AsyncLLM, LiteEngine, RuntimeController, and StepScheduler. Automatic Tier-B smoke and e2e benchmark coverage are present when the local GGUF exists. Default validation uses 4K context; 8K is a code cap, not a default regression promise. The standalone GPU smoke does not emit standard per-token streaming observer events, so `stream_visible=0%` is expected there. |
| Qwen2VL image multimodal | Experimental | Image preprocessing, `image_grid_thw`, mRoPE positions, real vision tower, and placeholder replacement are implemented. Vision-tower LoRA is unsupported. |
| LoRA runtime | Experimental | Safetensors PEFT adapters, text-layer LoRA, mixed LoRA batches, and Gemma4 projector LoRA exist; production policy tuning remains workload dependent. |
| Multimodal serving | Experimental | Gemma4 image support is maintained. Qwen2VL image support exists but still needs broader real-checkpoint smoke coverage. |
| Gemma4 E4B | Experimental | The local model can load, but it is not part of the supported regression surface and did not pass the Gemma4 image multimodal quality gate. |
| Legacy vLLM entrypoints | Compatibility | Kept for import and migration stability, not as a second runtime path. |

## Unsupported In Lite

- Full upstream vLLM model compatibility.
- Multi-GPU distributed runtime as an official execution path.
- TPU/XPU-specific kernels.
- C++/CUDA source extensions outside allowed third-party code.
- Speculative decoding.
