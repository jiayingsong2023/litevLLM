# LitevLLM (FastInference) Context

## Project Overview
**LitevLLM** is a high-performance, lightweight inference engine derived from vLLM. It is designed to run Large Language Models (LLMs) on single GPUs (NVIDIA & AMD) using **pure Python and OpenAI Triton**, completely removing C++ and CUDA C dependencies.

**Key Philosophy:**
*   **Python & Triton Only:** No compiled C++ extensions (`csrc` removed).
*   **Modular Architecture**: Standardized on `LiteBase` and a flattened engine structure.
*   **Stability-First**: Uses a hybrid approach (Triton for compute, PyTorch for sensitive IO) to support **Batch Size 32** on diverse hardware (including AMD APUs).

## Key Files & Directories

*   **`vllm/model_executor/layers/lite_linear.py`**: Unified Linear layer with **Global LRU Caching** and **LiteLoRA** support.
*   **`vllm/engine/async_llm.py`**: Flattened entrypoint for async inference, wrapper around `LiteEngine`.
*   **`vllm/multimodal/`**: 100% functional framework for real-image preprocessing and multi-modal tensor conversion.
*   **`vllm/structured_output/`**: Production-grade implementation using Outlines for constrained generation.
*   **`vllm/kernels/triton/`**: Contains optimized kernels for PagedAttention, GGUF Dequant, Activation, and Index-aware MoE.

## Ongoing Adaptations (March 2026)
*   **GLM-4.7-Flash**: Implementing `glm4_moe_lite` architecture with MLA and Shared Experts.
*   **Kimi-Linear**: Adapting for linear attention and long-context (128K+) performance.
*   **MiniMax-abab7**: Evaluating MoE routing logic for single-GPU optimization.

## Performance Milestones (AMD AI Max 60GB - Real Weights)
*   **DeepSeek-V2-Lite (16B MoE)**: **905.4 tokens/sec** (Batch 32, GGUF - 🔥 Stable Peak).
*   **TinyLlama-1.1B (Dense)**: **590.6 tokens/sec** (Batch 32, FP16).
*   **Qwen3.5-9B (GGUF)**: **233.8 tokens/sec** (Batch 32, Q4_K).
*   **Qwen3.5-9B (AWQ 4-bit)**: **147.7 tokens/sec** (Batch 32, Safetensors Load).
*   **Qwen3.5-35B (MoE)**: **3.5 tokens/sec** (Batch 1, GGUF - 🟢 Stable).


## Development Guidelines
*   **Adding Models**: Subclass `LiteModel` and `LiteDecoderLayer`. Use `LiteLinear` for all projections.
*   **No C++**: Never add code that requires a C++ compiler.
*   **Multi-modal**: Use `MultiModalInputProcessor` for any new modality support.
*   **No Distributed**: The engine is strictly single-GPU. `vllm/distributed` contains minimal shims only.
