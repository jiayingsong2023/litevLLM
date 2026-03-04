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

## Performance Milestones (AMD AI Max 60GB - Real Weights)
*   **DeepSeek-V2-Lite (16B MoE)**: **885.5 tokens/sec** (Batch 128, 4K Context).
*   **TinyLlama-1.1B (Dense)**: **668.7 tokens/sec** (Batch 32, FP16).
*   **Qwen3.5-9B (Dense)**: **148.3 tokens/sec** (Batch 32, Q4 GGUF).
*   **Qwen3.5-35B (MoE)**: **33.6 tokens/sec** (Batch 16, 4K Context).
*   **TinyLlama LoRA**: **578.0 tokens/sec** (Batch 32, Real Weights).

## Development Guidelines
*   **Adding Models**: Subclass `LiteModel` and `LiteDecoderLayer`. Use `LiteLinear` for all projections.
*   **No C++**: Never add code that requires a C++ compiler.
*   **Multi-modal**: Use `MultiModalInputProcessor` for any new modality support.
*   **No Distributed**: The engine is strictly single-GPU. `vllm/distributed` contains minimal shims only.
