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

## Performance Milestones (AMD Strix Point)
*   **TinyLlama LoRA**: **546.4 tokens/sec** (Batch 32).
*   **Qwen-MoE**: **540.9 tokens/sec** (Batch 32).
*   **Multimodal (576 vision context)**: **532.4 tokens/sec** (Batch 32).
*   **Llama-7B GGUF**: 195.7 tokens/sec (Batch 32).

## Development Guidelines
*   **Adding Models**: Subclass `LiteModel` and `LiteDecoderLayer`. Use `LiteLinear` for all projections.
*   **No C++**: Never add code that requires a C++ compiler.
*   **Multi-modal**: Use `MultiModalInputProcessor` for any new modality support.
*   **No Distributed**: The engine is strictly single-GPU. `vllm/distributed` contains minimal shims only.
