# LitevLLM (FastInference) Context

## Project Overview
**LitevLLM** is a high-performance, lightweight inference engine derived from vLLM. It is designed to run Large Language Models (LLMs) on single GPUs (NVIDIA & AMD) using **pure Python and OpenAI Triton**, completely removing C++ and CUDA C dependencies.

**Key Philosophy:**
*   **Python & Triton Only:** No compiled C++ extensions (`csrc` removed).
*   **Modular Architecture**: Standardized on `LiteBase` to eliminate code redundancy.
*   **Stability-First**: Uses a hybrid approach (Triton for compute, PyTorch for sensitive IO) to support **Batch Size 32** on diverse hardware (including AMD APUs).

## Key Files & Directories

*   **`vllm/model_executor/models/lite_base.py`**: Contains `LiteModel` and `LiteDecoderLayer`. Most Llama-like models inherit from these.
*   **`vllm/model_executor/layers/lite_linear.py`**: Unified Linear layer with **Global LRU Caching** for dequantized weights.
*   **`vllm/engine/lite_engine.py`**: Core orchestration logic for the single-GPU inference loop, integrated with `Scheduler`.
*   **`vllm/attention/backends/triton_attn.py`**: Production attention backend supporting PagedAttention and Fused Prefill.
*   **`vllm/kernels/triton/`**: Contains optimized kernels for PagedAttention, GGUF Dequant, Activation, and Index-aware MoE.

## Performance Milestones (Stable Mode)
*   **TinyLlama**: 27.4 tokens/sec.
*   **Llama-7B GGUF**: 195.7 tokens/sec (Batch 32).
*   **Qwen-MoE**: **533.2 tokens/sec** (Batch 32).

## Development Guidelines
*   **Adding Models**: Subclass `LiteModel` and `LiteDecoderLayer`.
*   **No C++**: Never add code that requires a C++ compiler. Use Triton or stable PyTorch paths.
*   **Memory Safety**: Always use masked loads/stores in Triton kernels and ensure `Index_Map` alignment for large batch sizes.
*   **No Distributed**: The engine is strictly single-GPU. Shims in `vllm/distributed` are for compatibility only.
