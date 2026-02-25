# LitevLLM (FastInference) Context

## Project Overview
**LitevLLM** is a high-performance, lightweight inference engine derived from vLLM. It is designed to run Large Language Models (LLMs) on single GPUs (NVIDIA & AMD) using **pure Python and OpenAI Triton**, completely removing C++ and CUDA C dependencies. This makes it highly portable and easier to modify.

**Key Philosophy:**
*   **Python & Triton Only:** No compiled C++ extensions (`csrc` removed).
*   **Modular Architecture:** Standardized on `LiteBase` to eliminate code redundancy across models.
*   **Single-GPU Optimization:** Streamlined for local inference without distributed system overhead.

## Key Files & Directories

*   **`vllm/model_executor/models/lite_base.py`**: **NEW.** Contains generic `LiteModel`, `LiteForCausalLM`, and `LiteDecoderLayer`. Most Llama-like models (Llama, Qwen2, Mistral) now inherit directly from these classes.
*   **`vllm/model_executor/layers/lite_linear.py`**: A unified Linear layer wrapper. Handles quantized/unquantized weights and replaces standard parallel linear layers.
*   **`vllm/engine/lite_engine.py`**: Core orchestration logic for the single-GPU inference loop.
*   **`vllm/distributed/`**: Compatibility shims (`parallel_state.py`, `utils.py`) for single-device execution.
*   **`vllm/attention/backends/triton_attn.py`**: The sole production attention backend (FlashAttention implementation in Triton).
*   **`vllm/config/`**: Refactored to remove distributed (`parallel_config.py` simplified), speculative, and profiler configurations. `VllmConfig` is streamlined.

## Usage & Development

### 1. Installation
```bash
uv pip install -e .
```

### 2. Running Benchmarks
*   **Standard Models:**
    ```bash
    uv run python -m vllm.entrypoints.cli.main bench latency \
      --model models/TinyLlama-1.1B-Chat-v1.0
    ```
*   **MoE Models:**
    ```bash
    uv run python -m vllm.entrypoints.cli.main bench latency \
      --model models/Qwen1.5-MoE-A2.7B-Chat --enforce-eager
    ```

### 3. Development Guidelines
*   **Adding Models:** Check `llama.py` for examples. Usually, you just need to alias `LiteModel` and `LiteForCausalLM` or subclass `LiteDecoderLayer` for custom logic (like `GemmaLiteModel`).
*   **Circular Imports:** Avoid importing from `vllm.model_executor.model_loader.weight_utils` at the top level of layer files; import inside methods if needed.
*   **No C++:** Never add code that requires a C++ compiler. Use Triton for kernels.
*   **No Distributed:** Do not re-introduce distributed logic or dependencies (e.g. `torch.distributed`). The engine is strict single-GPU.