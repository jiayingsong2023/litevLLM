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
*   **`vllm/model_executor/model_loader/`**: Self-healing loader with suffix matching for GGUF/AWQ.
*   **`vllm/kernels/triton/`**: Contains optimized kernels for PagedAttention, GGUF Dequant, Activation, and Index-aware MoE.

## Major Milestones (March 2026)
*   **DeepSeek-V2 / GLM-4.7-Flash**: ✅ Fully implemented MLA (Multi-Head Latent Attention) and MoE (Mixture of Experts) with 64 dynamic experts.
*   **Qwen3.5-9B**: ✅ Implemented Hybrid Attention Routing (Linear Attention + Full Attention).
*   **Semantic Audit**: ✅ Established `tests/verify_semantic_integrity.py` for 1:1 CosSim alignment with HuggingFace.

## Performance Milestones (AMD Radeon 8060S 65GB - Real Weights)
*   **TinyLlama-1.1B (Dense)**: **542.4 tokens/sec** (Batch 32, FP16 - 🟢 1.0000 CosSim).
*   **Qwen3.5-9B (AWQ)**: **205.1 tokens/sec** (Batch 32, 4-bit - 🟢 Stable).
*   **DeepSeek-V2-Lite (MoE)**: **112.7 tokens/sec** (Batch 16, GGUF - 🟢 Verified).
*   **GLM-4.7-Flash (MoE)**: **110.5 tokens/sec** (Batch 16, GGUF - 🟢 Verified).
*   **Qwen3.5-35B (MoE)**: **9.3 tokens/sec** (Batch 1, GGUF - 🟢 Stable).

## Development Guidelines
*   **Adding Models**: Subclass `nn.Module`. Use `LiteLinear` for all projections. Standardize on `model.layers.i` naming.
*   **No C++**: Never add code that requires a C++ compiler.
*   **Validation**: Always run `tests/verify_semantic_integrity.py` after architectural changes.
*   **No Distributed**: The engine is strictly single-GPU. `vllm/distributed` contains minimal shims only.

## Environment & Script Workflow (uv-first)
To keep dependency resolution and execution consistent across contributors, this project uses **uv** as the default toolchain.

### 1) Install and sync dependencies
```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create/update virtual environment from lockfile/pyproject
uv sync
```

### 2) Run Python scripts
Always run scripts through `uv run` so they execute in the managed environment:
```bash
uv run python tests/verify_semantic_integrity.py --model models/TinyLlama-1.1B-Chat-v1.0
uv run python tests/full_perf_regression.py
uv run python -m vllm.entrypoints.openai.api_server --model models/TinyLlama-1.1B-Chat-v1.0
```
