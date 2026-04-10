# LitevLLM (FastInference) Context

## Project Overview
**LitevLLM** is a high-performance, lightweight inference engine derived from vLLM. It is optimized for single-GPU deployments (NVIDIA & AMD) using **pure Python and OpenAI Triton**. The engine is designed to maximize performance on high-end consumer GPUs (32/48GB VRAM) while maintaining strict semantic accuracy.

**Key Philosophy:**
*   **Pure Python & Triton:** Zero C++ or CUDA C dependencies. All kernels are Triton-native.
*   **Unified Configuration:** Centralized control via `LiteInferenceConfig`, eliminating scattered environment variables.
*   **Path Convergence:** Exclusively focuses on **Safetensors + AWQ + TurboQuant** (GGUF support has been completely removed).
*   **High-Throughput Focus:** Optimized for batch sizes 8/16 and ultra-long context (up to 128k) using parallel sampling and aggressive KV pool strategies.

## Key Files & Directories
*   **`vllm/engine/inference_config.py`**: **[NEW]** Single Source of Truth for all inference settings.
*   **`vllm/engine/lite_engine.py`**: High-performance core with **Aggressive KV Allocation** and **Parallel Greedy Sampling**.
*   **`vllm/model_executor/layers/lite_linear.py`**: Focused on **Fused AWQ** Triton paths (Legacy LRU Caching removed).
*   **`vllm/kernels/triton/`**: Contains optimized kernels for **TurboQuant INT4**, FP8 KV, and Fused AWQ.

## Major Milestones (March 2026 - Post-Simplification)
*   **Phase 1 & 2 Success**: ✅ Consolidated 10+ environment variables into `LiteInferenceConfig`. ✅ Completely stripped GGUF loader and kernels.
*   **TurboQuant INT4 KV Cache**: ✅ Fixed accumulation logic and quantization offsets, achieving stable inference even on 1.1B models.
*   **Parallel Sampling**: ✅ Implemented vectorized greedy sampling, removing Python-loop bottlenecks for high concurrency.
*   **Semantic Audit**: ✅ 1:1 CosSim alignment with HuggingFace for BF16/FP8 paths (Tier-A verified).

## Performance Milestones (AMD Radeon 8060S 65GB)
*   **High-End GPU Mode**: Automatically activates on 24GB+ cards, bumping `max_active_requests` to 16 and doubling prefill chunk size.
*   **TinyLlama-1.1B**: **542+ tokens/sec** (Aggregate throughput, Batch 16+).
*   **Qwen3.5-9B (AWQ)**: **200+ tokens/sec** (Aggregate throughput, Batch 16, FP8 KV).

## Development Guidelines
*   **No C++**: Never add code that requires a C++ compiler.
*   **Config First**: Access all runtime settings via `attn_metadata["config"]`. Never use `os.environ` in model layers.
*   **Validation**: Always run `tests/run_inference_correctness_regression.sh` after changes.
*   **Model Implementation**: Use `LiteLinear` and `expand_metadata_for_paged_attention` (from `lite_engine`) for all new model architectures.

## Environment & Script Workflow (uv-first)
Always use `uv run` for consistent dependency resolution.
```bash
# Sync environment
uv sync

# Run accuracy regression
uv run bash tests/run_inference_correctness_regression.sh
```
