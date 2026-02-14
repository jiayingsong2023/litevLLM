# Optimization Report: GGUF Inference on AMD Strix Point (gfx1151)

## Executive Summary
We have successfully optimized `litevLLM` to run Llama-2-7B-Chat (GGUF Q4_K_M) on the AMD Ryzen AI 300 APU. The final configuration achieves **~38 tokens/sec** generation throughput with stable memory usage, resolving previous issues of "Memory Inflation" and "Illegal Memory Access".

## Key Achievements

1.  **Fixed HIP Illegal Memory Access**
    -   **Problem**: The original vectorized Triton dequantization kernel caused crashes on the RDNA3.5 (gfx1151) architecture.
    -   **Solution**: Implemented a robust, simplified Triton kernel (`vllm/kernels/triton/gguf_dequant.py`) that respects memory boundaries and hardware limitations.

2.  **Solved Memory Inflation (OOM)**
    -   **Problem**: The original implementation dequantized weights endlessly without freeing them, causing RAM to fill up until crash.
    -   **Solution**: Implemented an **LRU (Least Recently Used) Cache** for dequantized weights in `vllm/model_executor/layers/quantization/gguf_kernels.py`.

3.  **Restored High Performance (~38 t/s)**
    -   **Problem**: Initial "Fused Kernel" attempts were slow (~1.6 t/s) due to lack of hardware-specific matrix core (MFMA) optimization in Triton for this specific APU. Small cache sizes (32) caused "cache thrashing", leading to constant re-dequantization.
    -   **Solution**:
        -   Reverted to the **"Dequantize + Cache + MatMul"** strategy.
        -   Optimized the dequantization kernel to be vectorized (efficient).
        -   Increased recommended cache size to hold the full model working set.

## Recommended Configuration

To run 7B models with maximum performance on this hardware (assuming 16GB+ RAM available to GPU):

```bash
# Set cache size to cover all model layers (~224 tensors for 7B)
# This prevents re-dequantization during generation.
export VLLM_GGUF_CACHE_SIZE=300

# Force GFX version for ROCm compatibility
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Run vLLM
python -m vllm.entrypoints.cli.main bench latency \
    --model llama-2-7b-chat.Q4_K_M.gguf \
    --quantization gguf \
    --dtype half \
    --enforce-eager
```

## Performance Data

| Configuration | Throughput (t/s) | Notes |
| :--- | :--- | :--- |
| **Baseline (Crash)** | N/A | Crashed due to illegal memory access or OOM |
| **Fused Kernel (Naive)** | ~1.6 | Functional but extremely slow (no pipelining) |
| **Cache=32 (Thrashing)** | ~1.6 | Constant eviction/re-dequantization |
| **Cache=128 (Partial)** | ~17.5 | Better, but still some thrashing |
| **Cache=300 (Full)** | **~38.7** | **Optimal. Zero re-dequantization during gen.** |

## Code Changes
- **`vllm/kernels/triton/gguf_dequant.py`**: New vectorized, stable dequantization kernel.
- **`vllm/model_executor/layers/quantization/gguf_kernels.py`**: Integrated LRU Cache logic and fallback to PyTorch `matmul`.
