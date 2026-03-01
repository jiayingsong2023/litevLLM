# GGUF Quantization in FastInference

FastInference (vLLM Lite) provides industry-leading performance for GGUF models on single GPUs through a specialized **Dequantize-Cache-MatMul** strategy.

## 🚀 Key Optimizations

### 1. Triton Dequantization Kernel
Unlike standard vLLM which may rely on unoptimized fallbacks, FastInference utilizes a custom **OpenAI Triton kernel** specifically written for GGUF formats (e.g., Q4_0, Q8_0). 
- **Bit-level Unpacking**: Handles 4-bit and 8-bit unpacking directly in the GPU registers.
- **Hardware Alignment**: Optimized for memory coalescing on both NVIDIA and AMD architectures.

### 2. Global LRU Weight Caching
To eliminate the massive overhead of dequantizing weights at every inference step, we implemented a **Global LRU Cache** in `LiteLinear`:
- **First-Pass Cache**: Weights are dequantized into FP16 only once and stored in a managed cache.
- **Subsequent Speed**: Future requests using the same layer directly perform FP16 GEMMs, achieving near-native performance.
- **Configurable Capacity**: Default cache size is 128 layers, covering most 7B-13B models.

## 📊 Performance (Llama-2-7B GGUF)
- **Batch Size 1**: ~7 tokens/sec.
- **Batch Size 32**: **195.7 tokens/sec** (Enabled by LRU Caching).

## 🛠 Usage
GGUF models are automatically detected and optimized. Ensure your model file has the `.gguf` extension.

```python
from vllm import LLM
# FastInference handles the rest automatically
llm = LLM(model="llama-2-7b-chat.Q4_K_M.gguf")
```
