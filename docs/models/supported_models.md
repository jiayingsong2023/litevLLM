# Supported Models (FastInference)

FastInference (vLLM Lite) is optimized for **Llama-like architectures** and specialized for high-performance single-GPU inference on AMD/NVIDIA.

## 🚀 Optimized Architecures

The following model architectures use our 100% Triton core and high-speed `LiteBase`:

| Architecture | Model Families | Supported Formats | Status |
| :--- | :--- | :--- | :--- |
| **Llama** | Llama 2, Llama 3, Llama 3.1, Llama 3.2 | FP16, BF16, GGUF, AWQ, FP8 | ✅ **Golden Standard** |
| **Qwen3.5** | Qwen3.5-9B, Qwen3.5-Coder | AWQ, GGUF | ✅ **Hybrid Attn** |
| **DeepSeek-V2** | DeepSeek-V2-Lite, Chat | GGUF | ✅ **MLA + MoE** |
| **GLM-4.7** | GLM-4.7-Flash | GGUF | ✅ **Verified** |
| **Qwen2** | Qwen2, Qwen2.5, Qwen2.5-Coder | FP16, BF16, GGUF, AWQ | ✅ Supported |
| **Mixtral** | Mixtral 8x7B, 8x22B | GGUF, FP16 (Via Fused MoE) | ✅ Supported |
| **TinyLlama** | TinyLlama-1.1B | FP16, BF16 | ✅ **1.0000 CosSim** |

## 👁 Multi-modal Models

We provide a streamlined framework for vision-language models:

| Model | Capability | Status |
| :--- | :--- | :--- |
| **Qwen2-VL** | Image Understanding | ✅ **Optimized (532 TPS)** |
| **DeepSeek-VL2** | Image Understanding | ✅ Supported |
| **InternVL** | Image Understanding | 🟡 In-Progress |

## 🛠 Quantization Support

FastInference specializes in **dynamic dequantization**:

- **GGUF**: Best-in-class performance via **LRU Weight Caching**. Supports **3D Expert Tensors**.
- **AWQ**: Optimized Triton kernels for 4-bit weights.
- **FP8**: Supported for both weights and KV cache.

## ❌ Unsupported Features (Removed for Lite)

To maintain a LOC < 100k, we have removed support for:
- Distributed-only models (e.g. models requiring tensor parallelism > 1).
- TPU/XPU specific kernels.
- Legacy architectures (OPT, Bloom, GPT-2).
