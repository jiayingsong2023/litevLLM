# Supported Models (FastInference)

FastInference (vLLM Lite) is optimized for **Llama-like architectures** and specialized for high-performance single-GPU inference.

## 🚀 Optimized Architecures

The following model architectures use our 100% Triton core and high-speed `LiteBase`:

| Architecture | Model Families | Supported Formats |
| :--- | :--- | :--- |
| **Llama** | Llama 2, Llama 3, Llama 3.1, Llama 3.2 | FP16, BF16, GGUF, AWQ, FP8 |
| **Qwen2** | Qwen2, Qwen2.5, Qwen2.5-Coder | FP16, BF16, GGUF, AWQ |
| **Mistral** | Mistral v0.1, v0.3 | FP16, BF16, GGUF |
| **Mixtral** | Mixtral 8x7B, 8x22B | GGUF, FP16 (Via Fused MoE) |
| **Qwen-MoE** | Qwen1.5-MoE-A2.7B | GGUF, FP16 (Index-aware GEMM) |
| **TinyLlama** | TinyLlama-1.1B | FP16, BF16 |

## 👁 Multi-modal Models

We provide a streamlined framework for vision-language models:

| Model | Capability | Status |
| :--- | :--- | :--- |
| **Qwen2-VL** | Image Understanding | ✅ **Optimized (532 TPS)** |
| **DeepSeek-VL2** | Image Understanding | ✅ Supported |
| **InternVL** | Image Understanding | 🟡 In-Progress |

## 🛠 Quantization Support

FastInference specializes in **dynamic dequantization**:

- **GGUF**: Best-in-class performance via **LRU Weight Caching**.
- **FP8**: Supported for both weights and KV cache.
- **AWQ**: Optimized Triton kernels for 4-bit weights.

## ❌ Unsupported Features (Removed for Lite)

To maintain a LOC < 100k, we have removed support for:
- TPU/XPU/OpenVINO specific architectures.
- Distributed-only models (e.g. models requiring tensor parallelism > 1).
- Legacy architectures (OPT, Bloom, GPT-2).
- Speculative decoding models (Medusa, Eagle stubs remain but logic is disabled).
