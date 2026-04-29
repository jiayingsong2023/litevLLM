# Triton Migration Strategy (Post-Simplification)

LitevLLM has successfully migrated all core kernels to pure OpenAI Triton. This document tracks the status of these kernels and their performance impact.

## Core Kernels Status
| Kernel | Status | Description | Alignment |
| :--- | :--- | :--- | :--- |
| **PagedAttention** | ✅ Stable | Standard paged decoding with GQA support, Gemma4 softcap native | 1:1 CosSim |
| **Fused AWQ GEMM** | ✅ Active | Optimized INT4 GEMM + Dequant fusion with M=1 GEMV specialization | Tier-A Verified |
| **TurboQuant INT4 KV** | ✅ Fixed | Symmetric INT4 with Dynamic Scaling | Crucial for long-context stability |
| **Fused Gate-Up** | ✅ Active | MLP gate projection + up projection + activation fused in single kernel | Tier-A Verified |
| **Fused QKV** | ✅ Active | Q/K/V projection fused for decode, sharing activation loads | Tier-A Verified |
| **Reshape & Cache** | ✅ Stable | KV cache write with fp8/int4 quantization, CPU-side scale caching | 1:1 CosSim |

## Migration Status
| Model | Type | Native Throughput (Baseline) |
| :--- | :--- | :--- |
| **TinyLlama-1.1B** | Safetensors BF16 | **542.4 tokens/sec (Batch 32)** |
| **Qwen3.5-9B (AWQ)** | Safetensors AWQ INT4 | **205.1 tokens/sec (Batch 16)** |
| **Gemma4-26B-A4B (AWQ)** | Safetensors compressed-tensors INT4 MoE | **2.31 tokens/sec (Batch 1)** |
| **Gemma4-31B-it (AWQ)** | Safetensors compressed-tensors INT4 Dense | **0.90 tokens/sec (Batch 1)** |

---
*GGUF support and related legacy kernels have been removed to focus on Safetensors/AWQ performance.*
*The `moe_gemm.py`, `fused_attention.py`, `lite_rmsnorm.py`, and `rotary_embedding.py` Triton stubs have been cleaned up as dead code (2026-04).*
