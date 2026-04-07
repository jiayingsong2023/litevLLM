# Triton Migration Strategy (Post-Simplification)

LitevLLM has successfully migrated all core kernels to pure OpenAI Triton. This document tracks the status of these kernels and their performance impact.

## Core Kernels Status
| Kernel | Status | Description | Alignment |
| :--- | :--- | :--- | :--- |
| **PagedAttention** | ✅ Stable | Standard paged decoding with GQA support | 1:1 CosSim |
| **Fused AWQ** | ✅ Active | Optimized INT4 GEMM + Dequant fusion | Tier-A Verified |
| **TurboQuant** | ✅ Fixed | Symmetric INT4 with Dynamic Scaling | Crucial for long-context stability |

- **Parallel Greedy Sampling**: Vectorized decoding removes Python-loop bottlenecks, ensuring **542+ TPS** on TinyLlama.

### Migration Status
| Model | Type | Native Throughput (Baseline) |
| :--- | :--- | :--- |
| **TinyLlama-1.1B** | Safetensors | **542.4 tokens/sec (Batch 32)** |
| **Qwen3.5-9B (AWQ)** | Safetensors | **205.1 tokens/sec (Batch 16)** |
| **Qwen3.5-35B (AWQ)** | Safetensors | **~40 tokens/sec (Batch 8)** |

---
*Note: GGUF support and related legacy kernels have been removed to focus on Safetensors/AWQ performance.*
