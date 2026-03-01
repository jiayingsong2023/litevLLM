# FastInference (vLLM Lite)

`FastInference` (vLLM Lite) 是一个将 vLLM 核心代码从 **270,000 行物理精简至 81,000 行** 的极致单卡推理引擎。它完全移除了分布式复杂性、C++ 依赖和冗余架构，专注于 **纯 Python + Triton** 的单 GPU 性能巅峰。

## 🚀 核心成就 (LOC < 100k)
- **代码裁减 70%**: 移除了所有分布式 (`distributed`)、多后端、投机采样及非核心模块。
- **极致吞吐量 (AMD Strix Point 实测)**:
  - **MoE (Qwen-MoE-2.7B)**: **533 tokens/sec** (Batch Size 32, Index-aware GEMM).
  - **GGUF (Llama-7B)**: **195 tokens/sec** (Batch Size 32, LRU Weight Caching).
  - **Dense (TinyLlama)**: **27+ tokens/sec** (Batch Size 1, FP16).
- **架构代差级优化**:
  - **`LiteLinear`**: 内置 **Global LRU 权重缓存**，首次运行自动反量化并缓存。
  - **`Index-aware MoE`**: 彻底消除了专家调度中的数据重排 (Permute) 开销。
  - **`Quant-Aware Fused Prefill`**: 融合了 KV 写入、FP8 量化与 FlashAttention 计算。

## 🌟 核心理念
- **极致精简**: 物理删除 19 万行冗余代码，核心推理路径实现 100% 可读。
- **混合动力引擎**: 在计算密集型环节（GEMM/Dequant）使用全速 Triton，在内存/架构敏感环节使用 PyTorch 稳定版，确保 **Batch Size 32** 无非法内存访问。
- **零编译依赖**: 移除 `csrc`，无需 `nvcc`，支持 NVIDIA/AMD 一键运行。

## 🚀 快速开始

### 安装
```bash
# 无需 C++ 编译器，直接安装 Python 依赖
uv pip install -e .
```

### 运行基准测试
```bash
# 1. 密集模型性能测试
uv run python tests/e2e_perf_benchmark.py

# 2. GGUF 优化性能测试
uv run python tests/e2e_gguf_perf.py

# 3. MoE 吞吐量 Scaling 测试 (重点推荐)
uv run python tests/e2e_moe_batch_scaling.py
```

## 🛠 当前算子状态
| 算子类别 | 状态 | 备注 |
| :--- | :--- | :--- |
| **Linear / Dequant** | ✅ **Triton + Cache** | 支持 GGUF, LRU 缓存加速 |
| **Attention** | ✅ **Triton / Stable** | Paged 寻址，Fused Prefill 支持 |
| **MoE Routing** | ✅ **Index-aware** | 零拷贝专家调度 |
| **Norm / RoPE** | ✅ **Stable Path** | 确保 AMD APU 大 Batch 稳定性 |
| **Activation** | ✅ **Triton Fused** | Silu, Gelu, SiluAndMul |

## 📄 架构深度解析
请参考 [docs/ARCHITECTURE_LITE.md](./docs/ARCHITECTURE_LITE.md)。
