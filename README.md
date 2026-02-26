# FastInference (vLLM Lite)

`FastInference` (vLLM Lite) 是一个将 vLLM 核心代码从 **270,000 行物理精简至 81,000 行** 的极致单卡推理引擎。它完全移除了分布式复杂性、C++ 依赖和冗余架构，专注于 **纯 Python + Triton** 的单 GPU 性能巅峰。

## 🚀 核心成就 (LOC < 100k)
- **代码裁减 70%**: 移除了所有分布式 (`distributed`)、多后端、投机采样及非核心模型支持。
- **Lite 架构重组**: 
  - **`LiteLinear`**: 统一线性层，移除 TP/PP 逻辑，内置 **GGUF 权重缓存 (Caching)**。
  - **`LiteModel`**: 极简模型拓扑，Llama/Qwen 等核心模型代码量减少 90%。
  - **`Triton Only`**: 核心 Attention 与 MoE 算子全 Triton 化，无二进制编译依赖。
- **实测性能 (RTX 4090)**:
  - **Dense (TinyLlama)**: ~25 tokens/sec (Eager Mode).
  - **MoE (Qwen-MoE)**: ~32 tokens/sec.
  - **GGUF (Llama-7B)**: ~174 tokens/sec (Batch Size 32 + Caching).

## 🌟 核心理念
- **极致精简**: 物理删除 19 万行非核心代码，确保每一行代码都可读、可控。
- **单卡巅峰**: 移除多卡同步开销，利用 Triton 压榨消费级显卡性能。
- **零编译依赖**: 无需 `nvcc`，直接运行，支持 NVIDIA/AMD/Mac。

## 🚀 快速开始

### 安装
```bash
# 无需 C++ 编译器，直接安装 Python 依赖
uv pip install -e .
```

### 运行基准测试
```bash
# 端到端 Llama 性能测试
uv run python tests/e2e_perf_benchmark.py

# 优化的 GGUF 批量性能测试
uv run python tests/e2e_gguf_perf.py
```

## 🛠 当前支持模型
- **Llama 家族** (Llama 2/3, TinyLlama, Mistral, Yi)
- **Qwen 家族** (Qwen2, Qwen1.5-MoE)
- **DeepSeek 家族** (DeepSeek-V2/V3 Lite 版)
- **Kimi** (KDA/MLA 线性注意力支持)

## 📄 架构深度解析
请参考 [docs/ARCHITECTURE_LITE.md](./docs/ARCHITECTURE_LITE.md)。