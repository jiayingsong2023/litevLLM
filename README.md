# FastInference (vLLM Lite)

`FastInference` (vLLM Lite) 是一个将 vLLM 核心代码物理精简至 **81,000 行** 的极致单卡推理引擎。它完全移除了分布式复杂性、C++ 依赖和冗余架构，专注于 **纯 Python + Triton** 的单 GPU 性能巅峰。

## 🚀 核心成就 (v2.0 Performance Milestones)
在 v2.0 版本中，我们通过重构核心调度器与显存管理，实现了性能与稳定性的双重突破：

- **极致吞吐量 (AMD Strix Point 60GB 真实权重 + BS=32 实测)**:
  - **Qwen3.5-35B-MoE**: 🚀 **113.80 tokens/sec** (FP8 Expert Cache + Continuous Batching, 较 v1.0 提升 **32倍**).
  - **TinyLlama-1.1B (Dense)**: **536.22 tokens/sec** (Stable at 4096 Context).
  - **Qwen3.5-9B (GGUF)**: **155.15 tokens/sec** (含动态维度对齐保护).
  - **DeepSeek-V2-Lite**: **82.87 tokens/sec** (MLA 混合路径 + FP8 专家).

- **长文本能力 (4096 Context 深度加固)**:
  - **稳定性突破**: 通过 **分块异步写入** 与 **显式 CUDA 同步锁**，彻底解决了 AMD GPU 在长序列下的 `hipErrorIllegalAddress` 崩溃问题。
  - **Chunked Prefill**: 实现了长 Prompt 的自动切片处理，大幅降低多并发下的 TTFT（首字延迟）。

- **架构级创新**:
  - **`True Continuous Batching`**: 重写了 `LiteEngine` 调度器，支持 Prefill 与 Decode 的高效并发流水线。
  - **`Default FP8 KV Cache`**: 全局默认开启 FP8 量化缓存，显存利用率提升 100%。
  - **`GGUF-to-FP8 Dequant`**: 权重反量化直转 FP8，规避了中间 FP16 张量导致的显存峰值溢出。
  - **`Self-Healing ModelLoader`**: 具备自动属性映射与维度校准功能，完美适配非标准导出的 GGUF 权重。

## 🌟 核心特性
- **纯净计算图**: 100% Triton 化的核心算子，完全剥离 C++ 编译依赖。
- **混合加速路径**: Prefill 阶段利用硬件最强 SDPA 内核，Decode 阶段全量回归项目手写高性能 **Triton PagedAttention**。
- **激进显存 Offloading**: 自动识别巨型 Embedding 层并隔离至 CPU，为 4096+ 长上下文腾挪空间。

## 🚀 快速开始

### 0. 环境准备（uv）
```bash
# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步项目依赖并创建虚拟环境
uv sync
```

### 运行端到端回归测试
```bash
# 执行性能回归与正确性校验
uv run python tests/perf_regression.py

# 执行隔离进程下的压力测试
uv run python tests/run_isolated_benchmarks.py
```

## 🛠 当前算子与功能状态
| 类别 | 状态 | 备注 |
| :--- | :--- | :--- |
| **Continuous Batching** | ✅ **Active** | 支持多并发流式推理 |
| **Chunked Prefill** | ✅ **Enabled** | 解决长 Prompt 阻塞问题 |
| **FP8 KV Cache** | ✅ **Default** | 节约 50% 显存占用 |
| **Triton PagedAttn** | ✅ **Optimized** | Decode 阶段核心算子 |
| **Model Self-Healing** | ✅ **Integrated** | 自动修复 GGUF 维度与命名冲突 |

## 📄 架构深度解析
请参考 [docs/ARCHITECTURE_LITE.md](./docs/ARCHITECTURE_LITE.md)。

## 📘 API 文档
请参考 [docs/API_REFERENCE.md](./docs/API_REFERENCE.md)。

## 🛠️ 使用指南 (Usage)
### 1. 离线批处理推理 (Offline Batch Inference)
```python
from vllm import LLM, SamplingParams
# 支持 4096 上下文与 FP8 自动开启
llm = LLM(model="models/Qwen3.5-9B-GGUF")
prompts = ["Explain quantum computing in simple terms."]
sampling_params = SamplingParams(max_tokens=128)
outputs = llm.generate(prompts, sampling_params)
```

### 2. 在线服务 (OpenAI API Server)
```bash
# 启动具备 v2.0 性能特性的 API 服务器
uv run python -m vllm.entrypoints.openai.api_server --model models/Qwen3.5-9B-GGUF
```

## 🛡️ 显存优化提示
若在 60GB 以下显存遇到 OOM，可尝试：
```bash
# 强制开启专家权重 CPU Offload
FASTINFERENCE_DEEPSEEK_AGGRESSIVE=1 \
uv run python -m vllm.entrypoints.openai.api_server --model ...
```
