# FastInference (vLLM Lite)

`FastInference` (vLLM Lite) 是一个将 vLLM 核心代码物理精简至 **81,000 行** 的极致单卡推理引擎。它完全移除了分布式复杂性、C++ 依赖和冗余架构，专注于 **纯 Python + Triton** 的单 GPU 性能巅峰。

## 🚀 核心成就 (v2.0 Performance Milestones)
在 v2.0 版本中，我们针对 AMD AI Max (gfx1151) 进行了架构级深度优化，实现了全链路真实负载下的高性能与极致稳定性：

- **端到端性能实测 (AMD AI Max+395 60.75GB - 真实权重负载)**:
  | 模型 (Real Weights) | 配置 | 吞吐量 (Aggregate TPS) | 状态 |
  | :--- | :--- | :--- | :--- |
  | **TinyLlama-1.1B** | BS=8, 2048ctx | **185.45** | ✅ [STABLE] |
  | **DeepSeek-V2-Lite** | BS=8, 2048ctx | **128.73** | ✅ [REAL MLA/MoE] |
  | **GLM-4.7-Flash** | BS=8, 2048ctx | **78.87** | ✅ [FULL 47-LAYER] |
  | **Qwen3.5-9B (GGUF)** | BS=8, 2048ctx | **50.20** | ✅ [STABLE] |
  | **Qwen3.5-35B-MoE** | BS=1, 1024ctx | **31.11** | ✅ [HIGH MEM ADAPT] |

- **硬件稳定性突破 (AMD Hardware Guard)**:
  - **Metadata Expansion**: 针对并行预填充阶段实现了元数据展开，确保 PagedAttention 核函数在处理 Chunked Prefill 时具备精准的物理偏移索引。
  - **Block-based KV Cache**: 将显存布局重构为固定的 **16-token blocks**，彻底解决了 AMD GPU 在处理超长 contiguous 序列时触发的 `hipErrorIllegalAddress` (Illegal Memory Access) 崩溃。
  - **Aggressive VRAM Policy**: 支持最高 **92%** 的显存预分配策略，并默认启用 **FP8 (E4M3) KV Cache**，使 35B 以上模型能在 60GB 显存内顺畅运行。

- **架构级创新**:
  - **`Self-Healing ModelLoader`**: 具备自动架构检测功能，能正确识别并同步 DeepSeek/GLM 等非 Llama 架构的层数、专家数与维度信息。
  - **`GGUF-to-FP8 Dequant`**: 权重加载时即时转存为最优计算格式，大幅降低中间显存峰值。

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
