# FastInference (vLLM Lite)

`FastInference` (vLLM Lite) 是一个将 vLLM 核心代码物理精简至 **81,000 行** 的极致单卡推理引擎。它完全移除了分布式复杂性、C++ 依赖和冗余架构，专注于 **纯 Python + Triton** 的单 GPU 性能巅峰。

## 🚀 核心成就 (v2.0 Performance Milestones)
在 v2.0 版本中，我们针对 AMD AI Max (gfx1151 / Strix Point) 进行了架构级深度优化，实现了全链路真实负载下的高性能与极致稳定性：

- **端到端性能实测 (AMD Radeon 8060S 65GB - 真实权重负载)**:
  | 模型 (Real Weights) | 配置 | 吞吐量 (Aggregate TPS) | 状态 |
  | :--- | :--- | :--- | :--- |
  | **TinyLlama-1.1B** | BS=32, 2048ctx | **542.4** | ✅ [1.0000 CosSim] |
  | **Qwen3.5-9B (AWQ)** | BS=32, 2048ctx | **205.1** | ✅ [Hybrid Architecture] |
  | **DeepSeek-V2-Lite** | BS=16, 2048ctx | **112.7** | ✅ [MLA/MoE Verified] |
  | **GLM-4.7-Flash** | BS=16, 2048ctx | **110.5** | ✅ [Full MoE Path] |
  | **Qwen3.5-35B-MoE** | BS=1, 1024ctx | **9.3** | ✅ [High Memory Adapt] |

- **硬件稳定性突破 (AMD Hardware Guard)**:
  - **Metadata Expansion**: 针对并行预填充阶段实现了元数据展开，确保 PagedAttention 核函数在处理 Chunked Prefill 时具备精准的物理偏移索引。
  - **ROCm 7.2 Alignment**: 完全适配 ROCm 7.2 驱动，解决了 `torch.zeros` 分配导致的高频段段错误 (Segfault 139)。
  - **Unified Buffer Layout**: 统一 KV Cache 步长与内核读取逻辑，彻底消除了非对称维度下的 `hipErrorIllegalAddress` 崩溃。

- **架构级创新**:
  - **`Self-Healing ModelLoader`**: 具备深层后缀匹配功能，能自动识别并映射 DeepSeek/GLM 等非标准架构的 LoRA 投影、3D 专家张量及融合 QKV 权重。
  - **`Hybrid Attention Routing`**: 支持在一个模型中混合使用 `Linear Attention` (递归形式) 与 `Full Attention` (标准 Transformer)，完美适配 Qwen3.5 架构。

## 🌟 核心特性
- **纯净计算图**: 100% Triton 化的核心算子，完全剥离 C++ 编译依赖。
- **混合加速路径**: Prefill 阶段利用硬件最强 SDPA 内核，Decode 阶段全量回归项目手写高性能 **Triton PagedAttention**。
- **双轨质量验收**（详见 [`docs/INFERENCE_ACCURACY.md`](docs/INFERENCE_ACCURACY.md)）：
  - **主验收（B 档）**：典型 prompt 下续写是否**可读、可理解**——用 `tests/tools/quality_bar_spotcheck.py` 做固定 prompt 抽检（配合人工判读）。
  - **回归 / 排障（A 档）**：`tests/verify_semantic_integrity.py` 与 Hugging Face 参考对比 prefill logits / greedy token，用于内核与加载路径调试；量化与 HF 参考不一致时，数值接近度**不等于**产品观感。若输出仍为乱码，应继续修引擎或权重路径，而不是仅降低预期。

## 🚀 快速开始

### 0. 环境准备（uv）
```bash
# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步项目依赖并创建虚拟环境
uv sync
```

> **约定**：本仓库内所有 Python 脚本、pytest 与 `python -m ...` 均应通过 **`uv run`** 执行（见 `.cursor/rules/uv-python.mdc`），勿直接用系统 `python3`/`pip`，以免环境与依赖不一致。

### 1. 观感抽检（B 档，推荐日常入口）
```bash
# 固定 prompt 续写 + 可读性/连贯性粗筛（不对比 HF）
PYTHONPATH=. uv run python tests/tools/quality_bar_spotcheck.py \
  --model models/<YOUR_MODEL> --quant awq|gguf|none --prompt-subset minimal
```

### 2. 与 HF 数值对比（A 档，内核 / 加载器回归）
```bash
# 与 HF 参考对比 prefill / greedy（见 docs/INFERENCE_ACCURACY.md 与 tests/README.md）
PYTHONPATH=. uv run python tests/verify_semantic_integrity.py --model models/TinyLlama-1.1B-Chat-v1.0 --quant none
```

### 3. 运行性能回归测试
```bash
# 执行全量架构的单层吞吐量测试
PYTHONPATH=. uv run python tests/full_perf_regression.py
```

## 🛠 当前算子与功能状态
| 类别 | 状态 | 备注 |
| :--- | :--- | :--- |
| **Continuous Batching** | ✅ **Active** | 支持多并发流式推理 |
| **MLA (Latent Attn)** | ✅ **Integrated** | 适配 DeepSeek-V2 / GLM-4.7 |
| **MoE (Expert Routing)** | ✅ **Optimized** | 支持 64+ 动态专家路由 |
| **Hybrid Attention** | ✅ **Verified** | Qwen3.5 专用混合路由 |
| **FP8 KV Cache** | ✅ **Default** | 节约 50% 显存占用 |

## 📄 架构深度解析
请参考 [docs/ARCHITECTURE_LITE.md](./docs/ARCHITECTURE_LITE.md)。

## 📘 API 文档
请参考 [docs/API_REFERENCE.md](./docs/API_REFERENCE.md)。

## 🛠️ 使用指南 (Usage)
### 1. 离线批处理推理 (Offline Batch Inference)
```python
from vllm import LLM, SamplingParams
# 支持 4096 上下文与 FP8 自动开启
llm = LLM(model="models/TinyLlama-1.1B-Chat-v1.0")
prompts = ["Explain quantum computing in simple terms."]
sampling_params = SamplingParams(max_tokens=128)
outputs = llm.generate(prompts, sampling_params)
```

### 2. 在线服务 (OpenAI API Server)
```bash
# 启动 API 服务器
uv run python -m vllm.entrypoints.openai.api_server --model models/TinyLlama-1.1B-Chat-v1.0
```
