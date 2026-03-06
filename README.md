# FastInference (vLLM Lite)

`FastInference` (vLLM Lite) 是一个将 vLLM 核心代码物理精简至 **81,000 行** 的极致单卡推理引擎。它完全移除了分布式复杂性、C++ 依赖和冗余架构，专注于 **纯 Python + Triton** 的单 GPU 性能巅峰。

## 🚀 核心成就 (Performance Milestone)
- **极致吞吐量 (AMD AI Max 60GB 真实权重 + BS=128 实测)**:
  - **DeepSeek-V2-Lite (16B MoE)**: 🔥 **924.72 tokens/sec** (GGUF + **FP8 Triton Kernel**, 世界级巅峰性能).
  - **GLM-4.7-Flash (13B MoE)**: ⚡ **448.05 tokens/sec** (GGUF + **FP8**, BS=32, 提升 94.6%).
  - **TinyLlama-1.1B (Dense)**: **602.5 tokens/sec** (FP16).
  - **Qwen3.5-9B (GGUF)**: **240.6 tokens/sec** (🟢 **性能霸主**).
  - **Qwen3.5-9B (AWQ 4-bit)**: **147.7 tokens/sec** (Safetensors 真实加载).

- **长文本能力 (4096 Context 实测)**:
  - **DeepSeek-V2-Lite**: **389.2 tokens/sec** (BS=64, FP8).
  - **DeepSeek-V2-Lite (Single User)**: **21.23 tokens/sec** (BS=1, 响应无感).

- **架构级创新**:
  - **`FP8 Block-wise Scaling`**: 业内领先的 64x64 分块缩放 Triton 内核，兼顾性能与精度。
  - **`Global LiteLinear Acceleration`**: 全局 `LiteLinear` 层自动识别并缓存 FP8 权重，让 Qwen/Llama 系列自动受益。
  - **`LiteLoRA`**: 零拷贝低秩适配器注入，支持高并发多 Adapter 切换。
    - **TinyLlama-1.1B LoRA (Rank-16)**: **365.7 tokens/sec** (Batch 32, 🟢 **全量注入实测**).
    - **注意**: v1.0 版本对 Qwen3.5 和 DeepSeek-V2 的 LoRA 支持目前仅限于单并发测试；多并发 (BS>1) 适配将在 v2.0 版本中结合新的 Dynamic-Batching 调度器进行增强。
  - **`Real-Image Processing`**: 补齐多模态框架，支持原始 PIL 图像直接输入并自动转换为 GPU 张量。
  - **`Structured Output`**: 集成 Outlines 引擎，实现 100% 正确的 JSON Schema 和正则约束生成。

## 🌟 核心特性
- **纯净计算图**: 100% Triton 化的核心算子（Attention, Dequant, Norm, RoPE, Activation, Embedding）。
- **扁平化引擎**: 移除了 `v1` 嵌套目录，通过 `async_llm.py` 提供极简的异步推理入口。
- **稳定性保障**: 针对 AMD APU 优化的高级索引写入路径，彻底规避 BS=32 下的非法内存访问。
- **单例分词器**: 全局缓存机制，Mistral/Grok 等厂商特化分词器秒级加载。

## 🚀 快速开始

### 0. 环境准备（uv）
```bash
# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步项目依赖并创建虚拟环境
uv sync
```

### 运行端到端全量基准测试
```bash
# 执行密集、GGUF 与 MoE 综合测试
uv run python tests/e2e_full_benchmark.py

# 执行 LoRA 高并发扩展性测试
uv run python tests/e2e_lora_batch_scaling.py

# 执行真实图像输入预处理测试
uv run python tests/test_real_image_input.py
```

## 🛠 当前算子与功能状态
| 类别 | 状态 | 备注 |
| :--- | :--- | :--- |
| **LoRA** | ✅ **LiteLoRA** | 支持动态注入，BS=32 吞吐量 546 TPS |
| **Multi-modal** | ✅ **Full Framework** | 支持原始图像预处理，532 TPS |
| **Structured Output**| ✅ **Real Logic** | JSON/Regex 强约束生成 |
| **MoE Routing** | ✅ **Index-aware** | 零拷贝专家调度 |
| **KV Cache** | ✅ **FP8 / Paged** | 支持量化与物理分页管理 |

## 📄 架构深度解析
请参考 [docs/ARCHITECTURE_LITE.md](./docs/ARCHITECTURE_LITE.md)。

## 📌 稳定性修复说明
请参考 [docs/STABILITY_WORK_SUMMARY.md](./docs/STABILITY_WORK_SUMMARY.md)。

## 📘 API 文档
请参考 [docs/API_REFERENCE.md](./docs/API_REFERENCE.md)。

## ⚙️ Qwen3.5 策略默认值（固定策略）

- `Qwen3.5-9B (Dense)`：默认使用 **aggressive**（高吞吐优先）
  - 默认等价于 `FASTINFERENCE_QWEN9_AGGRESSIVE=1`
  - 如需切回稳定策略，可设置：`FASTINFERENCE_QWEN9_STABLE=1`
- `Qwen3.5-35B (MoE)`：默认使用 **stable**（可用性优先）
  - 线性层保持稳定优先配置，适合大模型长时间运行
  - MoE grouped 路径默认仅在 `BS>=2` 启用（`BS=1` 走 decode-only 快路径）

示例（强制 9B 稳定模式）：

```bash
FASTINFERENCE_QWEN9_STABLE=1 \
uv run python -m vllm.entrypoints.openai.api_server --model models/Qwen3.5-9B-GGUF
```

## 🛠️ 使用指南 (Usage)
### 1. 离线批处理推理 (Offline Batch Inference)
```python
from vllm import LLM, SamplingParams
llm = LLM(model="models/Qwen3.5-9B-GGUF")
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(max_tokens=32)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")
```

### 2. 在线服务 (OpenAI API Server)
```bash
# 启动极致性能的 OpenAI 兼容服务器
uv run python -m vllm.entrypoints.openai.api_server --model models/Qwen3.5-9B-GGUF
```

### 4. 依赖管理（uv）
```bash
# 新增运行时依赖
uv add <package>

# 新增开发依赖
uv add --dev <package>

# 更新锁文件
uv lock
```

### 3. 模型热切换 (Hot Switching)
```python
llm.switch_model("models/DeepSeek-V2-Lite-GGUF")
```
