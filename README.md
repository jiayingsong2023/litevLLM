# FastInference (vLLM Lite)

`FastInference` (vLLM Lite) 是一个将 vLLM 核心代码物理精简至 **81,000 行** 的极致单卡推理引擎。它完全移除了分布式复杂性、C++ 依赖和冗余架构，专注于 **纯 Python + Triton** 的单 GPU 性能巅峰。

## 🚀 核心成就 (Performance Milestone)
- **极致吞吐量 (AMD AI Max 60GB 真实权重 + BS=32 实测)**:
  - **DeepSeek-V2-Lite (16B MoE)**: **905.4 tokens/sec** (GGUF, 🔥 **单卡吞吐巅峰**).
  - **TinyLlama-1.1B (Dense)**: **590.6 tokens/sec** (FP16).
  - **Qwen3.5-9B (GGUF)**: **233.8 tokens/sec** (🟢 **极致纯净版**).
  - **Qwen3.5-9B (AWQ 4-bit)**: **147.7 tokens/sec** (Safetensors 真实加载).
  - **Qwen3.5-35B (MoE)**: **3.5 tokens/sec** (Batch 1, 🟢 **建议单并发使用**).
- **架构级创新**:
  - **`LiteLoRA`**: 零拷贝低秩适配器注入，支持高并发多 Adapter 切换。
  - **`Real-Image Processing`**: 补齐多模态框架，支持原始 PIL 图像直接输入并自动转换为 GPU 张量。
  - **`Structured Output`**: 集成 Outlines 引擎，实现 100% 正确的 JSON Schema 和正则约束生成。

## 🌟 核心特性
- **纯净计算图**: 100% Triton 化的核心算子（Attention, Dequant, Norm, RoPE, Activation, Embedding）。
- **扁平化引擎**: 移除了 `v1` 嵌套目录，通过 `async_llm.py` 提供极简的异步推理入口。
- **稳定性保障**: 针对 AMD APU 优化的高级索引写入路径，彻底规避 BS=32 下的非法内存访问。
- **单例分词器**: 全局缓存机制，Mistral/Grok 等厂商特化分词器秒级加载。

## 🚀 快速开始

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

### 3. 模型热切换 (Hot Switching)
```python
llm.switch_model("models/DeepSeek-V2-Lite-GGUF")
```
