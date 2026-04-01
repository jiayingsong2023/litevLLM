# FastInference (vLLM Lite)

`FastInference` (vLLM Lite) 是一个极度精简、极致优化的单卡推理引擎。通过彻底剥离 C++ 依赖、分布式复杂性及冗余加载路径，它在 **32/48GB 显存** 的消费级/高端卡上实现了长文本推理的性能巅峰。

## 🚀 核心架构与成就
本引擎专注于 **纯 Python + Triton** 路径，实现了全链路手写算子加速：

- **极致性能 (AMD Radeon 8060S 65GB 实测)**:
  | 模型 | 配置 | Aggregate TPS | 状态 |
  | :--- | :--- | :--- | :--- |
  | **TinyLlama-1.1B** | BS=32, 2048ctx | **542.4** | ✅ [1:1 HF 对齐] |
  | **Qwen3.5-9B (AWQ)** | BS=16, 4096ctx | **205.1** | ✅ [FP8 KV 稳定] |
  | **Qwen3.5-35B (AWQ)** | BS=8, 1024ctx | **~40** | ✅ [48G 显存压榨] |

- **针对 48GB GPU 的深度优化**:
  - **Aggressive KV Pool**: 自动探测大显存环境，默认开启 16+ 高并发槽位。
  - **Parallel Greedy Sampling**: 移除 Python 循环，通过矢量化算子实现 Batch 并行采样，消除并发瓶颈。
  - **TurboQuant INT4**: 深度修复的 4-bit KV 缓存算子，支持超长上下文（32k-128k）下的显存深度压缩。

## 🌟 核心特性
- **单一真相配置**: 所有推理参数通过 `LiteInferenceConfig` 统一管理，杜绝环境变量滥用。
- **纯净计算路径**: 仅支持 **Safetensors + AWQ**（已彻底移除 GGUF 冗余代码）。
- **混合加速 Prefill**: 预填充阶段利用硬件原生 SDPA，解码阶段全量回归手写高性能 **Triton PagedAttention**。
- **自动化元数据展开**: 统一的 `expand_metadata_for_paged_attention` 逻辑，完美支持全模型 Chunked Prefill。

## 🛠️ 配置指南 (LiteInferenceConfig)
通过环境变量控制 `LiteInferenceConfig` 行为，实现性能与精度的平衡：

| 环境变量 | 可选值 | 描述 |
| :--- | :--- | :--- |
| `FASTINFERENCE_KV_TYPE` | `fp16`, `fp8`, `turbo_int4` | 控制 KV 缓存精度（默认 `auto`->`fp8`） |
| `FASTINFERENCE_FUSION_LEVEL` | `0`, `1`, `2` | `1`: AB 融合; `2`: 全量算子融合 (默认) |
| `FASTINFERENCE_BLOCK_SIZE` | `16`, `32`, `64` | 物理块大小，长文本建议 `32/64` |
| `FASTINFERENCE_K_SCALE` | `float` | TurboQuant 专用 K 缩放因子 |

## 🚀 快速开始

### 0. 环境安装
```bash
# 安装 uv 并同步依赖
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 1. 运行回归测试 (验证精度)
```bash
# 自动验证 TinyLlama 与 Qwen3.5 的对齐情况
uv run bash tests/run_inference_accuracy_regression.sh
```

### 2. 启动 OpenAI API 服务
```bash
# 针对 4090 等 24G+ 显卡优化的默认启动
uv run python -m vllm.entrypoints.openai.api_server --model models/Qwen3.5-9B-AWQ
```

## 📄 开发与架构
- **架构深度解析**: 请参考 [docs/ARCHITECTURE_LITE.md](./docs/ARCHITECTURE_LITE.md)
- **API 文档**: 请参考 [docs/API_REFERENCE.md](./docs/API_REFERENCE.md)

---
*本项目完全遵循 pure-Python 哲学，严禁引入任何需要 C++ 编译器的代码。*
