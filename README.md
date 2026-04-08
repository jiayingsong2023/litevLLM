# FastInference (vLLM Lite)

`FastInference` (vLLM Lite) 是一个 `lite-only` 的单卡推理引擎。项目当前优先级是收敛到稳定、可维护的 **纯 Python + Triton** 主路径，而不是继续维持对上游 vLLM 全量能力或大模型特化分支的兼容。

## 🚀 核心架构与成就
本引擎专注于 **纯 Python + Triton** 路径，实现单卡推理的高吞吐与低维护复杂度：

- **极致性能 (AMD Radeon 8060S 65GB 实测)**:
  | 模型 | 配置 | Aggregate TPS | 状态 |
  | :--- | :--- | :--- | :--- |
  | **TinyLlama-1.1B** | BS=32, 2048ctx | **542.4** | ✅ [1:1 HF 对齐] |
  | **Qwen3.5-9B (AWQ)** | BS=16, 4096ctx | **205.1** | ✅ [FP8 KV 稳定] |

- **当前正式支持面**:
  - **运行模式**: 单卡、lite runtime、CUDA/ROCm 推理主路径。
  - **权重格式**: `Safetensors + AWQ` 为主。
  - **回归目标**: `TinyLlama-1.1B`、`Qwen3.5-9B-AWQ`。
  - **非目标**: 不再将 `Qwen3.5-35B` 作为正式支持模型推进。

## 🌟 核心特性
- **lite-only 主线**: 运行时以 `vllm/engine/lite_engine.py` 为核心，优先收敛单卡执行链路。
- **配置收敛目标**: 推理与 kernel 行为将逐步收敛到统一 typed config，而不是继续扩散模型专项环境变量。
- **纯净计算路径**: 主线优先维护 **Safetensors + AWQ**，实验性路径会逐步隔离。
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
# 自动验证当前正式支持模型的对齐情况
uv run bash tests/run_inference_accuracy_regression.sh
```

### 2. 启动 OpenAI API 服务
```bash
# 针对 4090 等 24G+ 显卡优化的默认启动
uv run python -m vllm.entrypoints.openai.api_server --model models/Qwen3.5-9B-AWQ
```

## 📄 开发与架构
- **架构深度解析**: 请参考 [docs/ARCHITECTURE_LITE.md](./docs/ARCHITECTURE_LITE.md)
- **lite-only 当前状态**: 请参考 [docs/LITE_ONLY_STATUS.md](./docs/LITE_ONLY_STATUS.md)
- **API 文档**: 请参考 [docs/API_REFERENCE.md](./docs/API_REFERENCE.md)

---
*本项目完全遵循 pure-Python 哲学，严禁引入任何需要 C++ 编译器的代码。*
