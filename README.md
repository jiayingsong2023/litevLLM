# FastInference (vLLM Lite)

`FastInference` (vLLM Lite) 是一个 `lite-only` 的单卡推理引擎。项目当前优先级是收敛到稳定、可维护的 **纯 Python + Triton** 主路径，而不是继续维持对上游 vLLM 全量能力或大模型特化分支的兼容。

## 🚀 核心架构与成就
本引擎专注于 **纯 Python + Triton** 路径，实现单卡推理的高吞吐与低维护复杂度：

- **极致性能 (AMD Radeon 8060S 65GB 实测)**:
  | 模型 | 配置 | Aggregate TPS | 状态 |
  | :--- | :--- | :--- | :--- |
  | **TinyLlama-1.1B** | BS=32, 2048ctx | **542.4** | ✅ [1:1 HF 对齐] |
  | **Qwen3.5-9B (AWQ)** | BS=16, 4096ctx | **205.1** | ✅ [FP8 KV 稳定] |
  | **Gemma4-26B-A4B (AWQ)** | BS=1, prompt~384, max_new=24, KV cap=512 | **5.53** | ✅ [MoE `two_stage` + 2D Sampler] |
  | **Gemma4-31B-it (AWQ)** | BS=1, prompt~384, max_new=24, KV cap=512 | **1.62** | ✅ [Dense 稳定 + 2D Sampler] |

  最新 Gemma4 数值来自 `tests/e2e_full_benchmark.py` 的当前默认测量形状
  （2026-05-21，benchmark recommended profile）。当前报告中：
  Gemma4-26B `TTFT p50=2424.9ms`、`prefill_tps_agg=162.48`、`decode_tps_agg=12.03`；
  Gemma4-31B `TTFT p50=8604.3ms`、`prefill_tps_agg=45.79`、`decode_tps_agg=3.71`。

- **当前正式支持面**:
  - **运行模式**: 单卡、lite runtime、CUDA/ROCm 推理主路径。
  - **权重格式**: `Safetensors + AWQ` 为主，包含 Gemma4 Q4 压缩张量路径。
  - **回归目标**: `TinyLlama-1.1B`、`Qwen3.5-9B-AWQ`、`Gemma4-26B-A4B-it-AWQ-4bit`、`Gemma4-31B-it-AWQ-4bit`。
  - **非目标**: 不再将 `Qwen3.5-35B` 作为正式支持模型推进。

## 🌟 核心特性
- **lite-only 主线**: 运行时以 `vllm/engine/lite_engine.py` 为核心，优先收敛单卡执行链路。
- **统一构建链**: offline 与 OpenAI server 现在统一通过 `vllm/serving/config_builder.py` 构建 `VllmConfig + RuntimeConfig`，不再维护服务端旁路配置对象。
- **分层执行架构**: `LiteEngine` 负责 orchestration，`StepScheduler` 做 step 级调度，`RequestScheduler` 做 request/slot 生命周期管理，`PrefillExecutor` / `DecodeExecutor` 做执行，`SamplingDriver` / `OutputPipeline` 做采样与输出拼装。
- **模型适配层**: 模型特性识别通过 `vllm/adapters/` 下的 adapter 完成，避免继续在 engine 主路径中扩散模型名特判。
- **配置收敛目标**: 生产运行策略以 `Runtime Profiles -> RuntimeConfig` 为真源，模型热路径读取 runtime policy；benchmark / diagnostic tools 才保留调参开关。
- **纯净计算路径**: 主线优先维护 **Safetensors + AWQ**，实验性路径会逐步隔离。
- **混合加速 Prefill**: 预填充阶段利用硬件原生 SDPA，解码阶段全量回归手写高性能 **Triton PagedAttention**。
- **自动化元数据展开**: 统一的 `expand_metadata_for_paged_attention` 逻辑，完美支持全模型 Chunked Prefill。

## 🧱 当前主路径
当前官方执行链路为：

```text
LLM / AsyncLLM / OpenAI API Server
  -> config_builder
  -> LiteEngine
  -> StepScheduler
  -> RequestScheduler
  -> PrefillExecutor / DecodeExecutor
  -> SamplingDriver
  -> OutputPipeline
```

运行时观测与错误语义已开始收口到：

- `vllm/engine/runtime_observer.py`
- `vllm/engine/errors.py`

## 🛠️ Runtime Profiles
生产运行通过 `FASTINFERENCE_PROFILE` 选择 `Runtime Profiles`，而不是在 README 中推广一组长期暴露的 kernel / KV 环境变量。

常用 profile：

| Profile | 目标 | 说明 |
| :--- | :--- | :--- |
| `auto` | 默认平衡 | 让 registry 按当前模型/硬件选择稳定默认。 |
| `benchmark` | 基线测量 | 用于仓库默认性能回归和 profile-level 基线比较。 |
| `latency` | 单请求延迟 | 优先 TTFT / decode latency 的保守策略。 |
| `throughput` | 吞吐 | 优先批量吞吐和设备利用率。 |
| `accuracy` | 精度保护 | 优先数值保守配置，适合做 correctness / audit。 |

`benchmark` 和诊断工具可能暴露额外 tuning switches，用于 profile 试验、kernel 探索或内部观测；这些开关不是常规 production runtime controls，也不应替代 `RuntimeConfig` / profile data 作为生产配置入口。

## 🚀 快速开始

### 0. 环境安装
```bash
# 安装 uv 并同步依赖
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 1. 运行回归测试 (验证精度)
```bash
# 快速 lite 回归（结构 smoke + 单测）
uv run bash tests/run_regression_suite.sh

# 推理精度/质量回归
uv run bash tests/run_inference_correctness_regression.sh

# Gemma4 26B/31B 默认性能回归
uv run python tests/e2e_full_benchmark.py
```

`tests/run_inference_correctness_regression.sh` 覆盖四个回归目标：TinyLlama-1.1B、Qwen3.5-9B-AWQ、Gemma4-26B-A4B、Gemma4-31B。Gemma4 模型目录自动探测，优先本地路径，仅在缺失时回退到 HuggingFace repo id。

Gemma4 的少量 profiling / 诊断开关仍通过模型安装阶段的 tuning 配置生效，当前仅保留为 benchmark / diagnostics 的内部观测用途，不作为常规 production runtime switches：

- `FASTINFERENCE_GEMMA4_LAYER_PROFILE`
- `FASTINFERENCE_GEMMA4_ROCTX_PROFILE`

默认准确度策略为：

- `<=14B`：`A-strict + B`
- `>14B`：`A-lite + B`（Gemma4-26B 例外，默认开启 A-strict + A-lite + B）

### 2. 启动 OpenAI API 服务
```bash
# 针对 4090 等 24G+ 显卡优化的默认启动
uv run python -m vllm.entrypoints.openai.api_server --model models/Qwen3.5-9B-AWQ
```

## 📄 开发与架构
- **能力矩阵（支持面真源）**: 请参考 [docs/CAPABILITY_MATRIX.md](./docs/CAPABILITY_MATRIX.md)
- **架构深度解析**: 请参考 [docs/ARCHITECTURE_LITE.md](./docs/ARCHITECTURE_LITE.md)
- **lite-only 当前状态**: 请参考 [docs/LITE_ONLY_STATUS.md](./docs/LITE_ONLY_STATUS.md)
- **交付状态（DONE/KNOWN_GAPS）**: 请参考 [docs/DELIVERY_STATUS.md](./docs/DELIVERY_STATUS.md)
- **API 文档**: 请参考 [docs/API_REFERENCE.md](./docs/API_REFERENCE.md)

---
*本项目完全遵循 pure-Python 哲学，严禁引入任何需要 C++ 编译器的代码。*
