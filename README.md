# FastInference (vLLM Lite)

`FastInference` (vLLM Lite) 是一个 `lite-only` 的单卡推理引擎。项目已完成大规模精简，删除了 `vllm/worker/`、`vllm/core/`、`vllm/distributed/` 等上游 worker/distributed runtime 子系统。当前优先级是收敛到稳定、可维护的 **纯 Python + Triton** 主路径。

## 核心架构与成就
本引擎专注于 **纯 Python + Triton** 路径，实现单卡推理的高吞吐与低维护复杂度：

- **极致性能 (AMD Radeon 8060S 65GB 实测)**:
  | 模型 | 配置 | Aggregate TPS | 状态 |
  | :--- | :--- | :--- | :--- |
  | **TinyLlama-1.1B** | BS=32, 2048ctx | **542.4** | [1:1 HF 对齐] |
  | **Qwen3.5-9B (AWQ)** | BS=16, 4096ctx | **205.1** | [FP8 KV 稳定] |
  | **Gemma4-26B-A4B (AWQ)** | BS=1, prompt~128, max_new=48, KV cap=512 | **8.00** | [e2e benchmark 2026-05-29] |
  | **Gemma4-31B-it (AWQ)** | BS=1, prompt~128, max_new=48, KV cap=512 | **2.06** | [e2e benchmark 2026-05-29] |
  | **DeepSeek V4 Flash Q2 GGUF** | batch=1, context=4096, greedy, direct GGUF | **~1 TPS target gate** | [experimental e2e direct smoke] |

  最新 Gemma4 数值来自 `tests/e2e_full_benchmark.py`（2026-05-29，benchmark profile）。
  Gemma4-26B `TTFT p50=2148ms`、`prefill_tps_agg=64.26`、`decode_tps_agg=12.19`；
  Gemma4-31B `TTFT p50=5515ms`、`prefill_tps_agg=25.02`、`decode_tps_agg=2.64`。

- **当前正式支持面**:
  - **运行模式**: 单卡、lite runtime、CUDA/ROCm 推理主路径。
  - **权重格式**: `Safetensors + AWQ` 为主，包含 Gemma4 Q4 压缩张量路径；DeepSeek V4 Flash 通过目标 DS4 Q2/IQ2 GGUF 走实验性 direct path。
  - **回归目标**: `TinyLlama-1.1B`、`Qwen3.5-9B-AWQ`、`Gemma4-26B-A4B-it-AWQ-4bit`、`Gemma4-31B-it-AWQ-4bit`，以及目标 GGUF 存在时的 `DeepSeek-V4-Flash` Tier-B smoke。
  - **非目标**: `Qwen3.5-35B` 不再作为正式支持模型。
  - **已删除**: `vllm/worker/`、`vllm/core/`、`vllm/distributed/` 等上游 runtime 子系统。
  - **已删除**: 上游 CLI、pooling/gRPC/executor、spec decode、Helion、vendored `third_party/triton_kernels` 等非 lite 主路径残留。

## 核心特性
- **lite-only 主线**: 运行时以 `vllm/engine/lite_engine.py` 为核心，单卡执行链路。
- **统一配置构建**: offline 与 OpenAI server 统一通过 `vllm/serving/config_builder.py` 构建 `VllmConfig + RuntimeConfig`。所有 tuning 参数通过 TOML 配置文件 `[tuning_keyvals]` 传递，不再使用 `os.environ`。
- **分层执行架构**: `LiteEngine` 负责 orchestration，`StepScheduler` 做 step 级调度，`RequestScheduler` 做 request/slot 生命周期管理，`PrefillExecutor` / `DecodeExecutor` 做执行，`SamplingDriver` / `OutputPipeline` 做采样与输出拼装。
- **模型适配层**: 模型特性识别通过 `vllm/adapters/` 下的 adapter 完成，policy keys 有 `TypedDict` 类型约束。
- **配置收敛**: 生产运行策略以 `Runtime Profiles -> RuntimeConfig` 为真源，`FASTINFERENCE_CONFIG` 指向的 TOML 配置文件是公共配置入口；旧 `FASTINFERENCE_*` 名称只作为兼容或工具级开关保留。
- **纯净计算路径**: 主线优先维护 **Safetensors + AWQ**。
- **混合加速 Prefill**: 预填充阶段利用硬件原生 SDPA，解码阶段全量回归手写 **Triton PagedAttention**。
- **自动化元数据展开**: 统一的 `expand_metadata_for_paged_attention` 逻辑，支持全模型 Chunked Prefill。

## 当前主路径

```text
LLM / AsyncLLM / OpenAI API Server
  -> config_builder (TOML config)
  -> LiteEngine
  -> StepScheduler
  -> RequestScheduler
  -> PrefillExecutor / DecodeExecutor
  -> SamplingDriver
  -> OutputPipeline
```

运行时观测与错误语义收口到：

- `vllm/engine/runtime_observer.py`
- `vllm/engine/errors.py`

## Runtime Profiles

TOML 配置示例：

```toml
profile = "benchmark"
kv_type = "turbo_int4"

[tuning_keyvals]
FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS = "1"
FASTINFERENCE_KV_MAX_MODEL_LEN = "512"
```

常用 profile：

| Profile | 目标 | 说明 |
| :--- | :--- | :--- |
| `auto` | 默认平衡 | 让 registry 按当前模型/硬件选择稳定默认。 |
| `benchmark` | 基线测量 | 用于仓库默认性能回归和 profile-level 基线比较。 |
| `latency` | 单请求延迟 | 优先 TTFT / decode latency 的保守策略。 |
| `throughput` | 吞吐 | 优先批量吞吐和设备利用率。 |
| `accuracy` | 精度保护 | 优先数值保守配置，适合做 correctness / audit。 |

## 快速开始

### 0. 环境安装
```bash
# 安装 uv 并同步依赖
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 1. 运行回归测试
```bash
# 快速代码回归（结构 smoke + 单测，无完整模型加载）
bash tests/run_regression_suite.sh

# 推理准确性/质量回归（本地模型语义门禁）
SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh

# Gemma4 26B/31B 默认性能回归
uv run python tests/e2e_full_benchmark.py
```

`tests/run_inference_correctness_regression.sh` 覆盖 TinyLlama-1.1B、Qwen3.5-9B-AWQ、Gemma4-26B-A4B、Gemma4-31B；当 DeepSeek V4 Flash 目标 GGUF 文件存在时，还会运行实验性 Tier-B quality smoke。Gemma4 模型目录自动探测，优先本地路径。

默认准确度策略为：`<=14B`：A-strict + B；`>14B`：A-lite + B（Gemma4-26B 例外，默认开启 A-strict + A-lite + B）。在 60GB 级 GPU 上可通过 `SKIP_A_TIER=1` 跳过 A-tier 测试以节省显存。

### 2. 启动 OpenAI API 服务
```bash
uv run python -m vllm.entrypoints.openai.api_server --model models/Qwen3.5-9B-AWQ
```

## 文档与架构

- **依赖闭包**: [docs/DEPENDENCY_CLOSURE.md](./docs/DEPENDENCY_CLOSURE.md) — lite 路径 import 边界
- **能力矩阵**: [docs/CAPABILITY_MATRIX.md](./docs/CAPABILITY_MATRIX.md)
- **架构解析**: [docs/ARCHITECTURE_LITE.md](./docs/ARCHITECTURE_LITE.md)
- **lite-only 状态**: [docs/LITE_ONLY_STATUS.md](./docs/LITE_ONLY_STATUS.md)

---
*本项目完全遵循 pure-Python 哲学，严禁引入任何需要 C++ 编译器的代码。*
