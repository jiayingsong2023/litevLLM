# FastInference (vLLM Lite)

`FastInference` (vLLM Lite) 是一个 `lite-only` 的单卡推理引擎。项目已完成大规模精简，删除了 `vllm/worker/`、`vllm/core/`、`vllm/distributed/` 等上游 worker/distributed runtime 子系统。当前优先级是收敛到稳定、可维护的 **纯 Python + Triton** 主路径。

## 核心架构与成就
本引擎专注于 **纯 Python + Triton** 路径，实现单卡推理的高吞吐与低维护复杂度：

- **性能快照（AMD Radeon 8060S，96GB reported VRAM）**:
  | 模型 | 配置 | 实测 TPS | 状态 |
  | :--- | :--- | :--- | :--- |
  | **TinyLlama-1.1B** | BS=32, 2048ctx | **542.4** | [历史结果；未在 2026-07-17 重跑] |
  | **Qwen3.5-9B (AWQ)** | BS=16, 4096ctx | **205.1** | [历史结果；未在 2026-07-17 重跑] |
  | **Gemma4-26B-A4B (AWQ)** | BS=1, prompt~384, max_new=24, FP8 KV | **5.52 / 12.39** | [three isolated runs median, 2026-07-17] |
  | **Gemma4-31B-it (AWQ)** | BS=1, prompt~384, max_new=24, FP8 KV | **2.42 / 4.56** | [single isolated e2e run, 2026-07-17] |
  | **DeepSeek V4 Flash Q2 GGUF** | prompt~32, max_new=16, custom GGUF data plane | **0.37 / 1.78** | [single isolated e2e run, 2026-07-17] |

  最新数值来自 `tests/e2e_full_benchmark.py --model-process-isolation` 的单次
  运行（aggregate / decode TPS）。26B 三次独立进程的中位数为
  `TTFT p50=2307.9ms`、`prefill_tps_agg=170.72`、`decode_tps_agg=12.39`；
  31B 单次为 `TTFT p50=4852.7ms`、`prefill_tps_agg=81.19`、
  `decode_tps_agg=4.56`；DeepSeek 单次为 `decode_tps_agg=1.78`。
  26B 三次 aggregate TPS 为 5.52、5.36、5.76（约 7.3% peak-to-peak
  波动）；31B 与 DeepSeek 仍是单次验证样本，不能单独作为回归结论。Standalone GPU smoke
  does not use the standard per-token streaming observer, so `stream_visible=0%`
  is an observability limitation rather than a correctness failure.

  DeepSeek V4 Flash 的近期性能优化集中在不改变模型语义的热路径：
  selected-expert Q2/IQ2 直接 payload kernel、Q8_0 raw matvec sign-extension
  精简、compressor/indexer profile 拆分、Triton indexer QAT，以及移除小型
  PyTorch launch。已否决的路线包括 graph/capture、full expert GPU tables、
  Q2 down static unroll、batched Q8 raw matvec 和 compressor dual Q8 projection；
  这些路线要么未命中当前 decode 热点，要么实测变慢、冷启动/显存成本过高。

- **当前正式支持面**:
  - **运行模式**: 单卡、lite runtime、CUDA/ROCm 推理主路径。
  - **权重格式**: `Safetensors + AWQ` 为主，包含 Gemma4 Q4 压缩张量路径；DeepSeek V4 Flash 是目标 DS4 Q2/IQ2 GGUF 的实验性专用数据面，并接入同一 LiteEngine 控制面。
  - **回归目标**: `TinyLlama-1.1B`、`Qwen3.5-9B-AWQ`、`Gemma4-26B-A4B-it-AWQ-4bit`、`Gemma4-31B-it-AWQ-4bit`，以及目标 GGUF 存在时的 `DeepSeek-V4-Flash` Tier-B smoke。
  - **非目标**: `Qwen3.5-35B` 不再作为正式支持模型。
  - **已删除**: `vllm/worker/`、`vllm/core/`、`vllm/distributed/` 等上游 runtime 子系统。
  - **已删除**: 上游 CLI、pooling/gRPC/executor、spec decode、Helion、vendored `third_party/triton_kernels` 等非 lite 主路径残留。

## 核心特性
- **lite-only 主线**: 运行时以 `vllm/engine/lite_engine.py` 为核心，单卡执行链路。
- **统一配置构建**: offline 与 OpenAI server 统一通过 `vllm/serving/config_builder.py` 构建 `VllmConfig + RuntimeConfig`。所有 tuning 参数通过 TOML 配置文件 `[tuning_keyvals]` 传递，不再使用 `os.environ`。
- **分层执行架构**: `LiteEngine` 负责 orchestration，`AsyncDriver` 在后台工作线程执行 `engine.step()` 以避免 GPU sync 阻塞事件循环，`StepScheduler` 做 step 级调度并委托 admission/budget 给 `vllm/engine/planners/`，`RequestScheduler` 做 request/slot 生命周期管理，`PrefillExecutor` / `DecodeExecutor` 做执行，`SamplingDriver` 委托 penalty/mask 和 sampling 给 `vllm/engine/sampling/`，`OutputPipeline` 做输出拼装。
- **模型适配层**: 模型特性识别通过 `vllm/adapters/` 下的 adapter 完成，policy keys 有 `TypedDict` 类型约束。DeepSeek V4 Flash 由 adapter 安装专用 executor 与 KV lifecycle；generic engine 不包含模型名分支。
- **RequestState / RequestScheduler 清债**: `RequestState` 移除 dict-like shim，成为严格 dataclass；`RequestScheduler` 仅接受 `RequestState`，使用 set 索引和 deque 空闲 slot 池优化，同时保留请求 admission 顺序。
- **StepScheduler 薄片拆分**: admission、budget 与 prefill/decode plan 组装已分别拆出 `AdmissionPlanner`、`BudgetComputer` 和 `DecodePrefillPlanner`，`StepScheduler` 仅保留 step 级编排与委托调用。
- **SamplingDriver 向量化**: penalty（repetition / frequency / presence）、EOS mask、anti-template mask 改为 batch 级 PyTorch 操作；结构化输出约束与特殊 context bias 仍走逐行 fallback。
- **瘦身 StepPlan**: `StepPlan` 只承载执行字段；observer/debug 统计字段放在 `StepPlanMetrics`，由 runtime observer 消费。
- **decode 热路径优化**: 单请求 decode fast path 跳过自适应 chunk 扫描；普通 decode batch 复用 `InputBatchBuilder` 内部 scratch tensors，减少每步 tensor 构造。
- **配置收敛**: 生产运行策略以 `Runtime Profiles -> RuntimeConfig` 为真源，`FASTINFERENCE_CONFIG` 指向的 TOML 配置文件是公共配置入口；旧 `FASTINFERENCE_*` 名称只作为兼容或工具级开关保留。
- **纯净计算路径**: 主线优先维护 **Safetensors + AWQ**。
- **混合加速 Prefill**: 预填充阶段利用硬件原生 SDPA，解码阶段全量回归手写 **Triton PagedAttention**。
- **自动化元数据展开**: 统一的 `expand_metadata_for_paged_attention` 逻辑，支持全模型 Chunked Prefill。

## 当前主路径

```text
LLM / AsyncLLM / OpenAI API Server
  -> config_builder (TOML config)
  -> LiteEngine
  -> async_driver.py          # 后台工作线程执行 engine.step
  -> step_scheduler.py
     -> planners/             # AdmissionPlanner / BudgetComputer / DecodePrefillPlanner
  -> request_scheduler.py
  -> prefill_executor.py / decode_executor.py
  -> sampling_driver.py
     -> sampling/             # PenaltyEncoder / Sampler
  -> output_pipeline.py
```

运行时观测与错误语义收口到：

- `vllm/engine/runtime_observer.py`
- `vllm/engine/errors.py`

## Runtime Profiles

TOML 配置示例：

```toml
profile = "balanced"
kv_type = "turbo_int4"

[tuning_keyvals]
FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS = "1"
FASTINFERENCE_KV_MAX_MODEL_LEN = "512"
```

常用 profile：

| Profile | 目标 | 说明 |
| :--- | :--- | :--- |
| `auto` | 默认平衡 | 让 registry 按当前模型/硬件选择稳定默认。 |
| `balanced` | 默认服务 | 已认证模型 envelope 的默认生产策略。 |
| `latency` | 单请求延迟 | 优先 TTFT / decode latency 的保守策略。 |
| `throughput` | 吞吐 | 优先批量吞吐和设备利用率。 |
| `benchmark` / `accuracy` | 诊断 | 基线测量或 correctness/audit，不是生产服务 policy。 |

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

# Gemma4 26B/31B 与 DeepSeek 默认性能基准
uv run python tests/e2e_full_benchmark.py --model-process-isolation
```

`tests/run_inference_correctness_regression.sh` 覆盖 TinyLlama-1.1B、Qwen3.5-9B-AWQ、Gemma4-26B-A4B、Gemma4-31B；Gemma4-26B/31B 还会运行真实端到端 image multimodal quality spotcheck。当 DeepSeek V4 Flash 目标 GGUF 文件存在时，还会运行实验性 Tier-B quality smoke。Gemma4 模型目录自动探测，优先本地路径。

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
