# FastInference (vLLM Lite) 架构解析

FastInference 是一个 `lite-only` 的单卡推理引擎。它基于 vLLM 代码基做了强裁剪和定制，目标是保留高性能推理核心，同时把项目边界收敛到可维护的 **纯 Python + Triton** 主路径。

## 1. 当前分层结构
当前代码已经从“单个大 `LiteEngine`”收敛到分层主路径：

```text
LLM / AsyncLLM / OpenAI API Server
  -> vllm/serving/config_builder.py  (TOML config -> VllmConfig + RuntimeConfig)
  -> vllm/engine/lite_engine.py          # orchestration
  -> vllm/engine/step_scheduler.py       # step budget & batch selection
  -> vllm/engine/request_scheduler.py    # request/slot lifecycle
  -> vllm/engine/prefill_executor.py     # prefill (hardware SDPA)
  -> vllm/engine/decode_executor.py      # decode (Triton PagedAttention)
  -> vllm/engine/sampling_driver.py      # sampling
  -> vllm/engine/output_pipeline.py      # output assembly
```

- **TOML 配置驱动**: 所有运行时 tuning 参数通过 `[tuning_keyvals]` TOML section 传递，不再使用 `os.environ`。production code 以 profile/runtime policy 为唯一策略真源。
- **模型能力适配**: `vllm/adapters/` 负责能力探测和 runtime capability 归一化，policy keys 有 `TypedDict` 类型约束。
- **执行解耦**: `LiteEngine` 只保留 orchestration，具体 prefill/decode 执行由 executor 层负责。
- **调度解耦**: `StepScheduler`（~18 参数，LoRA/MultiModal 已聚合成值对象）做 step 预算与批次选择。
- **输出解耦**: `SamplingDriver` 做采样，`OutputPipeline` 做完成态与文本输出处理。
- **观测与错误解耦**: `RuntimeObserver` 和 `errors.py` 统一 runtime 事件和错误语义。
- **Zero-C++**: 100% Python + Triton，确保无需额外 C++ 编译链。

## 2. High-Performance Architecture (2026 Q2 Update)

LitevLLM has been significantly optimized for Gemma4 dense and MoE models:

### 2.1 Operator Fusion & Kernel Specialization

- **MLP Gate-Up Fusion**: `packed_int4_symmetric_fused_gate_up_m1` fuses gate projection, up projection, and activation (SiLU/GELU) into a single Triton kernel, halving kernel launches for decode.
- **QKV Fusion**: `packed_int4_symmetric_group32_qkv_m1` fuses Q/K/V projections into a single kernel, sharing activation loads across three projections and saving ~2/3 of activation global memory traffic.
- **Split-K Decode**: Deep-K narrow-N projections (e.g., Gemma4-31B o_proj N=5376 K=16384) use split-K partial sums with atomic reduce to fill underutilized CUs.
- **M=1 GEMV Specialization**: All decode GEMM kernels have dedicated M=1 paths using broadcast FMA instead of `tl.dot`, avoiding wasted MFMA lanes on single-token batches.

### 2.2 Persistent & Cached Configurations

- **AWQ Fused Profile Cache**: Persistent JSON-based autotune profiles (`awq_fused_profile.json`) skip first-run autotune overhead for known shapes; production launch policy reads profile entries or heuristics, while tools may still explore alternate tiles offline.
- **CPU-Side Scalar Injection**: Engine-side `InputBatchBuilder` pre-computes `seq_lens_cpu`, `kv_start_indices_cpu` lists and passes them through `attn_metadata` so the per-layer model loop never triggers `hipMemcpyWithStream` / `.item()` D->H syncs — eliminated ~107ms across 60 layers on Gemma4-31B.
- **Reshape & Cache Scale Caching**: Scalar KV quantization scales are cached on the Python side via `_SCALE_TENSOR_CACHE` to avoid per-layer H->D memcpy for static scale values.

## 3. 精简版加载器 (Lite ModelLoader)
针对 Safetensors 和 AWQ 架构，我们优化了加载逻辑：
- **深层后缀匹配**: 自动识别 `layers.{i}.{proj}.{attr}` 模式。
- **配置强制修正**: 自动补全 Qwen3.5 系列的关键元数据。

## 4. 高性能注意力体系 (High-Performance Attention)
针对单卡高吞吐场景，实现了全链路 Triton 算子：
- **GQA 硬件对齐**: 深度优化的 `reshape_and_cache` 与 `paged_attention` 算子。
- **Fused Head Ops**: 集成了 RMSNorm 和 RoPE 的融合算子 `fused_head_ops`，在加载 Q/K/V 权重的同时完成归一化与旋转，减少了数千次 Kernel Launch。
- **FP32 累积精度**: 在 PagedAttention 算子的中间计算阶段强制使用 FP32。

## 5. TurboQuant：动态 KV 压缩
- **INT4 对称量化**: 使用 Signed 4-bit 存储，通过手动的 32-bit 位移符号扩展保持精度。
- **Per-token Per-head Scaling**: 针对每个 Token 的每个 Head 实时计算缩放因子。

## 6. 当前正式支持面
- **模型**: `TinyLlama-1.1B`、`Qwen3.5-9B-AWQ`、`Gemma4-26B-A4B-it-AWQ`、`Gemma4-31B-it-AWQ-4bit`
- **运行时**: `LiteEngine` 单卡主线
- **目标**: 稳定、可测试、可扩展的 lite runtime

## 7. 核心性能概览 (AMD Radeon 8060S 65GB)

最新 Gemma4 基线来自 2026-05-29 的 `tests/e2e_full_benchmark.py` 默认测量，
benchmark profile、`concurrent=1`、`prompt~128`、`max_new=48`、`KV cap=512`。

| 模型 | Aggregate TPS | TTFT p50 | Prefill TPS agg | Decode TPS agg | 状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TinyLlama-1.1B** | **542+** | 历史 fast-path 基线 | - | - | 1:1 HF 对齐 |
| **Gemma4-26B A4B (MoE)** | **8.00** | **2147.6ms** | **64.26** | **12.19** | P0/P1/P2 重构后稳定 |
| **Gemma4-31B (Dense)** | **2.06** | **5514.5ms** | **25.02** | **2.64** | P0/P1/P2 重构后稳定 |

---
*本项目完全遵循 pure-Python 哲学，严禁引入任何需要 C++ 编译器的代码。所有算子均通过 Triton JIT 即时编译生成。*
