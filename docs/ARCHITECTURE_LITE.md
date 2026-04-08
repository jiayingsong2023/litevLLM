# FastInference (vLLM Lite) 架构解析

FastInference 是一个 `lite-only` 的单卡推理引擎。它基于 vLLM 代码基做了强裁剪和定制，目标是保留高性能推理核心，同时把项目边界收敛到可维护的 **纯 Python + Triton** 主路径。

## 1. 当前分层结构
当前代码已经从“单个大 `LiteEngine`”收敛到分层主路径：

```text
LLM / AsyncLLM / OpenAI API Server
  -> vllm/serving/config_builder.py
  -> vllm/engine/lite_engine.py
  -> vllm/engine/step_scheduler.py
  -> vllm/engine/request_scheduler.py
  -> vllm/engine/prefill_executor.py
  -> vllm/engine/decode_executor.py
  -> vllm/engine/sampling_driver.py
  -> vllm/engine/output_pipeline.py
```

- **统一配置构建**: offline 与 server 共用 `config_builder.py`，统一生成 `VllmConfig + RuntimeConfig`。
- **模型能力适配**: `vllm/adapters/` 负责能力探测和 runtime capability 归一化。
- **执行解耦**: `LiteEngine` 只保留 orchestration，具体 prefill/decode 执行由 executor 层负责。
- **调度解耦**: `StepScheduler` 做 step 预算与批次选择，`RequestScheduler` 做 request/slot/stream 生命周期。
- **输出解耦**: `SamplingDriver` 做采样，`OutputPipeline` 做完成态与文本输出处理。
- **观测与错误解耦**: `RuntimeObserver` 和 `errors.py` 统一 runtime 事件和错误语义。
- **Zero-C++**: 核心哲学仍是 100% Python + Triton，确保在主流 Linux 环境下无需额外 C++ 编译链。

## 2. 精简版加载器 (Lite ModelLoader)
针对 Safetensors 和 AWQ 架构，我们优化了加载逻辑：
- **深层后缀匹配**: 自动识别 `layers.{i}.{proj}.{attr}` 模式，完美支持 Qwen/Llama 系列的非标准权重 Key。
- **配置强制修正**: 自动补全 Qwen3.5 系列的关键元数据（如 `head_dim` 等）。
- **边界约束**: 主线以 `Safetensors + AWQ` 为主，其他权重格式与历史兼容路径将逐步隔离出正式支持面。

## 3. 适配器与策略层
模型相关差异不再继续堆到 `LiteEngine`：

- **Adapter 层**: 位于 `vllm/adapters/`，负责模型 head 数、KV head 数、head dim、层数、KV 支持能力等归一化。
- **Policy 层**: 位于 `vllm/policies/`，当前作为对旧 `OutputProcessor` 的适配封装，承接 prompt / logits / output 相关策略。
- **设计规则**: 新模型优先通过 `adapter + policy + loader hook` 落地，而不是继续在 `LiteEngine` 写模型特判。

## 4. 高性能注意力体系 (High-Performance Attention)
针对单卡高吞吐场景，实现了全链路 Triton 算子：
- **GQA 硬件对齐**: 深度优化的 `reshape_and_cache` 与 `paged_attention` 算子，原生支持 GQA（Grouped Query Attention），解决了 16:2 等非对称比例下的索引越界风险。
- **强制 Paged Prefill**: 即使是首个 Prefill Chunk 也会通过 Paged 路径，确保 TurboQuant 的反量化逻辑在全生命周期内的一致性。
- **FP32 累积精度**: 在 PagedAttention 算子的中间计算（Softmax 归一化）阶段强制使用 FP32，显著提升了 FP8 和 INT4 路径的语义稳定性。

## 5. TurboQuant：动态 KV 压缩
为了在单卡上运行超长上下文，我们开发了 **TurboQuant**：
- **INT4 对称量化**: 使用 Signed 4-bit ([-7, 7]) 存储，通过手动的 32-bit 位移符号扩展保持精度。
- **Per-token Per-head Scaling**: 针对每个 Token 的每个 Head 实时计算缩放因子，最大限度保留激活值精度。
- **通用 MoE 基础设施**: 对大模型/MoE 的底层能力会保留并逐步抽象，但不再继续推进 `Qwen3.5-35B` 的正式产品支持。

## 6. LiteEngine 当前职责
截至当前代码，`LiteEngine` 主要负责：

- 组合 `StepScheduler`、`RequestScheduler`、executor、sampling、output pipeline
- 维护运行时主循环 `step()`
- 协调 abort、background error 与 request 完成态

以下职责已经迁出 `LiteEngine`：

- config 构建
- 模型能力探测
- request/slot 状态管理
- async 驱动
- prefill/decode 执行
- 采样
- 完成态判断与输出拼装
- runtime 事件观察与统一错误语义

## 7. 当前正式支持面
- **模型**: `TinyLlama-1.1B`、`Qwen3.5-9B-AWQ`
- **运行时**: `LiteEngine` 单卡主线
- **目标**: 稳定、可测试、可扩展的 lite runtime
- **非目标**: 不再将 `Qwen3.5-35B` 作为主线支持模型维护

## 8. 验证状态
当前架构重构后已验证：

- `bash tests/run_regression_suite.sh`
- `SKIP_A_TIER=1 bash tests/run_inference_accuracy_regression.sh`
- `uv run pytest tests/test_async_runtime_contracts.py -q`
- `uv run pytest tests/test_step_scheduler.py -q`
- `uv run pytest tests/test_runtime_observer.py -q`

## 9. 核心性能概览 (AMD Radeon 8060S 65GB)
- **TinyLlama-1.1B**: **542+ TPS** (Batch 32, 2k ctx)。
- **Qwen3.5-9B (AWQ)**: **200+ TPS** (Batch 16, 4k ctx)。

---
*本项目完全遵循 pure-Python 哲学，严禁引入任何需要 C++ 编译器的代码。所有算子均通过 Triton JIT 即时编译生成。*
