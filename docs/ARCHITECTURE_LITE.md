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

## 2. High-Performance Architecture (2026 Q2 Update)

LitevLLM has been significantly optimized for Gemma4 dense and MoE models:

### 2.1 MoE Single-Launch Architecture (Gemma4 26B/A4B)
To overcome Python-side dispatch overhead in deep MoE models, we implemented a Single-Launch architecture:
- **Block Mapping:** Tokens assigned to the same expert are partitioned into fixed-size blocks. A lookup table maps each Triton program block to a specific expert and token offset.
- **Unified Kernel:** Replaced fragmented sequential expert processing with a single-launch `awq_moe_grouped_gemm` operator. 
- **Physical Alignment:** Weights are processed in a consistent [E, N, K/8] 3D layout, eliminating layout ambiguities and ensuring high-fidelity AWQ dequantization.
- **Impact:** Achieved >9x throughput improvement for MoE models, effectively hiding launch latency.

### 2.2 Dense Model Fusion & Caching (Gemma4 31B)
For high-parameter dense models where memory bandwidth is the primary bottleneck:
- **Operator Fusion:** Implemented `fused_add_rmsnorm` to combine residual additions with RMSNorm, reducing global memory round-trips per transformer layer.
- **Persistent Cache:** Added `materialize_cache()` support in `LiteLinear` to selectively pre-dequantize early model layers (e.g., layers 0-4) into FP16. This bypasses AWQ dequantization logic for compute-heavy early stages, prioritizing memory bandwidth utilization on high-end GPUs.

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
- **TinyLlama-1.1B**: **542+ TPS**
- **Gemma4-26B (MoE)**: **2.31 TPS** (Verified stable)
- **Gemma4-31B (Dense)**: **0.90 TPS** (Memory Bound)

---
*本项目完全遵循 pure-Python 哲学，严禁引入任何需要 C++ 编译器的代码。所有算子均通过 Triton JIT 即时编译生成。*
