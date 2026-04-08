# FastInference (vLLM Lite) 架构解析

FastInference 是一个 `lite-only` 的单卡推理引擎。它基于 vLLM 代码基做了强裁剪和定制，目标是保留高性能推理核心，同时把项目边界收敛到可维护的 **纯 Python + Triton** 主路径。

## 1. 扁平化设计 (Flattened Architecture)
我们移除了原版 vLLM 中复杂的 `v1` 和 `v0` 目录嵌套，将核心入口统一为 `vllm/engine/lite_engine.py`。
- **解耦协议**: 核心数据结构被隔离在 `v1_protocol.py` 中，彻底解决了循环导入问题。
- **lite-only 收敛**: 正式支持面聚焦单卡、lite runtime 与核心 CUDA/ROCm 推理路径。
- **Zero-C++**: 核心哲学是 100% Python + Triton，确保在任何主流 Linux 环境下无需复杂编译即可运行。

## 2. 精简版加载器 (Lite ModelLoader)
针对 Safetensors 和 AWQ 架构，我们优化了加载逻辑：
- **深层后缀匹配**: 自动识别 `layers.{i}.{proj}.{attr}` 模式，完美支持 Qwen/Llama 系列的非标准权重 Key。
- **配置强制修正**: 自动补全 Qwen3.5 系列的关键元数据（如 `head_dim` 等）。
- **边界约束**: 主线以 `Safetensors + AWQ` 为主，其他权重格式与历史兼容路径将逐步隔离出正式支持面。

## 3. 高性能注意力体系 (High-Performance Attention)
针对单卡高吞吐场景，实现了全链路 Triton 算子：
- **GQA 硬件对齐**: 深度优化的 `reshape_and_cache` 与 `paged_attention` 算子，原生支持 GQA（Grouped Query Attention），解决了 16:2 等非对称比例下的索引越界风险。
- **强制 Paged Prefill**: 即使是首个 Prefill Chunk 也会通过 Paged 路径，确保 TurboQuant 的反量化逻辑在全生命周期内的一致性。
- **FP32 累积精度**: 在 PagedAttention 算子的中间计算（Softmax 归一化）阶段强制使用 FP32，显著提升了 FP8 和 INT4 路径的语义稳定性。

## 4. TurboQuant：动态 KV 压缩
为了在单卡上运行超长上下文，我们开发了 **TurboQuant**：
- **INT4 对称量化**: 使用 Signed 4-bit ([-7, 7]) 存储，通过手动的 32-bit 位移符号扩展保持精度。
- **Per-token Per-head Scaling**: 针对每个 Token 的每个 Head 实时计算缩放因子，最大限度保留激活值精度。
- **通用 MoE 基础设施**: 对大模型/MoE 的底层能力会保留并逐步抽象，但不再继续推进 `Qwen3.5-35B` 的正式产品支持。

## 5. 当前正式支持面
- **模型**: `TinyLlama-1.1B`、`Qwen3.5-9B-AWQ`
- **运行时**: `LiteEngine` 单卡主线
- **目标**: 稳定、可测试、可扩展的 lite runtime
- **非目标**: 不再将 `Qwen3.5-35B` 作为主线支持模型维护

## 6. 核心性能概览 (AMD Radeon 8060S 65GB)
- **TinyLlama-1.1B**: **542+ TPS** (Batch 32, 2k ctx)。
- **Qwen3.5-9B (AWQ)**: **200+ TPS** (Batch 16, 4k ctx)。

---
*本项目完全遵循 pure-Python 哲学，严禁引入任何需要 C++ 编译器的代码。所有算子均通过 Triton JIT 即时编译生成。*
