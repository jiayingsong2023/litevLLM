# FastInference (vLLM Lite) 架构解析

FastInference 是对 vLLM 的一种“外科手术式”重构，目标是建立一个高性能、纯 Triton 实现、完全剥离 C++ 依赖的单卡推理引擎。

## 1. 扁平化设计 (Flattened Architecture)
我们移除了原版 vLLM 中复杂的 `v1` 和 `v0` 目录嵌套，将核心入口统一为 `vllm/engine/lite_engine.py`。
- **解耦协议**: 核心数据结构被隔离在 `v1_protocol.py` 中，彻底解决了循环导入问题。
- **物理精简**: 移除了所有分布式、投机采样及多硬件 worker，仅保留核心 CUDA/ROCm 推理路径。
- **Zero-C++**: 核心哲学是 100% Python + Triton，确保在任何主流 Linux 环境下无需复杂编译即可运行。

## 2. 精简版加载器 (Lite ModelLoader)
针对 Safetensors 和 AWQ 架构，我们优化了加载逻辑：
- **深层后缀匹配**: 自动识别 `layers.{i}.{proj}.{attr}` 模式，完美支持 Qwen/Llama 系列的非标准权重 Key。
- **配置强制修正**: 自动补全 Qwen3.5 系列的关键元数据（如 `head_dim` 等）。
- **[DEPRECATED]**: 已彻底移除对 GGUF 及其相关 3D 专家解压逻辑的支持，以换取极简的代码维护性。

## 3. 高性能注意力体系 (High-Performance Attention)
针对单卡高吞吐场景，实现了全链路 Triton 算子：
- **GQA 硬件对齐**: 深度优化的 `reshape_and_cache` 与 `paged_attention` 算子，原生支持 GQA（Grouped Query Attention），解决了 16:2 等非对称比例下的索引越界风险。
- **强制 Paged Prefill**: 即使是首个 Prefill Chunk 也会通过 Paged 路径，确保 TurboQuant 的反量化逻辑在全生命周期内的一致性。
- **FP32 累积精度**: 在 PagedAttention 算子的中间计算（Softmax 归一化）阶段强制使用 FP32，显著提升了 FP8 和 INT4 路径的语义稳定性。

## 4. TurboQuant：动态 KV 压缩
为了在单卡上运行超长上下文，我们开发了 **TurboQuant**：
- **INT4 对称量化**: 使用 Signed 4-bit ([-7, 7]) 存储，通过手动的 32-bit 位移符号扩展保持精度。
- **Per-token Per-head Scaling**: 针对每个 Token 的每个 Head 实时计算缩放因子，最大限度保留激活值精度。
- **MoE 自动回退**: 针对 Qwen3.5 35B 等具有极端激活异常值（Outliers）的模型，引擎会自动检测并回退至 **FP8 KV Cache**，确保“精度优先”。

## 5. 核心性能概览 (AMD Radeon 8060S 65GB)
- **TinyLlama-1.1B**: **542+ TPS** (Batch 32, 2k ctx)。
- **Qwen3.5-9B (AWQ)**: **200+ TPS** (Batch 16, 4k ctx)。
- **Qwen3.5-35B (AWQ)**: **~40 TPS** (Batch 8, 1k ctx, FP8 KV)。

---
*本项目完全遵循 pure-Python 哲学，严禁引入任何需要 C++ 编译器的代码。所有算子均通过 Triton JIT 即时编译生成。*
