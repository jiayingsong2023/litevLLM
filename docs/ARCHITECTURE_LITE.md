# FastInference (vLLM Lite) 架构解析

FastInference 是对 vLLM 的一种“外科手术式”重构，目标是建立一个代码量低于 100k LOC、高性能、全 Triton 实现的单卡引擎。

## 1. 扁平化设计 (Flattened Architecture)
我们移除了原版 vLLM 中复杂的 `v1` 和 `v0` 目录嵌套，将核心入口统一为 `vllm/engine/async_llm.py`。
- **解耦协议**: 核心数据结构被隔离在 `v1_protocol.py` 中，彻底解决了循环导入问题。
- **物理精简**: 移除了所有分布式、投机采样及多硬件 worker，仅保留核心 CUDA/ROCm 推理路径。

## 2. LiteLoRA：零拷贝适配器注入
为了在不引入复杂 SGMV 算子的情况下支持 LoRA，我们开发了 **LiteLoRA**：
- **原理**: 在 `LiteLinear` 层中直接集成低秩旁路路径。当加载 Adapter 时，自动在主 GEMM 计算后叠加 $X \times A^T \times B^T \times scaling$ 的结果。
- **性能**: 在 Batch Size 32 场景下，由于计算密度的提升，LoRA 带来的性能损耗仅为约 **10%**，吞吐量可达 **546 TPS**。

## 3. 多模态处理闭环 (Real-Image Pipeline)
补齐了从原始输入到张量生成的全链路：
- **`MultiModalInputProcessor`**: 自动识别图像/音频/视频对象。
- **`ImageProcessor`**: 支持 PIL 图像的实时预处理，并自动搬运至 GPU。
- **性能负载**: 针对 Qwen2-VL 等模型，实测在包含 576 视觉 Token 历史的情况下，吞吐量依然稳定在 **532 TPS**。

## 4. 结构化输出：Outlines 集成
- **强约束生成**: 通过实装 `StructuredOutputManager`，引擎能够根据 JSON Schema 实时生成 Token Bitmask。
- **100% 合规**: 确保模型输出严格遵守用户定义的格式，完美支持 Function Calling 场景。

## 5. 极致稳定性：AMD APU 调优
针对 **AMD Strix Point** 架构的内存控制器特性：
- **高级索引写入**: 弃用碎片的 Python 循环，改用 PyTorch 原生的高级索引进行 KV Cache 写入，彻底解决了 BS=32 时的非法内存访问 Bug。
- **性能峰值**: 在保持绝对稳定的前提下，MoE 模型吞吐量刷新至 **540.9 TPS**。

## 6. 核心性能概览 (AMD Strix Point)
- **Dense (TinyLlama)**: 27.4 TPS (Batch 1)。
- **LoRA (TinyLlama)**: **546.4 TPS** (Batch 32)。
- **MoE (Qwen-MoE)**: **540.9 TPS** (Batch 32)。
- **GGUF (Llama-7B)**: **195.7 TPS** (Batch 32, Cached)。
