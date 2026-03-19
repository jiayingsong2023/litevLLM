# FastInference (vLLM Lite) 架构解析

FastInference 是对 vLLM 的一种“外科手术式”重构，目标是建立一个代码量低于 100k LOC、高性能、全 Triton 实现的单卡引擎。

## 1. 扁平化设计 (Flattened Architecture)
我们移除了原版 vLLM 中复杂的 `v1` 和 `v0` 目录嵌套，将核心入口统一为 `vllm/engine/async_llm.py`。
- **解耦协议**: 核心数据结构被隔离在 `v1_protocol.py` 中，彻底解决了循环导入问题。
- **物理精简**: 移除了所有分布式、投机采样及多硬件 worker，仅保留核心 CUDA/ROCm 推理路径。

## 2. 自愈式加载器 (Self-Healing ModelLoader)
针对 GGUF、AWQ 及多种非标准架构，我们开发了通用的 `model_loader`：
- **深层后缀匹配**: 不再依赖于固定的 Key 映射，而是通过 `layers.{i}.{proj}.{attr}` 的后缀模式自动识别权重位置。
- **3D 专家解压**: 支持从 GGUF 中即时反量化 3D 形状的专家张量 [Experts, Rows, Cols]，解决了传统加载器无法处理非对称 MoE 权重的问题。
- **配置强制修正**: 自动识别并补全 DeepSeek MLA 的 `head_dim` (128) 及 `kv_lora_rank` 等关键元数据。

## 3. 混合注意力机制 (Hybrid Attention)
在 Qwen3.5 等新模型中，FastInference 率先支持了混合层路由：
- **Linear Attention**: 在部分层实现了基于递归形式（Recurrent Form）的线性注意力，有效降低长序列计算复杂度。
- **Full Attention**: 在关键同步层保持标准的 Multi-Head Attention (MHA)，确保全局语义捕捉。
- **动态分发**: 引擎在加载时自动根据权重特征决定每一层的计算逻辑，无需用户干预。

## 4. MLA (Multi-Head Latent Attention) 实现
为了在 Strix Point 上高效运行 DeepSeek-V2 系列：
- **潜变量解压**: 在 Attention 计算前，利用高性能 PyTorch 算子将 KV 潜变量解压为物理 Q/K/V 矩阵。
- **RoPE 分离逻辑**: 完美实现了 Decoupled RoPE，将 nope 部分与 rope 部分在计算流中正确融合。

## 5. LiteLoRA：零拷贝适配器注入
为了在不引入复杂 SGMV 算子的情况下支持 LoRA，我们开发了 **LiteLoRA**：
- **原理**: 在 `LiteLinear` 层中直接集成低秩旁路路径。当加载 Adapter 时，自动在主 GEMM 计算后叠加 $X \times A^T \times B^T \times scaling$ 的结果。
- **性能**: 在 Batch Size 32 场景下，由于计算密度的提升，LoRA 带来的性能损耗仅为约 **10%**。

## 6. 核心性能概览 (AMD Radeon 8060S)
- **TinyLlama-1.1B**: **542.4 TPS** (Batch 32)。
- **Qwen3.5-9B (AWQ)**: **205.1 TPS** (Batch 32)。
- **DeepSeek-V2-Lite**: **112.7 TPS** (Batch 16)。
- **GLM-4.7-Flash**: **110.5 TPS** (Batch 16)。
