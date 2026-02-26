# FastInference (vLLM Lite) 架构解析

FastInference 是对 vLLM 的一种“外科手术式”重构，目标是建立一个代码量低于 100k LOC、高性能、全 Triton 实现的单卡引擎。

## 1. 物理规模管理 (LOC Reduction)
通过移除以下模块，我们将项目从 27 万行压缩至 **8.1 万行**：
- **分布式层**: `vllm/distributed` 简化为单卡 Mock。
- **投机采样**: 移除了 Medusa, Eagle 等数千行逻辑。
- **多平台冗余**: 移除了 TPU, XPU, OpenVINO 等 worker。
- **模型瘦身**: 仅保留 Llama, Qwen, DeepSeek, Mixtral, Kimi 五大核心系列。

## 2. 核心组件：`LiteLinear`
`LiteLinear` 是项目的心脏，负责单卡下的所有权重计算：
- **无并行开销**: 彻底删除了 Column/Row 并行逻辑。
- **权重缓存策略 (Weight Caching)**: 
  - 针对 GGUF 等量化格式，实施了 **Dequantize-Cache-MatMul**。
  - 在首次 Forward 时将量化权重反量化为 FP16 并常驻显存，后续推理直接执行纯 FP16 GEMM。
  - 此项优化让 GGUF 的吞吐量从 3.4 TPS 提升至 **174 TPS** (BS=32)。

## 3. 模型抽象：`LiteBase`
为了实现“极致可读性”，所有核心模型均继承自 `LiteModel`：
- **`Llama.py`**: 从原版的 500+ 行减少到 70 行。
- **`Qwen2_Moe.py`**: 统一了共享专家与稀疏专家的 Triton 调度路径。
- **`DeepSeek_V2.py`**: 精简了 MLA 投影层，强制单卡路径。

## 4. 算子层：Triton Only
- **`triton_attn.py`**: 唯一的注意力后端，移除了对 FlashAttention C++ 和 FlashInfer 的依赖。
- **`fused_moe.py`**: 精简版 Triton MoE 算子，硬编码了针对消费级 GPU (RTX 30/40系列) 的最佳 Tiling 配置，移除了繁琐的自动调优 (Autotune) 逻辑。

## 5. 性能基准 (RTX 4090)
- **TinyLlama-1.1B**: 24.8 TPS (Eager)。
- **Qwen1.5-MoE-2.7B**: 31.6 TPS (Eager)。
- **Llama-7B GGUF**: 174.3 TPS (BS=32, Cached)。
