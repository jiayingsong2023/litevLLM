# litevLLM 架构解析 (LiteEngine & Triton Only)

`litevLLM` 是一个基于 `vLLM` 深度重构的极致推理引擎，旨在通过 **纯 Python 和 Triton** 实现单 GPU 的高性能推理，完全剥离了原有的 C++、CUDA 和 ROCm 构建层。

## 1. 核心模块化重构：`LiteBase` 系统

为了提升可维护性并消除代码冗余，我们引入了 `vllm/model_executor/models/lite_base.py`。

### 1.1 统一架构基类
- **`LiteModel`**: 标准化的 Transformer Backbone 实现，统一处理 Embedding、DecoderLayers 循环和 Final Norm。
- **`LiteForCausalLM`**: 标准化的 Causal LM 封装，集成 `ParallelLMHead` 和 `LogitsProcessor`。
- **`LiteDecoderLayer`**: 通用的解码层，自动适配 Attention 和 MLP/MoE 模块。

### 1.2 极简模型适配
通过 `LiteBase`，Llama、Qwen2、Mistral 等架构现在只需几行代码即可完成定义：
```python
from .lite_base import LiteForCausalLM, LiteModel
LlamaModel = LiteModel
LlamaForCausalLM = LiteForCausalLM
```

## 2. 算子层标准化：`LiteLinear`

`vllm/model_executor/layers/lite_linear.py` 是本项目的核心算子封装：
- **统一接口**: 取代了标准的 `ColumnParallelLinear` 和 `RowParallelLinear`。
- **单 GPU 优化**: 移除了所有分布式通信（All-Reduce, All-Gather）开销。
- **智能权重加载**: `weight_loader` 能够自动识别并解析合并的权重（如 QKV 或 Gate-Up 投影），适配多种 Checkpoint 格式。
- **量化感知**: 内置对 AWQ、GGUF 和 FP8 的 Triton/Python 降级路径支持。

## 3. 推理引擎简化：`LiteEngine`

位于 `vllm/engine/lite_engine.py`：
- **扁平化调度**: 绕过了标准 vLLM 复杂的分布式调度器，直接管理 GPU 显存和请求队列。
- **强制 Triton 路径**: 核心 Attention 逻辑硬编码锁定为 `TRITON_ATTN` 后端。
- **分布式伪装 (`Shims`)**: 在 `vllm/distributed/` 中提供兼容层，让模型代码在单 GPU 下无感知地运行。

## 4. 注意力机制后端标准化

我们彻底清理了 `vllm/attention/backends/` 下的所有 C++ 编译后端：
- **删除**: 删除了 `FlashAttention` (C++), `FlashInfer`, `FlexAttention` 等二进制依赖。
- **唯一核心**: 全面转向 **`triton_attn.py`**。它实现了 FlashAttention 算法的 Triton 版本，确保了性能与可移植性的平衡。

## 5. 当前特性与限制

### 特性
- **零编译**: 无需 `nvcc` 或 `hipcc`，`uv pip install -e .` 即可运行。
- **高性能**: 在单 GPU 场景下，通过减少 Python/C++ 边界切换，性能优于原始 vLLM 的 Eager 模式。
- **全架构支持**: 支持 Llama, Qwen2, MoE, Gemma 等主流架构。

### 限制
- **单卡限制**: 不支持跨卡并行（Tensor Parallelism / Pipeline Parallelism）。
- **Graph Capture**: 部分 MoE 逻辑可能需要 `--enforce-eager` 以避免复杂 Python 控制流导致的 Graph 捕获失败。