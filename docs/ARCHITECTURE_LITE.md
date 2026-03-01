# FastInference (vLLM Lite) 架构解析

FastInference 是对 vLLM 的一种“外科手术式”重构，目标是建立一个代码量低于 100k LOC、高性能、纯 Python + Triton 实现的单卡推理引擎。

## 1. 物理规模管理 (LOC Reduction)
通过移除以下模块，我们将项目从 27 万行压缩至 **8.1 万行**：
- **分布式层**: `vllm/distributed` 简化为单卡 Mock。
- **投机采样**: 移除了 Medusa, Eagle 等复杂采样逻辑。
- **多平台冗余**: 移除了 TPU, XPU, OpenVINO 等 worker，仅保留 CUDA/ROCm 路径。
- **存根化 (Stubbing)**: 修复了大量损坏的 multimodal 组件，确保核心推理循环的纯净。

## 2. 核心组件：`LiteLinear` 与 LRU Caching
`LiteLinear` 引入了 **Global LRU 权重缓存策略**：
- **原理**: 针对 GGUF 等量化格式，在首次推理时调用 Triton Kernel 进行反量化，并将结果缓存为 FP16。
- **管理**: 采用 LRU 淘汰机制（默认容量 128 层），确保显存占用可控。
- **收益**: 消除重复解压开销，GGUF 吞吐量在 Batch Size 32 下提升至 **195+ TPS**。

## 3. MoE 极致优化：Index-aware GEMM
为了解决 MoE 模型在单卡上的调度开销，我们实现了 **索引感知矩阵乘法 (Index-aware GEMM)**：
- **零拷贝调度**: 在 Triton Kernel 读取数据时直接进行索引映射，彻底消除了显式的数据重排 (Permute/Gather) 开销。
- **原子累加**: 利用 `tl.atomic_add` 支持多个专家并发写回输出缓冲区。
- **性能飞跃**: Qwen-MoE 在 Batch Size 32 下实测吞吐量达到 **533 TPS**。

## 4. 算子融合与稳定性权衡
为了在 AMD APU (Strix Point) 等特殊架构上支持大规模并发，我们采取了灵活的算子策略：
- **计算密集型 (GEMM/Dequant)**: 100% 采用极致优化的 Triton Kernels。
- **稳定性敏感型 (Norm/RoPE/KV-Write)**: 在大 Batch 下自动切换至 PyTorch 稳定版算子（高级索引赋值），规避非法内存访问 (Error 700) 风险。
- **Fused Prefill**: 实现了一套支持 Paged 寻址、FP8 实时量化与 FlashAttention 计算三合一的预填充内核。

## 5. 性能基准 (AMD Strix Point / RTX 系列对齐)
- **TinyLlama-1.1B**: 27.4 TPS (Decode, Batch 1)。
- **Qwen1.5-MoE-2.7B**: **533.2 TPS** (Total Throughput, Batch 32)。
- **Llama-7B GGUF**: **195.7 TPS** (Total Throughput, Batch 32, Cached)。
