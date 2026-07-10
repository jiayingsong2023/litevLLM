# Gemma4-26B-A4B 性能优化回顾报告

## 背景

项目对 Gemma4-26B-A4B（AWQ INT4 MoE）在 AMD ROCm 环境下的 BS=1 decode 性能进行专项优化。Baseline 约为 **11.4 tok/s**（roofline 等效 ~15 GB/s），明显低于同机 31B 的 ~52 GB/s，说明 26B 没有触到硬件带宽墙，存在优化空间。

最终结论：BS=1 decode 的瓶颈是 **kernel launch overhead 与小 kernel 碎片化**，不是单一算子或内存带宽问题。当前 ~11 tok/s 已接近该路径的工程上限；要大幅提升只能走 **batch decode** 或 **speculative / multi-token verifier** 这类摊薄 overhead 的架构方向。

---

## 探索路线与走过的弯路

### 1. 误判瓶颈性质：从 31B/DeepSeek 结论外推

- **初始假设**：26B 比 31B 慢，可能是 AWQ M=1 GEMV、MoE launch、或 KV cache 带宽没利用好。
- **实际数据**：26B roofline 等效带宽仅 ~15 GB/s，远低于 31B 的 ~52 GB/s；说明计算/launch 效率低，而不是硬件带宽没吃满。
- **教训**：不同模型/量化的瓶颈可能完全不同，不能直接套用其他模型的 profile 结论。

### 2. INT4 KV cache A/B：收益可忽略

- **做法**：在确认 runtime 实际生效的前提下，对比 `FASTINFERENCE_GEMMA4_ALLOW_INT4_KV=0`（fp8 KV）与 `=1`（turbo_int4 KV）。
- **结果**：decode TPS 中位数从 11.39 提升到 11.46，**仅 0.6%**。
- **教训**：理论上的带宽节省在 BS=1 时被其他开销淹没；涉及 accuracy guard 的环境变量不应轻易当性能开关推广。

### 3. MoE materialization / expert cache：死胡同

- **现象**：layer profile 最初显示 `moe_materialize_one_expert_awq` 占 17.8%，于是做了 cache size sweep（0/32/64/128）。
- **结果**：所有档位 `hit=0`，decode TPS 在噪声范围内，内存从 31 GB 涨到 94 GB。
- **根因**：
  1. 早期 profile 没把 warmup/prefill 与 decode 分开，把 prefill 阶段的 materialization 误归因到 decode。
  2. 26B A4B 的专家路由在 BS=1 decode 下几乎没有局部性，LRU cache 无法命中。
- **教训**：看到 materialization 热点时，必须先确认它发生在哪个阶段；cache 优化只在有 reuse locality 时才有意义。

### 4. Profiling 工具自身的坑导致绕路

- **问题 1**：`_gemma4_profile_marker` 早期只用 `layer_config.profile_enabled`，导致 `moe_int4_decode_fallback:*` 等 marker 大量丢失。
- **问题 2**：warmup 后只 reset 了 profile stats，没 reset MoE cache counters，`[gemma4-moe-cache]` 是进程级而非 decode-only。
- **修正后**：layer profile 显示 decode int4 fast path 基本命中；ROCTx 进一步确认 decode materialization 仅占约 **1%**。
- **教训**：测量工具要先校准，再基于测量结果做优化；否则容易追着 prefill/warmup 热点去优化 decode。

### 5. 过早把具体 draft/kernel 方案写进计划

- **做法**：计划中一度把 M=2/M=4 batched AWQ kernel、TinyLlama draft speculative 等具体实现直接列为 P2/P3 任务。
- **反思**：
  - batched AWQ micro-optimization 确实过早，M=2/M=4 microbench 没有证明收益。
  - 但在 BS=1 单点优化触顶后，**batch decode / speculative / multi-token verifier 作为架构级摊薄 overhead 的方向仍然成立**。
- **教训**：不要把具体实现提前落进计划；先证明单点优化到头，再决定走哪条架构级路线。

---

## 最终测量结论

使用 `rocprofv3 --kernel-trace --marker-trace` 配合 `FASTINFERENCE_GEMMA4_ROCTX_PROFILE=1`，在 decode-dominant 条件（prompt 8、warmup off、24 decode tokens）下做 innermost-range 归属，decode 阶段 GPU 时间分布大致如下：

| 区域 | GPU 时间占比 | 说明 |
|---|---|---|
| `layer_self_attn` | ~11% | rope、RMSNorm、attention 内部未单独打 range 的部分 |
| `moe_int4_decode_attempt` | ~10% | MoE int4 fast path（gate_up / down reduce） |
| `attn_o_proj` | ~7% | AWQ o_proj GEMV |
| `layer_dense_mlp` + 未归属 hipBLAS GEMM | ~18% | 库 kernel，非手写优化目标 |
| `layer_moe_router` | ~1.5% | router GEMV + top-k sort |
| `moe_materialize_one_expert_awq` | ~1% | decode 阶段 materialization 已不是瓶颈 |
| 大量 RMSNorm/elementwise/copy 小 kernel | 分散 | 多发生在 layer range 之间，每个 kernel 仅几 ms |

- **单 kernel 没有 >=15% 的热点**；最大的 hipBLAS GEMM 是库函数。
- 每个 decode token 约 launch **~5,600 个 kernel**，真正计算时间被 launch、同步、Python 调度、量化辅助 op 和 RMSNorm 碎片大量稀释。

---

## 建议

### 不要再做的事

- 不要继续优化 BS=1 单算子（MoE materialization、expert cache、KV dtype、batched AWQ GEMV、attention kernel）。
- 不要把临时诊断代码（profile markers、cache counters、atexit 打印、warmup 后 reset hooks）长期留在 runtime。

### 如果还要提升性能，只剩两条路

1. **batch decode（BS≥2）**
   - 把 GEMM/MoE 形状撑大，hipBLAS/AWQ/MoE 效率会立即提升。
   - 这是把 launch overhead 摊薄的最直接手段。

2. **speculative decoding / multi-token verifier**
   - 一次 forward 产生多个 token，摊薄 per-token 的 Python/launch/scheduling 开销。
   - 在 BS=1 场景下，这是绕过“单 token launch 过多”问题的合理架构方向。

### 工程决策

- **接受 BS=1 ~11 tok/s 作为当前工程上限**，把精力转移到支持 batching 或 speculative 的架构工作上。
- 如果业务场景强制 BS=1，当前性能已经是该路径下可接受的天花板，继续单点优化 ROI 很低。
