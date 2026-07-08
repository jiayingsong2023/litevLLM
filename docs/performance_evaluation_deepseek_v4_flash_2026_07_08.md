# DeepSeek V4 Flash 性能收敛报告

> 日期：2026-07-08  
> 分支：`feat/p0-perf-gates`  
> 测试硬件：Radeon 8060S，96 GB UMA（统一显存），ROCm  
> 目标模型：`models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`

---

## 执行摘要

DeepSeek V4 Flash 在当前实现下，**warm-resident 路径稳定在 1.6–1.8 tok/s**，这是 96 GB UMA 机器上的真实性能基线。围绕 cache/prefetch/staging 的优化（async prefetch、CPU payload cache）已被证明上限很低，不会突破该基线。

**战略转向**：
- 把 96 GB 机器的主路径定为 **full-resident / warm-cache**。
- cold-cache 优化只作为防退化手段，不再作为主要收益来源。
- 下一轮优化必须转向 **warm 路径下 `layer_moe` 内部的 GPU kernel/launch 分解**（ROCm profiler），而不是继续调整 staging 层。

---

## 当前性能指标

| 场景 | 命令/配置 | 结果 |
|---|---|---|
| **warm-cache gate** | `KV_TYPE=fp16`, `BLOCK_SIZE=32`, `FULL_RESIDENT=1`, `PIN_HOT_EXPERTS=1`, `STAGING_BUDGET_GB=1`, warmup=16, repeat=3 | 近期 median **1.625–1.885 tok/s**；P0-B 完整回滚后最新一次为 **1.710 / 1.817 / 1.821 tok/s**（median **1.817 tok/s**） |
| **e2e benchmark** | 同上，由 `tests/e2e_full_benchmark.py --models deepseek_v4_flash_q2_gguf` 调用 | decode_tps ≈ **1.79 tok/s** |
| **cold-start script, steady after cache warmup** | 默认 staging，`ASYNC_PREFETCH=0` | first run (true cold) **0.439 tok/s**；runs 2–3 median ≈ **1.67 tok/s**（方差大，受缓存状态影响） |
| **async prefetch on** | `ASYNC_PREFETCH=1`，cold-start（历史数据） | first run **0.525 tok/s**；runs 2–3 median ≈ **1.67 tok/s**，无回归。运行时代码已移除 |
| **CPU payload cache on** | `CPU_PAYLOAD_CACHE_BYTES=2GB`，cold-start（历史数据） | first run 与 steady 均无收益，`cpu_cache_hits=0`。运行时代码已移除 |

> 说明：cold-start 的第一次运行才是真正的 cold-cache 指标；由于 UMA 机器的默认 staging budget 足够容纳整个 21.6 GB working set，第一次运行后很快就进入“准 warm”状态。因此 **cold path 不宜作为主要优化目标**，只用于防退化。

---

## 已完成的实验与结论

### 1. Async expert prefetch（已移除运行时代码）

- **实现**：在独立 CUDA stream 上预取下一层可静态预测的专家权重，与当前层 compute 重叠。
- **覆盖率**：`deepseek_async_prefetch_scheduled_layers / opportunities` ≈ **4.8%**（例如 96/2016）。
- **瓶颈**：`_likely_hash_routed_expert_ids_for_token()` 只能对带 `expert_token_to_expert_ids` 表的层做静态预测；多数 MoE 层的路由依赖 router/hidden-state 输出，无法提前预测。
- **结论**：实现有效但上限低。产品分支已**移除 async prefetch 运行时代码**和相关环境变量 `FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH`；相关 A/B 脚本仅作为历史记录保留。

### 2. CPU payload cache（已移除运行时代码）

- **实现**：在 `stage_grouped_expert_payload()` 中对 `raw_grouped_expert_payload()` + `torch.frombuffer(...).clone()` 的结果做 LRU CPU 缓存。
- **实测**：`cpu_payload_cache_hits=0`，`misses=5703`，`evictions=4793`。
- **原因**：默认 UMA staging budget 已足够大，GPU cache 能装下整个 working set，每个 payload 只被 stage 一次，CPU 缓存没有第二次命中机会。
- **结论**：在 96 GB 机器上无收益，反而因 churn 略慢。产品分支已**移除 CPU payload cache 运行时代码**和相关环境变量 `FASTINFERENCE_DEEPSEEK_V4_FLASH_CPU_PAYLOAD_CACHE_BYTES`；相关 A/B 脚本仅作为历史记录保留。

### 3. Staging 耗时分解

通过新增 profiler section 得到 cold-path 单次 staging 的 breakdown：

| Section | 典型耗时 | 说明 |
|---|---|---|
| `raw_payload_read_clone` | ~1.0–1.8 ms / payload | CPU mmap 读取 + `torch.frombuffer(...).clone()`，占 staging 大头 |
| `h2d_copy_enqueue` | ~0.15–0.19 ms / payload | `cpu_payload.to(device, non_blocking=True)`，很小 |
| `cache_insert` | ~0.007 ms / payload | cache dict / LRU 更新，可忽略 |

这说明：**steady decode 的瓶颈不在 H2D copy，也不在 async/event 调度，而在 warm 路径的 MoE compute / attention**。

### 4. Warm-resident layer_moe breakdown（已完成）

固定 kept-path 配置，用 Python profiler 拆出的 warm decode breakdown（3-run，每 run 16 tokens，共 48 timed tokens + 16 warmup）：

| Section | count | total_ms | avg_ms | 说明 |
|---|---:|---:|---:|---|
| `layer_attention` | 6063 | 23,840.8 | 3.932 | **当前最大单section**；每层 attention 平均约 3.93 ms |
| `layer_moe` | 6063 | 23,120.8 | 3.813 | MoE 总 wrapper，含子 section |
| `moe_routed_experts` | 5640 | 14,431.6 | 2.559 | routed expert 总 wrapper |
| `router_selected_experts_kernel` | 6063 | 13,846.4 | 2.284 |  fused/quantized selected expert kernel 调用（含内部 up/gate/down） |
| `compressed_kv_update` | 5781 | 11,436.1 | 1.978 | compressed KV cache update，第三大热点 |
| `selected_expert_up_gate` | 6063 | 8,872.2 | 1.463 | selected expert gate + up projection + activation |
| `selected_expert_down` | 6063 | 4,768.0 | 0.786 | selected expert down projection |
| `moe_router_topk` | 5640 | 4,326.4 | 0.767 | router logits + top-k |
| `moe_shared_expert` | 6063 | 2,379.2 | 0.392 | shared expert 总 wrapper |
| `moe_shared_expert_gate_up` | 6063 | 1,580.7 | 0.261 | shared expert gate + up |
| `moe_shared_expert_down` | 6063 | 675.5 | 0.111 | shared expert down |
| `moe_hash_router` | 6063 | 1,506.7 | 0.249 | hash-routed/static expert lookup（部分层） |
| `router_expert_stage` | 6063 | 1,449.0 | 0.239 | warm 下剩余 staging，已很小 |

关键发现：
- `layer_attention`（23.8 s）已超过 `layer_moe`（23.1 s），是 warm 路径当前第一大热点。
- 在 `layer_moe` 内部，`router_selected_experts_kernel` 占主导；其内部 `selected_expert_up_gate` 约是 `selected_expert_down` 的 **1.86 倍**。
- `moe_router_topk` 平均 0.77 ms/层，不是最大头，但 61 层累计不少。
- staging 相关 section（`router_expert_stage`、`raw_payload_read_clone`、`h2d_copy_enqueue`）在 warm 路径下总量已经很小。

### 5. ROCm warm-resident kernel profile（已完成）

由于当前设备上 `rocprof` v1 不被支持，实际使用 `rocprofv3` 采集 kernel dispatch trace：

```bash
FASTINFERENCE_KV_TYPE=fp16 \
FASTINFERENCE_BLOCK_SIZE=32 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_FULL_RESIDENT=1 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_PIN_HOT_EXPERTS=1 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB=1 \
  rocprofv3 --kernel-trace --stats --summary -d /tmp/ds_warm_rocprof -f csv -- \
  uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
    --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
    --context-length 4096 --prompt-length 32 --max-tokens 16 --warmup-tokens 16 --repeat 3 \
    --min-steady-decode-tps 0.0 --profile-json /tmp/ds_warm_moe_rocprof.json
```

> 注意：profiler 采样开销使本次 steady TPS 降至约 **1.33 tok/s**，因此只用于 kernel 占比分析，不用于基线 TPS 衡量。

总 GPU kernel 耗时约 **57.65 s**，共 98 个不同 kernel。Top 10 如下：

| Kernel / Section | Calls | Total (s) | Avg (ms) | % |
|---:|---:|---:|---:|---:|
| `_iq2_xxs_selected_experts_activation_direct6_kernel` | 8,084 | 13.955 | 1.726 | 24.20 |
| `_q8_0_raw_matvec_kernel` | 106,596 | 12.197 | 0.114 | 21.16 |
| `_q2_k_selected_experts_down_projection_direct6_kernel` | 8,084 | 7.359 | 0.910 | 12.76 |
| `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT8x8x8_SN_...` (hipblaslt GEMM) | 31,396 | 6.993 | 0.223 | 12.13 |
| `pytorch elementwise_kernel_manual_unroll (div/copy/add)` 等 | 701,692+ | 1.518+ | ~0.002 | 2.63+ |
| `_q8_0_raw_gate_up_activation_kernel` | 8,084 | 1.501 | 0.186 | 2.60 |
| `pytorch reduce_kernel` 系列 | 341,564+ | 1.447+ | ~0.004 | 2.51+ |
| `__amd_rocclr_copyBuffer` | 311,991 | 0.870 | 0.003 | 1.51 |
| `rocblas_gemvt_warp_reduce_kernel` | 16,168 | 0.521 | 0.032 | 0.90 |
| `pytorch reduce_kernel (MeanOps/ArgMinOps)` | 70,468+ | 0.449+ | ~0.006 | 0.78+ |

按功能归并：

| 功能组 | Total (s) | % |
|---|---:|---:|
| `iq2_xxs_selected_experts_activation_direct`（up/gate 融合激活） | 13.955 | 24.20 |
| `q8_0_raw_matvec`（量化 matvec，来源待拆分） | 12.197 | 21.16 |
| `q2_k_selected_experts_down_projection_direct`（down projection） | 7.359 | 12.76 |
| hipblaslt / `Cijk_*` GEMM（attention/linear 通用 GEMM） | 7.202 | 12.49 |
| PyTorch elementwise / copy / fill / cat / arange 等 | ~12.9 | ~22.4 |
| `q8_0_raw_gate_up_activation_kernel` | 1.501 | 2.60 |
| 其他 | ~2.4 | ~4.2 |

#### Q8 raw matvec 调用归因

为拆分 `_q8_0_raw_matvec_kernel` 的 10.6 万次调用，在 4 个主要调用点外加 Python profiler section 后重跑 warm path（section 本身带来额外开销，本次 steady TPS 约 **1.09 tok/s**，只看相对分布）：

| Section | Calls | Total (ms) | Avg (ms) | 占 Q8 matvec 比例 |
|---:|---:|---:|---:|---:|
| `q8_projection_generic` | 24,252 | 6,762.4 | 0.279 | 47.0% |
| `q8_attention_output_grouped` | 48,504 | 5,869.5 | 0.121 | 40.8% |
| `q8_output_projection_chunk` | 1,128 | 1,071.7 | 0.950 | 7.4% |
| `q8_shared_expert_down` | 6,063 | 680.8 | 0.112 | 4.7% |

进一步用 `--profile-events` 按 `tensor.name` 聚合 `q8_projection_generic`：

| Tensor 类型 | Calls | Total (ms) | 占 generic 比例 |
|---:|---:|---:|---:|
| `attn_output_b.weight` | 6,063 | 3,857.3 | 53.1% |
| `attn_q_b.weight` | 6,063 | 1,628.6 | 22.4% |
| `attn_q_a.weight` | 6,063 | 1,054.1 | 14.5% |
| `attn_kv.weight` | 6,063 | 723.3 | 9.96% |

关键发现：
- `_q8_0_raw_matvec` 几乎**全部花在 attention 相关 Q8 projection** 上。
  - `q8_projection_generic` 100% 是 attention：`attn_output_b`（53%）、`attn_q_b`（22%）、`attn_q_a`（15%）、`attn_kv`（10%）。
  - `q8_attention_output_grouped` 100% 是 `attn_output_a.weight`。
  - 两者合计约 **13,690 ms**，占 Q8 raw matvec 总耗时的 **84.6%**。
- `q8_output_projection_chunk`（`output.weight`，7.4%）与 `q8_shared_expert_down`（4.7%）占比小，暂不优先。
- hipblaslt GEMM 占 **12.5%**，是 attention 路径上另一条独立热点，与 Q8 attention projection 不重叠。
- staging / H2D / cache 相关 kernel 在 warm 路径下几乎不可见，再次验证 staging 不是瓶颈。

### 6. P0-B 后的 fresh warm ROCm profile

固定 kept-path 配置，在 P0-B 代码合并后重跑 `rocprofv3`（profiler 开销使本次 steady TPS 降至约 **1.24–1.34 tok/s**，只看 kernel 占比）：

```bash
FASTINFERENCE_KV_TYPE=fp16 \
FASTINFERENCE_BLOCK_SIZE=32 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_FULL_RESIDENT=1 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_PIN_HOT_EXPERTS=1 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB=1 \
  rocprofv3 --kernel-trace --stats --summary \
    -d /tmp/ds_warm_rocprof_after_p0b -f csv -- \
    uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
      --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
      --context-length 4096 --prompt-length 32 --max-tokens 16 --warmup-tokens 16 --repeat 3 \
      --min-steady-decode-tps 0.0 --profile-json /tmp/ds_warm_after_p0b_profile.json
```

Top 10 kernel 对比（P0-B 前 → 后）：

| Rank | Kernel | P0-B 前 Calls / Total / % | P0-B 后 Calls / Total / % |
|---:|---|---:|---:|
| 1 | `_iq2_xxs_selected_experts_activation_direct6_kernel` | 8,084 / 13.955 s / 24.20% | 8,084 / 14.121 s / **24.61%** |
| 2 | `_q8_0_raw_matvec_kernel` | 106,596 / 12.197 s / 21.16% | **41,924 / 8.107 s / 14.13%** |
| 3 | `_q2_k_selected_experts_down_projection_direct6_kernel` | 8,084 / 7.359 s / 12.76% | 8,084 / 7.448 s / 12.98% |
| 4 | hipblaslt `Cijk_Alik_Bljk_*` GEMM | 31,396 / 7.202 s / 12.49% | 31,396 / 7.015 s / **12.23%** |
| 5 | `_q8_0_raw_grouped_matvec_kernel` | — | **8,084 / 3.444 s / 6.00%** |
| 6 | PyTorch div elementwise | ~701k / 1.501 s / 2.63% | 701,692 / 1.548 s / 2.70% |
| 7 | `_q8_0_raw_gate_up_activation_kernel` | 8,084 / 1.501 s / 2.60% | 8,084 / 1.519 s / 2.65% |

关键发现：
- **`_iq2_xxs_selected_experts_activation_direct6_kernel` 仍是 top-1**，占比 24.61%，绝对耗时与 P0-B 前基本持平。
- **`_q8_0_raw_matvec_kernel` 调用数从 106,596 降到 41,924**，占比从 21.16% 降到 14.13%；新增 `_q8_0_raw_grouped_matvec_kernel` 占 6.00%。两者合计约 20.13%，与 P0-B 前的 21.16% 基本持平，说明 P0-B 只是把 8 次 launch 合并，没有显著降低总 Q8 attention compute 时间。
- **hipblaslt GEMM 仍是 #4**，占比 12.23%，没有超过 MoE 成为主导。
- Python profiler 同期显示 `layer_attention`（33.0 s）> `layer_moe`（29.0 s），但 ROCm 视角下单个 `_iq2_xxs_selected_experts_activation_direct6_kernel` 仍是最大热点；`layer_attention` 的热是多个中小 kernel 叠加的结果。

结论：P0-B 后的 warm path 上，**`_iq2_xxs_selected_experts_activation_direct6_kernel` 仍是唯一占比 >20% 的独立热点**，满足 P0-A sweep 的前置条件。后续 P0-B 代码已完整回滚，本节 profile 仅作为当时实验证据保留。

### 7. P0-A 常量 sweep（negative result）

针对 `_iq2_xxs_selected_experts_activation_direct6_kernel` 做了窄范围常量 sweep。在 `deepseek_v4_iq2_xxs_selected_experts_activation_direct()` 中通过临时 env var 覆盖 `ROWS_PER_BLOCK` 和 `num_warps`，每组跑 warm no-profiler `repeat=3`。

测试配置：

| Config | ROWS_PER_BLOCK | num_warps |
|---|---|---|
| baseline | 4 | 8 |
| rows8 | 8 | 8 |
| warps16 | 4 | 16 |

第一轮 sweep 结果（顺序 baseline → rows8 → warps16）：

| Config | steady median | agg median |
|---|---|---|
| baseline | 1.436 tok/s | 1.429 tok/s |
| rows8 | 1.481 tok/s | 1.476 tok/s |
| warps16 | 1.600 tok/s | 1.600 tok/s |

第二轮反向确认（顺序 warps16 → baseline）：

| Config | steady median | agg median |
|---|---|---|
| warps16 | 1.648 tok/s | 1.641 tok/s |
| baseline | **1.885 tok/s** | **1.876 tok/s** |

分析：
- 第一轮 sweep 中 warps16 看似比 baseline 快 11.4%，但第二轮反向运行显示 **baseline（1.885）明显快于 warps16（1.648）**。
- 第一轮的结果更符合 **连续运行后的 thermal/cache 漂移**（数值单调上升 1.436 → 1.481 → 1.600），而非 kernel 常量本身的收益。
- 两次独立 run 中，baseline 本身波动很大（1.436 vs 1.885），说明在当前测试环境下，single-session 3-run median 受运行状态影响显著；常量 sweep 的信噪比不足以支撑结论。
- **没有任何一组达到硬退出条件**（warm 3-run median >= 1.70 tok/s）。

处置：
- **回滚 sweep 代码**：移除 `q2_iq2_moe.py` 中的 env var 覆盖，恢复 `_SELECTED_IQ2_ROWS_PER_BLOCK=4` 和 `num_warps=8`。
- 保留新增的 `test_iq2_xxs_selected_experts_activation_direct_matches_fused`，为 direct activation 路径提供正确性覆盖。
- 在报告中记录本次 negative result，不再继续调该 kernel 常量。

---

## 关键判断

1. **staging 层优化已触顶；实验代码已清理**
   - warm path 已经把 staging 绕开（full resident + pinned hot experts）。
   - cold path 的 async prefetch / CPU cache 实验上限很低，且运行时代码已从产品分支移除。
   - 继续调 event/stream/cache 只会改善冷启动/抖动，不会突破 1.6–1.8 tok/s 的 steady 上限。

2. **1.6–1.8 tok/s 是当前真实基线**
   - 在 96 GB UMA 机器上，`FULL_RESIDENT=1 + PIN_HOT_EXPERTS=1 + warmup=16` 可稳定复现。
   - 回归 gate 应以此为主：warm gate 阈值 1.5；cold gate 保守防退化即可。

3. **ROCm kernel profile 与 Q8 归因已完成；P0-B 证明 attention Q8 batching 不是当前突破口**
   - Python profiler 已证明 staging 不是 steady 上限。
   - ROCm profiler（P0-B 前）确认最大两个 kernel 是 `_iq2_xxs_selected_experts_activation_direct`（24.2%）与 `_q8_0_raw_matvec`（21.2%）。
   - **Q8 归因把 `_q8_0_raw_matvec` 拆清**：`q8_projection_generic` 100% 是 attention 权重，`q8_attention_output_grouped` 100% 是 `attn_output_a`。两者合计占 Q8 raw matvec的 **84.6%**。
   - P0-B 针对 `q8_attention_output_grouped` 做了 batched kernel，调用数从 48k 降到 ~1.7k，但 warm median 仍停留在 **1.625 tok/s**，未产生可测量的 e2e 提升。说明 attention 路径上还有与 Q8 projection 并行的主导开销（hipblaslt GEMM、整体 `layer_attention`），单纯减少 Q8 launch count 不足以突破基线。
   - 因此**下一步不是继续扩大 Q8 batching**，而是先重新采样 P0-B 后的 warm ROCm profile，再看热点是否已转移到 hipblaslt / `layer_attention`。

---

## 下一步建议

### P0：先 attribution，再优化

1. **P0-B：batched raw Q8 grouped matvec（已完成集成，局部成功、端到端未验证）**
   - 动机：`vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py:1958` 的 `output_groups` 循环每层对 `attn_output_a` 调用 8 次 `q8_0_raw_linear()`，再 `torch.stack()`；48k 调用中 40.8% 来自此处。
   - 实现：新增 `q8_0_raw_grouped_linear()`，`_grouped_output_projection()` 已切换为单次 launch。
   - 正确性：`tests/deepseek_v4_flash/test_triton_q8_linear.py::test_q8_0_raw_grouped_linear_matches_loop_of_raw_linear` 通过。
   - **局部结果**：`q8_attention_output_grouped` 调用次数从 **48,504 降到 1,677**（约每层 1 次）。
   - **端到端结果**：`tests/run_deepseek_v4_flash_real_smoke.sh` warm gate（repeat=3，无 profiler）steady TPS 为 **1.599 / 1.625 / 1.655 tok/s**（median **1.625 tok/s**），与此前基线 **1.6–1.8 tok/s** 持平，**未观察到可测量的 e2e 提升**。
   - **判定与处置**：P0-B 是 **局部成功、端到端失败的 bounded experiment**。由于它实际替换了默认的 `attn_output_a` 路径（从 8 次 `q8_0_raw_linear()` 循环改为单次 `q8_0_raw_grouped_linear()`），但又没有可测量的 TPS 收益，保留自定义 Triton kernel 只是债务。因此**已完整回滚**：
     - `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py` 恢复为原始的 8 次循环 + `torch.stack()`。
     - `vllm/kernels/triton/deepseek_v4_flash/q8_linear.py` 移除 `_q8_0_raw_grouped_matvec_kernel` 与 `q8_0_raw_grouped_linear()`。
     - `tests/deepseek_v4_flash/test_triton_q8_linear.py` 移除对应测试。
     - 不再继续深挖同类 Q8 batching。

2. **P0-A：`_iq2_xxs_selected_experts_activation_direct` 常量 sweep（24.61%，negative result）**
   - 已完成 2–3 组常量 sweep（baseline / rows8 / warps16），每组 warm no-profiler `repeat=3`。
   - 结果：第一轮 sweep 中 warps16 看似提升 11.4%，但反向确认显示 baseline（1.885 tok/s）明显优于 warps16（1.648 tok/s），提升来自连续运行的 thermal/cache 漂移，而非 kernel 常量优化。
   - **未达到硬退出条件**：没有任何一组 warm 3-run median >= 1.70 tok/s。
   - **已回滚**：移除 `q2_iq2_moe.py` 中的 env var 覆盖，恢复默认 `ROWS_PER_BLOCK=4` / `num_warps=8`。
   - 保留 `test_iq2_xxs_selected_experts_activation_direct_matches_fused` 作为 direct activation 路径的正确性覆盖。

3. **P0-C：按 tensor 单独评估其他 attention Q8 projections**
   - `q8_projection_generic` 已按 tensor 拆清：53% `attn_output_b`、22% `attn_q_b`、15% `attn_q_a`、10% `attn_kv`。它们每层各调用 1 次，不是 P0-B 那种 "8 launch → 1 launch" 问题，**不默认复用 P0-B 方案**。
   - 方向：用 fresh ROCm/Python profile 重新采样 P0-B 后的 warm path，再对每个 tensor 单独判断是调 block size、融合相邻 projection，还是保持不变。
   - 若 P0-A 优化后，`_q2_k_selected_experts_down_projection_direct`（12.8%）成为新瓶颈，再考虑优化 down projection kernel。

为验证优化效果，固定回归命令：

```bash
FASTINFERENCE_KV_TYPE=fp16 \
FASTINFERENCE_BLOCK_SIZE=32 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_FULL_RESIDENT=1 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_PIN_HOT_EXPERTS=1 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB=1 \
  uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
    --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
    --context-length 4096 --prompt-length 32 --max-tokens 16 --warmup-tokens 16 --repeat 3 \
    --min-steady-decode-tps 1.5 --profile-json /tmp/ds_opt_regression.json
```

### 可选：修正小显存实验模型

如果还要评估内存受限场景，需要先加一个真正的 GPU staging 硬上限（例如按字节或 GB 直接 cap `max_staged_bytes`），让 GPU cache 发生真实 eviction，再测 CPU cache / 批量 stage。否则 cold-path 实验在 96 GB 机器上不成立。

### 暂不投入

- async prefetch 与 CPU payload cache（运行时代码已移除）
- 图/capture（已被 `CAPABILITY_MATRIX` 明确拒绝）

---

## 相关文件与 Artifact

- `tests/run_deepseek_v4_flash_real_smoke.sh` — warm/cold gate
- `tests/run_deepseek_v4_flash_async_prefetch_ab.sh` — async prefetch A/B（历史记录，运行时代码已移除）
- `tests/run_deepseek_v4_flash_cpu_payload_cache_ab.sh` — CPU payload cache A/B（历史记录，运行时代码已移除）
- `docs/superpowers/plans/2026-07-07-deepseek-async-prefetch.md` — 实现计划与实验记录
- `docs/performance_evaluation_gemma4_deepseek_2026_07_07.md` — 早期评估报告
- `/tmp/ds_warm_rocprof/*/kernel_stats.csv` — ROCm warm-resident kernel profile 原始数据
- `/tmp/ds_warm_moe_rocprof.json` — 同次运行的 Python profiler JSON
- `/tmp/ds_q8_attribution.json` — Q8 raw matvec 调用归因 aggregate
- `/tmp/ds_q8_attribution_events.json` — Q8 raw matvec 调用归因 events（含 tensor metadata）

---

## 结论

DeepSeek V4 Flash 在 96 GB UMA 机器上的当前稳态性能为 **1.6–1.8 tok/s**（warm-resident），多次 fresh verification median 在 **1.625–1.885 tok/s** 之间波动。staging/cache/prefetch 层的优化潜力已耗尽；Python-level、ROCm-level 以及 Q8 raw matvec 调用归因均已完成。

- **P0-B 已完整回滚**：batched grouped Q8 matvec 把 `q8_attention_output_grouped` 调用数从 48k 降到 ~1.7k，但 warm median 仍停留在 **1.625 tok/s**，未超过基线噪声范围。由于它替换了默认的 `attn_output_a` 路径又没有可测量的 e2e 收益，保留自定义 Triton kernel 是债务，因此已恢复原始循环实现并移除相关 kernel/test。
- **P0-A 是 negative result**：对 `_iq2_xxs_selected_experts_activation_direct6_kernel` 的常量 sweep（rows8 / warps16）未通过硬退出条件；反向确认表明首轮“收益”是连续运行的 thermal/cache 漂移。已回滚 env var 覆盖。
- 当前 warm path 上，`_iq2_xxs_selected_experts_activation_direct6_kernel` 仍是 ROCm profile 中占比最高的单一 kernel（24.61%），但调常量无法撬动端到端 TPS。
- hipblaslt GEMM（12.23%）和整体 `layer_attention` 是并行的独立瓶颈。

下一步：**把目标从单 kernel 常量调参移开**。需要重新评估是继续拆解 `layer_attention` 内部（hipblaslt GEMM、Q8 attention projections、KV cache compression），还是接受 1.6–1.8 tok/s 为当前实现下的性能天花板。任何新优化都必须以 warm no-profiler 3-run median 的统计显著提升为准，并做反向顺序验证以排除 thermal 漂移。
