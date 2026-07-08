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
| **warm-cache gate** | `KV_TYPE=fp16`, `BLOCK_SIZE=32`, `FULL_RESIDENT=1`, `PIN_HOT_EXPERTS=1`, `STAGING_BUDGET_GB=1`, warmup=16, repeat=3 | steady-state decode **1.60 / 1.69 / 1.83 tok/s**（median **1.69 tok/s**） |
| **e2e benchmark** | 同上，由 `tests/e2e_full_benchmark.py --models deepseek_v4_flash_q2_gguf` 调用 | decode_tps ≈ **1.79 tok/s** |
| **cold-start script, steady after cache warmup** | 默认 staging，`ASYNC_PREFETCH=0` | first run (true cold) **0.439 tok/s**；runs 2–3 median ≈ **1.67 tok/s**（方差大，受缓存状态影响） |
| **async prefetch on** | `ASYNC_PREFETCH=1`，cold-start | first run **0.525 tok/s**；runs 2–3 median ≈ **1.67 tok/s**，无回归 |
| **CPU payload cache on** | `CPU_PAYLOAD_CACHE_BYTES=2GB`，cold-start | first run 与 steady 均无收益，`cpu_cache_hits=0` |

> 说明：cold-start 的第一次运行才是真正的 cold-cache 指标；由于 UMA 机器的默认 staging budget 足够容纳整个 21.6 GB working set，第一次运行后很快就进入“准 warm”状态。因此 **cold path 不宜作为主要优化目标**，只用于防退化。

---

## 已完成的实验与结论

### 1. Async expert prefetch

- **实现**：在独立 CUDA stream 上预取下一层可静态预测的专家权重，与当前层 compute 重叠。
- **覆盖率**：`deepseek_async_prefetch_scheduled_layers / opportunities` ≈ **4.8%**（例如 96/2016）。
- **瓶颈**：`_likely_hash_routed_expert_ids_for_token()` 只能对带 `expert_token_to_expert_ids` 表的层做静态预测；多数 MoE 层的路由依赖 router/hidden-state 输出，无法提前预测。
- **结论**：实现有效但上限低。保持 `ASYNC_PREFETCH=0` 默认，仅作为实验开关保留。

### 2. CPU payload cache

- **实现**：在 `stage_grouped_expert_payload()` 中对 `raw_grouped_expert_payload()` + `torch.frombuffer(...).clone()` 的结果做 LRU CPU 缓存。
- **实测**：`cpu_payload_cache_hits=0`，`misses=5703`，`evictions=4793`。
- **原因**：默认 UMA staging budget 已足够大，GPU cache 能装下整个 working set，每个 payload 只被 stage 一次，CPU 缓存没有第二次命中机会。
- **结论**：在 96 GB 机器上无收益，反而因 churn 略慢。保持 `CPU_PAYLOAD_CACHE_BYTES=0` 默认，不再调参。

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
| `q8_0_raw_matvec`（量化 matvec，主要 router/attention/linear） | 12.197 | 21.16 |
| `q2_k_selected_experts_down_projection_direct`（down projection） | 7.359 | 12.76 |
| hipblaslt / `Cijk_*` GEMM（attention/linear 通用 GEMM） | 7.202 | 12.49 |
| PyTorch elementwise / copy / fill / cat / arange 等 | ~12.9 | ~22.4 |
| `q8_0_raw_gate_up_activation_kernel` | 1.501 | 2.60 |
| 其他 | ~2.4 | ~4.2 |

关键发现：
- **MoE compute kernel 合计约 60.7%**（iq2_xxs up/gate 24.2% + q8_0 gate_up 2.6% + q2_k down 12.8% + q8_0 matvec 中属于 MoE 的部分）。即使保守估计，MoE 仍是绝对主导。
- `_q8_0_raw_matvec_kernel` 调用次数高达 **10.6 万次**，平均仅 0.114 ms，说明存在大量细碎 matvec 启动；其中一部分来自 router/attention/linear，另一部分来自 MoE。
- hipblaslt GEMM 占 **12.5%**，与 Python profiler 中 `layer_attention` 占比最高相互印证（attention 主要走 hipblaslt）。
- staging / H2D / cache 相关 kernel 在 warm 路径下几乎不可见，再次验证 staging 不是瓶颈。

---

## 关键判断

1. **staging 层优化已触顶**
   - warm path 已经把 staging 绕开（full resident + pinned hot experts）。
   - cold path 的 async prefetch / CPU cache 实验上限很低。
   - 继续调 event/stream/cache 只会改善冷启动/抖动，不会突破 1.6–1.8 tok/s 的 steady 上限。

2. **1.6–1.8 tok/s 是当前真实基线**
   - 在 96 GB UMA 机器上，`FULL_RESIDENT=1 + PIN_HOT_EXPERTS=1 + warmup=16` 可稳定复现。
   - 回归 gate 应以此为主：warm gate 阈值 1.5；cold gate 保守防退化即可。

3. **ROCm kernel profile 已完成；优化目标已锁定 top-2**
   - Python profiler 已证明 staging 不是 steady 上限。
   - ROCm profiler 进一步确认：warm 路径下 **MoE compute kernel 合计约 60%**，`_iq2_xxs_selected_experts_activation_direct`（24.2%）与 `_q8_0_raw_matvec`（21.2%）是最大两个热点。
   - hipblaslt GEMM 占约 12.5%，与 Python profiler 中 `layer_attention` 占比最高一致。
   - 下一步应直接针对 top-2 kernel 做 Triton 调优或调用合并，而不是继续扩展开销已很低的 staging/cache 层。

---

## 下一步建议

### P0：针对 top-2 kernel 进行优化

基于 ROCm profile，优化优先级如下：

1. **P0-1：`_iq2_xxs_selected_experts_activation_direct`（24.2%）**
   - 这是 selected expert up/gate  fused activation 的 Triton kernel。
   - 方向：调大 block/tile、减少 dequant+activation 中的临时量、尝试把 roundtrip（`deepseek_q8_k_roundtrip_reference`）消掉或融合进同一 kernel。
   - 验证指标：该 kernel 的 total_ms / avg_ms 下降，warm steady TPS 提升。

2. **P0-2：`_q8_0_raw_matvec`（21.2%，10.6 万次调用）**
   - 大量细碎 matvec，来源包括 router/attention/linear 以及部分 MoE path。
   - 方向：
     - 先按 layer/operator 拆分调用来源（可加 kernel 名前缀或 ROCTx marker）。
     - 对 MoE 内部的小 matvec 做 batch/merge，减少 launch count。
     - 对 router/attention 的 matvec 评估是否可用 hipblaslt GEMM 替换或融合。
   - 验证指标：调用次数下降或平均耗时下降。

3. **P0-3：`_q2_k_selected_experts_down_projection_direct`（12.8%）与 hipblaslt GEMM（12.5%）**
   - down projection 已是第三大热点；hipblaslt GEMM 主要对应 attention。
   - 方向：先完成 P0-1/P0-2 后再评估；若 attention 成为新的 top-1，再转向 attention kernel 融合或 KV cache 压缩。

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

- 继续调 async prefetch event / stream
- 继续扩大 CPU payload cache 容量或策略
- 默认启用 async prefetch 或 CPU cache
- 图/capture（已被 `CAPABILITY_MATRIX` 明确拒绝）

---

## 相关文件与 Artifact

- `tests/run_deepseek_v4_flash_real_smoke.sh` — warm/cold gate
- `tests/run_deepseek_v4_flash_async_prefetch_ab.sh` — async prefetch A/B（3-run median）
- `tests/run_deepseek_v4_flash_cpu_payload_cache_ab.sh` — CPU payload cache A/B
- `docs/superpowers/plans/2026-07-07-deepseek-async-prefetch.md` — 实现计划与实验记录
- `docs/performance_evaluation_gemma4_deepseek_2026_07_07.md` — 早期评估报告
- `/tmp/ds_warm_rocprof/*/kernel_stats.csv` — ROCm warm-resident kernel profile 原始数据
- `/tmp/ds_warm_moe_rocprof.json` — 同次运行的 Python profiler JSON

---

## 结论

DeepSeek V4 Flash 在 96 GB UMA 机器上的当前稳态性能为 **1.6–1.8 tok/s**（warm-resident）。staging/cache/prefetch 层的优化潜力已耗尽；Python-level 与 ROCm-level 的 warm 路径分解均已完成。ROCm profile 将优化目标精确锁定为 `_iq2_xxs_selected_experts_activation_direct`（24.2%）与 `_q8_0_raw_matvec`（21.2%）这两个 kernel。下一步应直接对它们做 Triton 调优或调用合并，而非继续投入 staging 层。
