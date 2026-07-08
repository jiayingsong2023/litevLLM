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

---

## 关键判断

1. **staging 层优化已触顶**
   - warm path 已经把 staging 绕开（full resident + pinned hot experts）。
   - cold path 的 async prefetch / CPU cache 实验上限很低。
   - 继续调 event/stream/cache 只会改善冷启动/抖动，不会突破 1.6–1.8 tok/s 的 steady 上限。

2. **1.6–1.8 tok/s 是当前真实基线**
   - 在 96 GB UMA 机器上，`FULL_RESIDENT=1 + PIN_HOT_EXPERTS=1 + warmup=16` 可稳定复现。
   - 回归 gate 应以此为主：warm gate 阈值 1.5；cold gate 保守防退化即可。

3. **Python-level MoE 分解已完成；下一轮必须看 warm path 的 GPU kernel breakdown**
   - Python profiler 已证明 staging 不是 steady 上限，并给出 `layer_attention` / `router_selected_experts_kernel` / `selected_expert_up_gate` 等子项占比。
   - 需要用 ROCm profiler 看 `layer_moe` 与 attention 的 GPU kernel 耗时：router、expert up/gate、down projection、shared expert、kernel launch、attention 等各自占比。
   - 只有 top-1/top-2 热点才值得写 Triton 或做融合。

---

## 下一步建议

### P0：ROCm warm-resident kernel profile

固定命令：

```bash
FASTINFERENCE_KV_TYPE=fp16 \
FASTINFERENCE_BLOCK_SIZE=32 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_FULL_RESIDENT=1 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_PIN_HOT_EXPERTS=1 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB=1 \
  rocprof --stats -o /tmp/ds_warm_rocprof.csv \
  uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
    --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
    --context-length 4096 \
    --prompt-length 32 \
    --max-tokens 16 \
    --warmup-tokens 16 \
    --repeat 3 \
    --min-steady-decode-tps 0.0
```

目标输出：
- top kernels（按总耗时排序）
- 每 kernel 的调用次数、平均耗时
- `layer_moe` 内部各子步骤（router、up/gate、down、shared）的 GPU 时间占比

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

---

## 结论

DeepSeek V4 Flash 在 96 GB UMA 机器上的当前稳态性能为 **1.6–1.8 tok/s**（warm-resident）。staging/cache/prefetch 层的优化潜力已耗尽；Python-level warm MoE breakdown 已完成并指向 `layer_attention` 与 `router_selected_experts_kernel` / `selected_expert_up_gate`。要突破该上限，必须基于 ROCm kernel profile 对 warm 路径的 GPU compute 进行针对性优化。
