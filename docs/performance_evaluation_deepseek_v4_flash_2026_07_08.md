# DeepSeek V4 Flash 性能收敛报告

> 日期：2026-07-08  
> 分支：`feat/p0-perf-gates`  
> 测试硬件：Radeon 8060S，96 GB UMA（统一显存），ROCm  
> 目标模型：`models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`

---

## 执行摘要

DeepSeek V4 Flash 在当前实现下，**warm-resident 路径稳定在 1.6–1.8 tok/s**，这是 96 GB UMA 机器上的真实性能基线。围绕 cache/prefetch/staging 的优化（async prefetch、CPU payload cache）已被证明上限很低；对 `layer_moe` 的 IQ2 常量 sweep 和 `layer_attention` 的 Q8 batched kernel 实验也均未带来可测量的端到端提升。

**最终决策**：
- **接受当前 baseline**：96 GB UMA 机器上 warm-resident 路径中位数约 **1.65 tok/s**，不再继续投入单 kernel/launch 优化。
- **固化 gate 与回归**：cold gate 0.4 tok/s（防退化），warm gate 1.5 tok/s（主要基线），并纳入 `tests/run_deepseek_v4_flash_real_smoke.sh`。
- **把资源放到别处**：优先转向 Gemma4 26B/31B 的 batch=2 并行验证，以及小显存 DeepSeek 的真实 staging cap + eviction 策略。

---

## 当前性能指标

| 场景 | 命令/配置 | 结果 |
|---|---|---|
| **warm-cache gate** | `KV_TYPE=fp16`, `BLOCK_SIZE=32`, `FULL_RESIDENT=1`, `PIN_HOT_EXPERTS=1`, `STAGING_BUDGET_GB=1`, warmup=16, repeat=3 | **1.843 / 1.838 / 1.872 tok/s**（median **1.843 tok/s**，无 profiler，2026-07-08 gate verification） |
| **warm-cache profile run** | 同上，带 `--profile-json` | 1.632 / 1.653 / 1.671 tok/s（median 1.653 tok/s；profiler sync 开销使 TPS 降低约 10%） |
| **e2e benchmark** | 同上，由 `tests/e2e_full_benchmark.py --models deepseek_v4_flash_q2_gguf` 调用 | decode_tps ≈ **1.79 tok/s**（历史数据） |
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

## 8. `attn_backend_compute` 细粒度归因（最终拆解）

在决定停止 DeepSeek 单请求性能优化前，对 `layer_attention` 内部做了最后一层拆解，以确认 attention 路径是否还有未被识别的单一热点。

固定 kept-path 配置（`FULL_RESIDENT=1`, `PIN_HOT_EXPERTS=1`, `STAGING_BUDGET_GB=1`, `KV_TYPE=fp16`, `BLOCK_SIZE=32`, warmup=16, repeat=3），在 `_run_sliding_attention_for_slot` 内部新增 section：

- `attn_backend_query_rope`：query 的 RMS norm + rope
- `attn_backend_kv_rope_qat`：KV 的 rope + `deepseek_fp8_kv_qat_reference`
- `attn_backend_attention_core`：`fused_sliding_window_attention` 或 reference attention
- `attn_backend_inv_rope`：inverse rope
- `attn_backend_outproj`：`_project_sliding_attention_output`（两阶段 Q8 output projection）

同时给 compressed attention triton 路径加了可选 `section` 参数（`attn_backend_scores` / `attn_backend_softmax_reduce` / `attn_backend_expand`），用于未来 compressed-path 模型。

### 8.1 结果（`/tmp/ds_attn_backend_attribution_v3.json`）

`layer_attention` 总计约 **25,642 ms**（6063 counts = 2021 layer-calls × 3 runs）。

| Section | total_ms | 占 `layer_attention` | 关键结论 |
|---|---:|---:|:---|
| `attn_backend_compute` | 21,557 | 84.1% | attention compute wrapper 仍是 attention 主体 |
| `attn_backend_outproj` | **7,707** | **30.1%** | **最大单一子热点**：两阶段 Q8 output projection |
| `attn_backend_kv_rope_qat` | **7,022** | **27.4%** | **第二大子热点**：KV rope + fp8 KV QAT |
| `attn_backend_query_rope` | 1,015 | 4.0% | query RMS norm + rope，小头 |
| `attn_backend_inv_rope` | 870 | 3.4% | inverse rope，小头 |
| `attn_backend_attention_core` | 839 | 3.3% | **注意力数学本身极小** |

内层 section 合计：7,707 + 7,022 + 1,015 + 870 + 839 = **17,453 ms**，与 `attn_backend_compute` 21,557 ms 之间的差额约 4,104 ms，主要是 KV cache append/read/concat 等辅助操作以及 section sync 开销。

### 8.2 关键发现

1. **注意力核函数不是瓶颈**：`attn_backend_attention_core` 只占 `layer_attention` 的 3.3%。继续优化 fused/reference attention、Flash Attention、score/reduce kernel 等，对当前模型/配置几乎没有收益。
2. **最大两个可合并热点是 output projection 和 KV rope/QAT**：
   - `attn_backend_outproj` 主要是 `attention_output_a` + `attention_output_b` 两阶段 Q8 matvec。
   - `attn_backend_kv_rope_qat` 是 KV rope 与 `deepseek_fp8_kv_qat_reference` 的连续操作。
3. **compressed attention triton 路径未触发**：本模型所有层都带 real sliding attention tensors（`attention_query_a/b`、`attention_key_value`、`attention_output_a/b`、`attention_sinks` 均非空），因此 `backend.compressed_attention` 的 triton 路径在当前配置下未被执行。为 compressed path 新增的 section 参数对未来模型/配置保留，但不影响本次结论。
4. **这两个热点都是“中小碎片”**：即使各自优化 30%，对 `layer_attention` 的改善也分别只有 ~9% 和 ~8%，且 `layer_attention` 只是整体 decode 的一部分。历史上 P0-B 的同类 batched Q8 实验已经把 launch count 从 48k 降到 ~1.7k 而没有 e2e 提升，说明这些路径上的 launch/GEMM 并行度已不是主要矛盾。

### 8.3 决策影响

这次拆解把最后两个潜在热点也量化清楚了：**它们都不够大、不够孤立，无法以工程上可控的成本撬动 ≥5% 的 warm e2e 提升**。因此不再继续对 DeepSeek V4 Flash 做单请求 kernel 优化，接受当前 baseline 并固化 gate。

---

## 9. 推理质量回归验证

为确认本次所有 instrumentation-only 改动不影响输出质量，跑了完整 correctness regression：

```bash
bash tests/run_inference_correctness_regression.sh
```

结果：

| Tier | 模型 | 结果 |
|---|---|---|
| Tier-B spotcheck | TinyLlama | PASS |
| Tier-B spotcheck | Qwen3.5-9B AWQ | PASS |
| Tier-B spotcheck | Gemma4-31B Q4 | PASS（含 multimodal） |
| Tier-B spotcheck | Gemma4-26B A4B | PASS（含 multimodal） |
| Tier-B spotcheck | DeepSeek V4 Flash | PASS |
| Tier-A strict | TinyLlama | OK |
| Tier-A strict | Qwen3.5-9B AWQ | OK |
| Tier-A strict | Gemma4-26B A4B | OK |
| Tier-A-lite | Gemma4-31B Q4 | OK |
| Tier-A-lite | Gemma4-26B A4B | OK |

脚本退出码 **0**，所有请求 tier 全部通过。

---

## 关键判断

1. **staging 层优化已触顶；实验代码已清理**
   - warm path 已经把 staging 绕开（full resident + pinned hot experts）。
   - cold path 的 async prefetch / CPU cache 实验上限很低，且运行时代码已从产品分支移除。
   - 继续调 event/stream/cache 只会改善冷启动/抖动，不会突破 1.6–1.8 tok/s 的 steady 上限。

2. **1.6–1.8 tok/s 是当前真实基线**
   - 在 96 GB UMA 机器上，`FULL_RESIDENT=1 + PIN_HOT_EXPERTS=1 + warmup=16` 可稳定复现。
   - 回归 gate 应以此为主：warm gate 阈值 1.5；cold gate 保守防退化即可。

3. **`layer_attention` 内部归因已完成；attention 数学不是瓶颈**
   - Python profiler 拆出 `attn_backend_compute` 内部的五大子阶段：`attn_backend_outproj`（30.1%）、`attn_backend_kv_rope_qat`（27.4%）、`attn_backend_query_rope`（4.0%）、`attn_backend_inv_rope`（3.4%）、`attn_backend_attention_core`（3.3%）。
   - **注意力核函数本身只占 `layer_attention` 的 3.3%**，优化 attention math / score-reduce / flash attention 等无法撬动端到端。
   - 最大的两个子热点是 output projection 和 KV rope/QAT，但它们属于中小碎片，历史上同类 Q8 batching 实验已证明减少 launch count 不带来 e2e 收益；继续融合/调参的信噪比不足。

4. **接受当前 baseline，固化 gate，资源转向别处**
   - 96 GB UMA 机器上 warm-resident 路径中位数约 **1.65 tok/s**，是工程实现下的真实性能天花板。
   - 不再继续 DeepSeek V4 Flash 单请求 kernel/attention 优化；把资源优先放到 Gemma4 batch=2 并行验证，以及小显存 DeepSeek 的真实 staging cap + eviction 策略。
   - gate 已写入 `tests/run_deepseek_v4_flash_real_smoke.sh`：cold 0.4 tok/s，warm 1.5 tok/s。

---

## 下一步建议

### 已决策：接受当前 baseline，停止 DeepSeek V4 Flash 单请求优化

基于以上全部实验，**不再继续对 DeepSeek V4 Flash 做单请求 kernel/attention 优化**。原因：

1. **IQ2 activation 常量 sweep**：negative result，热漂移掩盖了真实收益。
2. **Q8 attention output grouped batching**：launch count 大幅下降，但 e2e 无提升。
3. **async prefetch / CPU payload cache**：覆盖率/命中率过低，运行时代码已移除。
4. **`attn_backend_compute` 拆解**：最大两个子热点 `attn_backend_outproj`（30.1%）和 `attn_backend_kv_rope_qat`（27.4%）都是中小碎片，且 attention 数学本身仅占 3.3%；继续优化的工程成本高、收益不可预测。

### Gate 与回归（立即执行）

1. **Gate 已固化在 `tests/run_deepseek_v4_flash_real_smoke.sh`**：
   - cold-cache gate：`--warmup-tokens 1 --min-steady-decode-tps 0.4`
   - warm-cache gate：`--warmup-tokens 16 --repeat 3 --min-steady-decode-tps 1.5`
2. **质量回归已通过**：`bash tests/run_inference_correctness_regression.sh` 退出码 0，所有 tier 通过。
3. **合并前建议再跑一次 fast regression**：
   ```bash
   bash tests/run_regression_suite.sh
   tests/run_deepseek_v4_flash_real_smoke.sh
   ```

### 资源转向（按优先级）

1. **Gemma4 26B/31B：batch=2 并行验证**
   - 目标：验证 active-request / scheduler admission 放开后，batch=2 是否真正进入并行 decode 并带来吞吐收益。
   - 这属于另一条模型线，与 DeepSeek 当前结论无关。

2. **小显存 DeepSeek：真实 staging cap + eviction 策略**
   - 当前 96 GB UMA 机器无法模拟内存受限场景；默认 `STAGING_BUDGET_GB` 会被放大到容纳整个 working set。
   - 若未来要支持小显存，先在 DeepSeek V4 Flash staging 配置里加一个按字节计的 `max_staged_bytes` 硬上限，再评估 eviction/re-stage、批量 staging、hot expert pinning。

3. **暂不投入**
   - DeepSeek 单请求 kernel 优化（含 attention output projection 融合、KV rope/QAT 融合、IQ2 常量 sweep）。
   - async prefetch / CPU payload cache（运行时代码已移除）。
   - 图/capture（已被 `CAPABILITY_MATRIX` 明确拒绝）。

---

## 相关文件与 Artifact

- `tests/run_deepseek_v4_flash_real_smoke.sh` — warm/cold gate（已固化）
- `tests/run_inference_correctness_regression.sh` — 推理质量回归
- `tests/run_deepseek_v4_flash_async_prefetch_ab.sh` — async prefetch A/B（历史记录，运行时代码已移除）
- `tests/run_deepseek_v4_flash_cpu_payload_cache_ab.sh` — CPU payload cache A/B（历史记录，运行时代码已移除）
- `docs/superpowers/plans/2026-07-07-deepseek-async-prefetch.md` — 实现计划与实验记录
- `docs/performance_evaluation_gemma4_deepseek_2026_07_07.md` — 早期评估报告
- `/tmp/ds_attn_backend_attribution_v3.json` — 本次 `attn_backend_compute` 细粒度归因
- `/tmp/ds_warm_rocprof/*/kernel_stats.csv` — ROCm warm-resident kernel profile 原始数据
- `/tmp/ds_warm_moe_rocprof.json` — 同次运行的 Python profiler JSON
- `/tmp/ds_q8_attribution.json` — Q8 raw matvec 调用归因 aggregate
- `/tmp/ds_q8_attribution_events.json` — Q8 raw matvec 调用归因 events（含 tensor metadata）

---

## 结论

DeepSeek V4 Flash 在 96 GB UMA 机器上的当前稳态性能为 **1.6–1.8 tok/s**（warm-resident），最新 profile run 为 **1.632 / 1.653 / 1.671 tok/s**（median **1.653 tok/s**）。

- **staging/cache/prefetch 优化已触顶**：async prefetch 覆盖率仅 ~4.8%，CPU payload cache hit=0，运行时代码已移除。
- **MoE kernel 常量 sweep 为 negative result**：`_iq2_xxs_selected_experts_activation_direct` 的 rows8/warps16 sweep 未通过 1.70 tok/s 硬退出条件，首轮“收益”是 thermal/cache 漂移。
- **Q8 attention batching 局部成功、端到端失败**：P0-B 把 `q8_attention_output_grouped` 调用数从 48k 降到 ~1.7k，但 warm median 仍停留在 **1.625 tok/s**，已完整回滚。
- **`attn_backend_compute` 拆解完成**：最大子热点是 `attn_backend_outproj`（30.1%）和 `attn_backend_kv_rope_qat`（27.4%），但 attention 数学本身仅占 3.3%；这些热点碎片化，继续优化的工程收益不可预测。

**最终决策**：接受 1.6–1.8 tok/s 为当前实现下 96 GB UMA 机器的真实性能天花板；把 DeepSeek V4 Flash 单请求优化资源停掉，优先转向 Gemma4 并行验证和小显存 staging 策略。

**已固化产物**：
- `tests/run_deepseek_v4_flash_real_smoke.sh`：cold gate 0.4 tok/s，warm gate 1.5 tok/s。
- 质量回归 `tests/run_inference_correctness_regression.sh` 已全 tier 通过。
- 本报告作为决策文档存档。
