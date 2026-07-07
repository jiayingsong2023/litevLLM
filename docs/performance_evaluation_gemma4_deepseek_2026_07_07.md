# FastInference 推理性能深度评估：Gemma4 26B/31B 与 DeepSeek V4 Flash

> 评估日期：2026-07-07  
> 评估人：Kimi Code CLI  
> 目标硬件：AMD Ryzen AI MAX+ 395 / Radeon 8060S Graphics（gfx1151），96 GB 统一内存  
> 运行环境：ROCm，Python 3.12，`uv` 虚拟环境

> **更新（2026-07-08）**：DeepSeek V4 Flash 的后续实验与收敛结论已写入 `docs/performance_evaluation_deepseek_v4_flash_2026_07_08.md`。staging/cache/prefetch 优化已证明上限很低；下一步需对 warm-resident 路径做 ROCm kernel profile。

---

## 1. 执行摘要

本评估针对 `FastInference` 项目中用户最关心的三个模型——**Gemma4-26B-A4B（Q4 MoE）**、**Gemma4-31B-Q4（dense）**、**DeepSeek V4 Flash Q2 GGUF**——在 Radeon 8060S 统一内存平台上做了端到端基准测试、Python 级 profiling 与内存/算力 roofline 估算。

**核心结论：**

| 模型 | 实测 decode TPS | 主要瓶颈 | 与硬件上限关系 |
|---|---|---|---|
| Gemma4-26B-A4B | **11.4 tok/s**（batch=1） | 单请求小 GEMV、kernel launch/同步；batch=2 因 AWQ decode GEMV / gate-up fusion 仅优化 M=1 而退化 | 远低于内存带宽墙，M=1 kernel specialization 是瓶颈 |
| Gemma4-31B-Q4 | **3.7 tok/s** | 内存带宽墙（dense 60 层大权重读取） | 已接近统一内存有效带宽上限；KV/batch/warmup 均不敏感 |
| DeepSeek V4 Flash | cold-cache **0.46 tok/s**，kept-path warm-cache **1.6–1.63 tok/s** | cold run 未命中专家缓存；warm run 复现 `CAPABILITY_MATRIX` 记录的 1.6–1.9 tok/s | 离硬件上限仍远，但需先固化 warm-cache gate 防止回归 |

**最大机会点：**
1. **Gemma4-31B**：已触碰内存墙，短期靠有效连续批处理/投机采样降低单 token 读取量，中长期靠权重量化。KV 精度切换几乎无效。
2. **Gemma4-26B**：远离内存墙，但 batch=2 严重退化。根因不是 scheduler admission：engine 已将两个请求放入同一 decode step，但 `vllm/model_executor/models/gemma4/attention.py` 的 `packed_int4_symmetric_fused_qkv_m1_safe` 与 `_fused_awq_pair.py` 的 `packed_int4_symmetric_fused_gate_up_m1_safe` 仅对 M=1 做了 hand-optimized kernel；batch>1 时这些 fusion 失效并 fallback 到较慢路径。让 batch=2 真正生效需要 **batched AWQ decode kernel**（或投机采样），不是改 env/scheduler。
3. **DeepSeek V4 Flash**：cold-cache 0.46 tok/s 与 kept-path warm-cache 1.60–1.63 tok/s 差距来自专家缓存是否命中。P0 是固化 kept-path 命令/artifact/profile counters，并把现有 `--min-steady-decode-tps` gate 分 cold/warm 阈值接入回归；graph/capture 是 rejected experiment，不作为恢复项。

---

## 2. 硬件与软件基线

### 2.1 GPU 规格（来自 `rocminfo`）

```text
Marketing Name:  Radeon 8060S Graphics
Name:            gfx1151
Compute Unit:    40
Max Clock:       2900 MHz
Wavefront Size:  32
Pool Size:       ~96 GB unified memory
```

**粗略峰值估算（仅供参考）：**
- FP16 算力：40 CU × 64 ALU × 2 ops/FMA × 2.9 GHz ≈ **15 TFLOPS**（保守）到 ~30 TFLOPS（理想 packed）。
- 有效内存带宽：LPDDR5X 统一内存，GPU 可分到约 **60–120 GB/s**（受 CPU、北桥、page 占用影响）。

### 2.2 关键软件配置

- PyTorch ROCm / Triton（通过 `vllm/triton_utils`）。
- KV 缓存默认 `turbo_int4`，但 **Gemma4 adapter 会强制回退到 fp8**（精度保护），除非显式设置 `FASTINFERENCE_GEMMA4_ALLOW_INT4_KV=1`。
- 所有三个模型在 e2e 中默认 `concurrent_reqs=1`，即单请求 decode，吞吐等于单 token 延迟的倒数。

---

## 3. 端到端测量数据

### 3.1 本次实测（`tests/e2e_full_benchmark.py`）

| 模型 | prompt tokens | max_new_tokens | TTFT (ms) | Prefill TPS | Decode TPS | 总显存占用 |
|---|---|---|---|---|---|---|
| Gemma4-26B-A4B | ~394 | 24 | 2252 | 175.0 | **11.42** | ~17.0 GB |
| Gemma4-31B-Q4 | ~394 | 24 | 5475 | 72.0 | **3.72** | ~22.8 GB |

> 注：batch=1 fp8 warm 取自 §3.4 A/B 表同一命令。

> 注：实际 KV 类型为 fp8（e4m3fn），因为 Gemma4 accuracy guard 强制覆盖。

### 3.2 历史/临时数据对比（命令不统一，仅作参考）

以下数据来自不同上下文长度和不同命令，**不能直接与 §3.1/§3.4 的 A/B 结果做定量比较**：

- `.tmp_perf_gemma_1024.json`（更长上下文 1536/1024 tokens，具体命令未记录）：
  - Gemma4-26B：decode TPS **11.81**
  - Gemma4-31B：decode TPS **3.43**
- `.tmp_perf_regression_baseline.json` / `post_fix.json`（短上下文 512/24 tokens，具体命令/warmup/artifact 未记录）：
  - Gemma4-26B：9.4 → 8.9 tok/s
  - Gemma4-31B：3.14 → 2.99 tok/s

**可比较的固定命令结果在 §3.4 A/B 表**：所有行均来自 `tests/e2e_full_benchmark.py`，prompt ~394 tokens、max_new_tokens=24、默认 scheduler profile、JSON artifact 位于 `/tmp/fastinference_eval_ab/e2e_*.json`。

### 3.3 DeepSeek V4 Flash 实测

#### cold-cache（旧命令，仅作对比）

```bash
FASTINFERENCE_KV_TYPE=fp16 \
FASTINFERENCE_BLOCK_SIZE=32 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_PIN_HOT_EXPERTS=1 \
uv run python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --context-length 4096 \
  --max-tokens 16 \
  --warmup-tokens 1 \
  --profile-json /tmp/ds_cold.json
```

| 指标 | 值 |
|---|---|
| Decode TPS (aggregate) | **0.46 tok/s** |
| Decode TPS (steady state) | **0.46 tok/s** |
| `full_resident_enabled` | 0 |
| `staged_bytes` / `max_staged_bytes` | 21.6 GB / 90.3 GB |
| `grouped_misses` | 5703 |

#### kept-path warm-cache（current e2e 命令）

`tests/e2e_full_benchmark.py` 对 DeepSeek V4 Flash 使用的 kept-path 命令如下（已固化在 `_run_deepseek_v4_flash_direct_benchmark`，见 `tests/e2e_full_benchmark.py:2868-2876`）：

```bash
FASTINFERENCE_KV_TYPE=fp16 \
FASTINFERENCE_BLOCK_SIZE=32 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_FULL_RESIDENT=1 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_PIN_HOT_EXPERTS=1 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB=1 \
uv run python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --context-length 4096 \
  --prompt-length 32 \
  --max-tokens 16 \
  --warmup-tokens 16 \
  --repeat 3 \
  --profile-json /tmp/ds_warm.json
```

结果（连续 3 次独立进程运行，每次 `--repeat 3`，共 9 个 steady-state 样本）：

| Run | decode_tps_steady_state（3 repeats） | mean | 说明 |
|---|---|---|---|
| #1 | 1.62 / 1.60 / 1.63 | **1.62** | 三次 repeat 均 warm |
| #2 | 1.62 / 1.61 / 1.61 | **1.61** | 稳定 |
| #3 | 1.60 / 1.62 / 1.63 | **1.62** | 稳定 |
| **汇总** | min=1.60, max=1.63 | **1.62** | 落在 `CAPABILITY_MATRIX` 1.6–1.9 区间下沿 |

> 注：一次独立快速测试曾测到 1.75–1.79 tok/s；本次 3-run 复现为 1.60–1.63。两者均在 1.6–1.9 区间内，说明 warm-cache 基线可复现，但存在 ~10% 的 run-to-run 波动。本次 3-run artifact：`/tmp/fastinference_eval_ab/deepseek_warm_run{1,2,3}.json`。

`CAPABILITY_MATRIX.md:37` 记录 warm-cache 目标为 **1.6–1.9 tok/s**，来源是 kept optimizations：direct selected Q2/IQ2 payload kernels、Q8_0 raw sign-extension trim、fused indexer-select scale、Triton indexer QAT（`CAPABILITY_MATRIX.md:59`）。rejected experiments 明确包含 graph/capture、full expert GPU tables 等。

**重要：0.46 tok/s 与 1.60–1.63 tok/s 的差距来自专家/权重缓存是否命中，而不是 `STAGING_BUDGET_GB=1` 的因果效应。** 默认 UMA budget 在 96 GB 系统上会自动放大到 `total_memory - 8 GB`（`vllm/model_executor/models/deepseek_v4_flash/config.py:11-38`），默认 expert cache 为 2 GB（`config.py:122`）。`STAGING_BUDGET_GB=1` 只是当前 e2e 命令中的一个固定参数，没有证据表明它是性能差异的 root cause。

### 3.4 Gemma4 A/B 表（batch / KV dtype / warmup）

固定命令：`tests/e2e_full_benchmark.py --models <model> --gemma26b-concurrent <n> --gemma31b-concurrent <n> --json-out <artifact>`。变量通过环境变量控制：KV dtype 由 `FASTINFERENCE_GEMMA4_ALLOW_INT4_KV={unset,1}` 决定（fp8 vs int4）；warmup 由 `FASTINFERENCE_BENCH_WARMUP_PRESET={default,off}` 决定。

| 模型 | batch | KV dtype | warmup | decode TPS (agg) | TTFT (ms) | 备注 |
|---|---|---|---|---|---|---|
| Gemma4-26B-A4B | 1 | fp8 | warm | **11.42** | 2252 | 默认 e2e，int4 guard 关闭 |
| Gemma4-26B-A4B | 1 | int4 | warm | **11.17** | 2265 | `ALLOW_INT4_KV=1` |
| Gemma4-26B-A4B | 2 | fp8 | warm | **1.29** | 3523 | 并发=2 严重退化；engine 已同时调度两个请求，但 AWQ M=1 kernel 失效 |
| Gemma4-26B-A4B | 2 | int4 | warm | **4.25** | 3530 | 并发=2 仍退化；int4 KV 降低显存压力，但权重 GEMV batch 路径仍是瓶颈 |
| Gemma4-26B-A4B | 1 | fp8 | cold | **11.24** | 3299 | `WARMUP_PRESET=off` |
| Gemma4-31B-Q4 | 1 | fp8 | warm | **3.72** | 5475 | 默认 e2e |
| Gemma4-31B-Q4 | 1 | int4 | warm | **3.73** | 5484 | `ALLOW_INT4_KV=1`，与 fp8 几乎无差别 |
| Gemma4-31B-Q4 | 2 | fp8 | warm | **1.04** | 8481 | 并发=2 严重退化；权重读取已接近内存墙，batch 收益有限 |
| Gemma4-31B-Q4 | 1 | fp8 | cold | **3.62** | 6835 | `WARMUP_PRESET=off` |

**关键发现：**
1. **KV dtype 对 decode TPS 几乎无影响**：26B fp8 11.42 vs int4 11.17；31B fp8 3.72 vs int4 3.73。这说明权重读取仍是主导，KV 缓存精度不是当前瓶颈。
2. **31B 对 batch/KV/warmup 都不敏感**：3.62–3.73 tok/s 范围极窄，支持**内存墙**判断。
3. **26B batch=1 稳定 ~11.3 tok/s，但 batch=2 不升反降**：初步假设是 `FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS=1` 或 scheduler admission 限制。后续验证（见 §7/P0-3）表明，即使把 runtime policy 切到 aggressive、把 active request cap 放开到 2/4，engine 仍会把两个请求放入同一 decode step（`steps=37` 处理 46 tokens，远高于串行的 68 步），但 decode TPS 仍只有 ~4.2。代码检查确认：`vllm/model_executor/models/gemma4/attention.py:86` 的 `packed_int4_symmetric_fused_qkv_m1_safe` 与 `vllm/model_executor/models/_fused_awq_pair.py:225` 的 `packed_int4_symmetric_fused_gate_up_m1_safe` 都明确要求 `a.shape[0] == 1`；batch>1 时这些 hand-optimized M=1 kernel 失效并 fallback 到较慢路径。
4. **warmup 对 Gemma4 影响有限**：cold vs warm 差异 < 5%，说明 Triton compile cache 已命中或编译时间占比小。

---

## 4. Roofline / 带宽估算

下表估算单次 decode token 需要从显存读取的权重大小（忽略激活、output projection、attention softmax 中间结果）。

| 模型 | MLP/MoE 权重读取 (GB) | Attention 权重读取 (GB) | KV 读取 (GB) | 合计 (GB) | 实测 TPS | 等效带宽 (GB/s) |
|---|---|---|---|---|---|---|
| Gemma4-26B-A4B | 0.71 | 0.48 | 0.05 | **1.24** | 11.42 | **14.2** |
| Gemma4-31B-Q4 | 10.40 | 3.47 | 0.20 | **14.07** | 3.72 | **52.3** |
| DeepSeek V4 Flash | 1.62 | 0.72 | 0.01 | **2.35** | 1.62 (warm) | **3.8** |

**解读：**

- **Gemma4-31B**：等效 ~52 GB/s，已接近统一内存 GPU 分到的有效带宽上限。decode 瓶颈是**内存墙**——每 token 需要把 14 GB 量化权重从显存读到 CU。要显著提升，必须减少单 token 读取量（更大 batch、更低位宽、稀疏/压缩）或提高有效带宽。
- **Gemma4-26B**：等效仅 ~15 GB/s，远低于内存墙。MoE 每次只激活 top-8 expert，单 token 读取仅 1.2 GB，但 decode TPS 只有 12。瓶颈更可能是**小 batch GEMV 利用率低、kernel launch 次数多、MoE 专家切换/缓存抖动**。
- **DeepSeek V4 Flash**：权重读取量本身很小（Q2/IQ2 ~2 bit），但 TPS 极低。瓶颈显然不是原始权重带宽，而是**专家 staging、CPU/GPU 同步、kernel launch overhead、压缩 attention 两段式执行**。

---

## 5. 模型级瓶颈分析

### 5.1 Gemma4-31B-Q4：内存墙主导的 dense 模型

**模型特征：**
- 60 层 dense，hidden=5376，intermediate=21504。
- AWQ INT4 量化，head_dim=256，本地/全局混合 attention。
- 每层 MLP 权重约 173 MB，60 层约 **10.4 GB**。

**瓶颈证据：**
- decode TPS 仅随模型大小线性下降（26B 11.4 tok/s → 31B 3.7 tok/s），与权重读取量成正比。
- A/B 表显示 31B 对 KV dtype（fp8 vs int4：3.72 vs 3.73）、warmup（warm vs cold：3.72 vs 3.62）均不敏感；batch=2 退化部分来自内存墙（aggregate 仅 1.04，远低于 2× 3.72），也受 AWQ M=1 kernel fallback 影响。
- AWQ fused GEMM 命中率 100%，说明 kernel 选择正确，但 MLP 仍是主导。
- `FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ`、`FASTINFERENCE_AWQ_GROUP32_GEMV_ALL` 等开关已开启，31B 已走专用 GEMV tile。

**距离硬件上限：**
- 若有效带宽 80 GB/s，理论 decode TPS 上限 ≈ 80 / 14 ≈ **5.7 tok/s**。
- 当前 3.7 tok/s 相当于 ~65% 带宽利用率，再提升空间有限，除非降低读取量（更大 batch 有效时、更低 bit 权重量化、投机采样）。

### 5.2 Gemma4-26B-A4B：算力/同步主导的 MoE 模型

**模型特征：**
- 30 层，hidden=2816，128 experts，top-8，moe_intermediate=704。
- 每 token 激活专家权重约 0.7 GB（INT4），注意力权重约 0.5 GB。

**瓶颈证据：**
- 带宽需求仅 ~15 GB/s，远低于硬件上限，说明不是内存墙。
- A/B 表显示 KV dtype 变化几乎不影响 TPS（fp8 11.42 vs int4 11.17），权重读取不是瓶颈。
- **batch=2 严重退化**（fp8 1.29 / int4 4.25）。验证显示 engine 已把两个并发请求放入同一 decode step，但 `packed_int4_symmetric_fused_qkv_m1_safe` / `packed_int4_symmetric_fused_gate_up_m1_safe` 仅对 M=1 优化；batch>1 时这些关键 fusion 失效，fallback 到较慢的通用路径。
- `moe_int4_kernel` 已启用，但单请求 decode 中每个 expert GEMV 很小（2816×704），无法喂饱 40 CU；30 层 × top-8 = 240 个专家矩阵切换/launch 每 token。

**距离硬件上限：**
- 若以 31B 的 65% 带宽利用率作为参照，26B 若完全消除同步/launch overhead 并有效 batch，在 15 GB/s 需求下理论 TPS 上限 ≈ 80 / 1.24 × 0.65 ≈ **42 tok/s**。
- 实际 batch=1 仅 11.4 tok/s，且 batch=2 不升反降，说明主要瓶颈是**M=1 AWQ kernel specialization 与 MoE launch/sync overhead**，不是算力或纯内存带宽。让 batch=2 真正生效需要 batched AWQ decode kernel。

### 5.3 DeepSeek V4 Flash：同步与实现 overhead 主导

**模型特征：**
- 43 层，hidden=4096，256 experts，top-6，expert_intermediate=2048。
- Q2_K / IQ2_XXS 量化，压缩 attention（ratio 4/128），sliding window=128。
- Adapter-owned direct GGUF runtime，非 LiteEngine 标准路径。

**kept-path warm-cache profile 关键数据（`FULL_RESIDENT=1`、`PIN_HOT_EXPERTS=1`、warmup=16）：**

> 注：total_ms 是该 section 在整个 decode 过程中的累计耗时（多次调用求和）；avg_ms 是 `total_ms / count`，即单次调用的平均耗时，不是“每 token”。decode 阶段共 16 个 token，43 层，因此 attention/MoE/router 等每层每 token 的调用次数约为 688（≈43×16）。

| Section | count | total_ms (累计) | avg_ms (单次) | 占比 |
|---|---|---|---|---|
| `layer_moe` | 688 | 8923 | 12.97 | **51.1%** |
| `moe_routed_experts` | 640 | 7641 | 11.94 | 43.8% |
| `router_expert_stage` | 688 | 6243 | 9.07 | 35.8% |
| `layer_attention` | 688 | 2658 | 3.86 | 15.2% |
| `router_selected_experts_kernel` | 688 | 1834 | 2.67 | 10.5% |
| `compressed_kv_update` | 656 | 1240 | 1.89 | 7.1% |
| `output_projection` | 16 | 97 | 6.07 | 0.6% |

| Counter | warm-cache | cold-cache（旧命令） | 说明 |
|---|---|---|---|
| `full_resident_enabled` | 1 | 0 | warm run 启用全驻留专家表 |
| `pinned_entries` | 4560 | 54 | hot experts 被 pin 到 GPU |
| `prefetch_hits` / `prefetch_misses` | 6462 / 348 | 165 / 498 | warm run 预取命中率显著提高 |
| `staged_bytes` | 21.6 GB | 21.6 GB | 两次 staged 数据量相近，说明差异在缓存命中而非 staged 总量 |
| `streamed_bytes` | 0 | 0 | 无 streaming，已走缓存路径 |

**关键发现：**
1. **warm-cache 已复现 1.60–1.63 tok/s，落在 `CAPABILITY_MATRIX.md:37` 的 1.6–1.9 区间下沿**。这说明 kept optimizations（direct selected Q2/IQ2 payload kernels、Q8_0 raw sign-extension trim、fused indexer-select scale、Triton indexer QAT）已生效。
2. **cold-cache 0.46 tok/s 与 warm-cache 1.60–1.63 tok/s 的差距来自专家缓存命中**：cold run `grouped_misses=5703`、`pinned_entries=54`；warm run `pinned_entries=4560`、`prefetch_hits=6462`。`STAGING_BUDGET_GB=1` 只是当前 e2e 命令的固定参数，不是已证的 root cause。
3. **`router_expert_stage` + `router_selected_experts_kernel` 在 warm run 仍占 ~46%**，说明即使缓存命中，专家选择/调度仍是最大开销；实际 GEMM（`moe_routed_experts`）占 ~44%。
4. **Graph/capture 是 rejected experiment**：`CAPABILITY_MATRIX.md:59` 明确把 graph/capture 列为 rejected；当前 `use_graph=false` 是正确默认，不应作为短期恢复项。

**距离硬件上限：**
- 即使按保守 5 GB/s 有效权重带宽，Q2 量化下理论 decode TPS 也应达数 tok/s；warm-cache 1.60–1.63 tok/s 仍主要由 Python-level staging/launch overhead 限制，距离硬件上限仍远。

---

## 6. Kernel 级 Profiling（rocprofv3）

对 DeepSeek V4 Flash 跑了 `rocprofv3 --kernel-trace --stats`（4 tokens decode）。总 GPU kernel 时间约 **2425.6 ms**，Top kernel 如下：

| Kernel | 总耗时 (ms) | 占比 | Calls | 平均 (us) | 说明 |
|---|---|---|---|---|---|
| `_iq2_xxs_selected_experts_activation_direct6_kernel` | 817.2 | **33.7%** | 215 | 3800.8 | 选中专家的 gate/up 激活 |
| `_q8_0_raw_matvec_kernel` | 456.4 | **18.8%** | 2835 | 161.0 | Q8_0 主线性层 / shared expert |
| `_q2_k_selected_experts_down_projection_direct6_kernel` | 324.1 | **13.4%** | 215 | 1507.3 | 选中专家的 down projection |
| `Cijk_Alik_Bljk_...` (rocblas GEMM) | 226.7 | 9.3% | 835 | 271.4 | PyTorch/rocBLAS GEMM 回退 |
| PyTorch `reduce_kernel` / `elementwise_kernel` 等 | ~300 | ~12% | 数万 | <10 | Norm、softmax、clamp、scale 等逐元素/归约 |
| `_q8_0_raw_gate_up_activation_kernel` | 52.9 | 2.2% | 215 | 246.3 | Q8_0 gate/up 融合激活 |
| `__amd_rocclr_copyBuffer` | 29.1 | 1.2% | 7785 | 3.7 | 显存拷贝 |

**关键结论：**
1. **MoE 相关 Triton kernel 合计占 ~66%**，是 decode 的绝对热点；`direct6` 优化已经落地（每次调用约 3.8 ms / 1.5 ms），但单 token 仍需 215 次调用（43 层 × 5 token 平均），launch 与 kernel 内小矩阵利用率是主要限制。
2. **Q8_0 raw matvec 占 18.8%**，调用次数高达 2835 次，说明 shared expert / linear 层仍是重量级非 MoE 开销。
3. **PyTorch 级 reduce/elementwise 占 ~12%**（数万次小 kernel），主要来源是 RMSNorm、softmax、clamp、scale、argmax。这些适合进一步融合进相邻 kernel。
4. **显存拷贝仅占 1.2%**，说明 `direct6` 已经去掉了大部分 payload stack copy；staging 瓶颈更多体现在 Python-level 的 `router_expert_stage`（占时 36.5%）而非纯 H2D 拷贝。

对 Gemma4 未单独跑 rocprofv3（其 AWQ fused GEMM 命中率已达 100%，主要问题是 31B 内存墙、26B 小 GEMV），如需进一步 tile 调优，可用 `benchmarks/kernels/benchmark_moe.py` 与 `perf_grid_search.py` 做离线扫描。

---

## 7. 下一步：先固化基线与回归 gate，再谈优化方向

§3.3 和 §3.4 已经给出可复现的 DeepSeek 3-run warm-cache 基线与 Gemma4 A/B 证据；当前最缺的是**把 DeepSeek 基线固化为回归 gate**，并**正确理解 Gemma4 batch=2 退化的根因**。建议先完成以下三件事：

### P0-1：固化 DeepSeek kept-path warm-cache gate

当前 `tests/e2e_full_benchmark.py` 已经固化的 kept-path 命令（见 `tests/e2e_full_benchmark.py:2868-2876`）：

```bash
FASTINFERENCE_KV_TYPE=fp16 \
FASTINFERENCE_BLOCK_SIZE=32 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_FULL_RESIDENT=1 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_PIN_HOT_EXPERTS=1 \
FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB=1 \
uv run python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --context-length 4096 \
  --prompt-length 32 \
  --max-tokens 16 \
  --warmup-tokens 16 \
  --repeat 3 \
  --profile-json /tmp/ds_warm_run${i}.json
```

- 同一命令连续运行 3 次（每次 `--repeat 3`），记录 `decode_tps_steady_state` 与 `gpu_backend`/`gpu_staging`/`usable_inference_metrics` counters。
- 已验证该命令在 96 GB Radeon 8060S 上可达 **1.60–1.63 tok/s**（单次快速测试曾到 1.75–1.79），落在 `CAPABILITY_MATRIX.md:37` 的 1.6–1.9 区间内。
- **不要提 graph**：`CAPABILITY_MATRIX.md:59` 已把 graph/capture 列为 rejected experiment。

### P0-2：把现有 `--min-steady-decode-tps` 接入回归，分 cold/warm 阈值

`tests/tools/run_deepseek_v4_flash_gpu_smoke.py:65` 已提供 `--min-steady-decode-tps`，`validate_steady_decode_tps` 在 `line 113` 已实现。建议：

- 在 `tests/run_deepseek_v4_flash_real_smoke.sh` 或 `tests/e2e_full_benchmark.py` 中固化两条命令：
  - **cold-cache gate**：`--warmup-tokens 1 --min-steady-decode-tps 0.4`（当前 cold 实测 0.46，留 0.4 作为回归下限，避免基线直接失败）
  - **warm-cache gate**：`--warmup-tokens 16 --min-steady-decode-tps 1.5`（与 §3.3 kept-path 命令一致；确认 `--warmup-tokens 4` 也稳定后再降）
- 保存 JSON artifact 到固定路径，便于 CI 对比 counters。

### P0-3：Gemma4 batch=2 无法通过 env/scheduler 修复（blocked by M=1 AWQ kernels）

已完成验证，结论：**不是 scheduler/active-request 限制，而是当前 AWQ decode kernel 仅对 M=1 做了 hand-optimized fusion**。

验证过程：

1. **移除 `FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS=1`**：从 `_GEMMA4_31B_RECOMMENDED_ENV` / `_GEMMA4_26B_RECOMMENDED_ENV` 中删除该行，26B batch=2 decode TPS 仍为 4.23（与保留 =1 时几乎相同）。
2. **强制 aggressive execution policy**：在 `_build_vllm_config` 中把 `runtime_policy_mode` 设为 `aggressive`，26B batch=2 仍为 3.88（无提升），31B batch=2 从 1.04 微升到 1.40，仍远低于内存墙下的 7.4。
3. **运行时指标**：engine 分配 16 seq slots，`steps=37` 处理 46 tokens，说明两个请求已被同时调度，而不是串行执行。
4. **代码检查**：
   - `vllm/model_executor/models/gemma4/attention.py:86` 的 `packed_int4_symmetric_fused_qkv_m1_safe` 要求 `a.shape[0] == 1`；
   - `vllm/model_executor/models/_fused_awq_pair.py:225` 的 `packed_int4_symmetric_fused_gate_up_m1_safe` 同样要求 M=1；
   - batch>1 时这两个关键 fusion 失效并 fallback 到较慢路径。

决策：
- **不修改 Gemma4 默认 recommended env**（已把试验性改动 revert）。
- **P0-3 关闭**：在现有代码路径下，batch=2 没有可预期的吞吐收益；真正的下一步是 **batched AWQ decode kernel** 或 **投机采样**，属于 kernel/算法优化，不在当前 P0 范围内。
- 已把试验用的 `FASTINFERENCE_BENCH_POLICY_MODE` 覆盖逻辑从 `tests/e2e_full_benchmark.py` 中移除，保持 benchmark 代码不变。

### 其他方向暂缓

- **Gemma4 batch=2 优化**：已确认需要 batched AWQ decode kernel（或投机采样），不是 env/scheduler 调整。这项工作应作为独立 kernel 项目立项，先写微基准验证 M=2/M=4 路径的收益，再决定是否投入完整实现。
- **31B 内存墙**：短期靠投机采样或更低 bit 权重量化；在 batch>1 的 AWQ kernel 未就绪前，单纯提高并发不会带来预期收益。
- **DeepSeek 新 kernel**：在 warm-cache gate 已固化前，不宜扩大投入。

---

## 8. 风险与验证建议

1. **DeepSeek 归因风险**：`STAGING_BUDGET_GB=1` 与 0.46 tok/s 只是相关性，不是因果。默认 UMA budget 在 96 GB 系统已自动放大，expert cache 默认 2 GB，必须先复现 warm-cache 才能定位漂移。
2. **Gemma4 A/B 已完成并更新**：§3.4 表确认 31B 内存墙；26B batch=2 退化**不是** `KV_MAX_ACTIVE_REQUESTS=1` 或 scheduler admission 导致。验证显示 engine 已同时调度两个请求，但 AWQ decode GEMV / gate-up fusion 仅优化 M=1，batch>1 时 fallback。下一步是 kernel 项目（batched AWQ）而非 env/scheduler 调参。
3. **Graph/capture 不是选项**：`CAPABILITY_MATRIX.md:59` 已明确拒绝；继续投入 graph 会偏离 kept path。
4. **ROCm Triton 编译时间长**：任何新 kernel 改造前，先用 `benchmarks/kernels/` 微基准或 offline 脚本验证 tile 选择，避免在 43-layer decode 上直接试错。

---

## 9. 结论

- **Gemma4-31B**：A/B 表确认**内存墙**——KV dtype、warmup 变化对 TPS 几乎无正面影响；batch=2 受内存墙与 AWQ M=1 fallback 双重限制。短期应关注投机采样或更低 bit 权重量化，而不是继续调 scheduler。
- **Gemma4-26B**：batch=1 稳定 ~11.4 tok/s，但 **batch=2 严重退化**。验证已排除 scheduler/active-request 限制：engine 会同时调度两个请求，但 `packed_int4_symmetric_fused_qkv_m1_safe` / `packed_int4_symmetric_fused_gate_up_m1_safe` 仅对 M=1 优化，batch>1 时失效 fallback。主要瓶颈是 **M=1 AWQ kernel specialization + MoE launch/sync**，不是纯算力或内存带宽。让 batch=2 真正生效需要 **batched AWQ decode kernel**。
- **DeepSeek V4 Flash**：kept-path warm-cache 已复现 **1.60–1.63 tok/s**，落在 `CAPABILITY_MATRIX` 1.6–1.9 区间下沿；cold-cache 0.46 tok/s 的差距来自专家缓存是否命中。**下一步是固化 kept-path 命令并把现有 `--min-steady-decode-tps` gate 分 cold/warm 阈值接入回归，而不是投向 graph 或新 kernel。**

---

*报告完成。所有数据来自 2026-07-07 在 Radeon 8060S 上的实测与代码静态分析。*
