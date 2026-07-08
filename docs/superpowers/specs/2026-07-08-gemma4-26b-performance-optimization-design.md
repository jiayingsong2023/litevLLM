# Gemma4-26B-A4B 推理性能优化设计

> 设计日期：2026-07-08  
> 目标模型：Gemma4-26B-A4B-it-AWQ-4bit（Q4 MoE）  
> 目标硬件：AMD Ryzen AI MAX+ 395 / Radeon 8060S Graphics（gfx1151），96 GB 统一内存  
> 基线性能：BS=1 decode **11.4 tok/s**，BS=2 decode 严重退化至 **1.29–4.25 tok/s**

---

## 1. 背景与现状

### 1.1 为什么单独给 26B 立项

FastInference 已有的性能评估（`docs/performance_evaluation_gemma4_deepseek_2026_07_07.md`）同时覆盖了 Gemma4-31B、Gemma4-26B 和 DeepSeek V4 Flash，但三者的瓶颈完全不同：

- **31B**：dense 60 层，已接近统一内存带宽墙，优化空间主要在权重量化/投机采样。
- **DeepSeek V4 Flash**：GGUF Q2 直接 runtime，瓶颈在专家缓存命中与 Python-level staging。
- **26B**：A4B MoE，30 层，128 experts / top-8。roofline 估算每 token 仅读取约 **1.24 GB** 权重，等效带宽仅 **~15 GB/s**，远低于硬件上限。

因此 26B 的问题不是“硬件吃满”，而是 **小 GEMV 利用率低、MoE expert 切换/launch overhead、M=1 kernel specialization 导致 batch>1 退化**。这些属于常规 kernel/runtime 优化范畴，比 speculative decoding 更可控。

### 1.2 关键证据

来自 `tests/e2e_full_benchmark.py`（2026-07-07 实测）：

| 模型 | batch | KV dtype | decode TPS (agg) | TTFT (ms) | 结论 |
|---|---|---|---|---|---|
| Gemma4-26B-A4B | 1 | fp8 | **11.42** | 2252 | 稳定基线 |
| Gemma4-26B-A4B | 1 | int4 | 11.17 | 2265 | KV dtype 几乎无影响 |
| Gemma4-26B-A4B | 2 | fp8 | **1.29** | 3523 | 严重退化 |
| Gemma4-26B-A4B | 2 | int4 | 4.25 | 3530 | 仍退化 |

退化根因已确认：**AWQ fused QKV / gate-up kernel 仅对 M=1 做了 hand-optimized fusion**。

- `vllm/model_executor/models/gemma4/attention.py:52`：`x2.shape[0] != 1` 时直接返回 `None`。
- `vllm/model_executor/models/_fused_awq_pair.py:218`：同样要求 `x2.shape[0] == 1`。
- `StepScheduler` 已将两个请求放入同一 decode step（`steps=37` 处理 46 tokens），因此不是 scheduler 限制。

### 1.3 优化目标

| 阶段 | 目标 | 验证方式 |
|---|---|---|
| P0 | 产出 26B 专属热点 profile 与 fast-path coverage 报告 | rocprofv3 + Python section timer + AWQ audit |
| P1 | BS=1 decode >= **13 tok/s** | `tests/e2e_full_benchmark.py --models gemma4_26b_a4b` |
| P2 | BS=1 decode >= **15 tok/s**，且 BS=2 不退化 | 同上 + `--gemma26b-concurrent 2` |
| P3 | 评估 speculative decoding 可行性 | 独立 prototype |

---

## 2. 优化路线图

### 2.1 优先级总览

| 优先级 | 任务 | 核心收益 | 风险 | 预估投入 |
|---|---|---|---|---|
| P0 | 26B profiling + fast-path coverage audit | 定位真实热点，避免盲目优化 | 低 | 1–2 天 |
| P1 | MoE int4 decode path 优化 | 提升 BS=1 吞吐，减少 expert overhead | 中 | 3–5 天 |
| P2 | M=2/M=4 batched AWQ decode kernel | 解决 BS>1 退化，支撑未来 verifier batch | 中 | 5–10 天 |
| P3 | Speculative decoding viability | 探索 BS=1 进一步加速 | 高 | 独立项目 |

---

## 3. P0：26B Profiling + Fast-Path Coverage Audit

### 3.1 目标

- 拿到 26B 在 Radeon 8060S 上的 **ROCm kernel trace**。
- 确认 Python-level section timer 中 attention / MoE / router 的占比。
- 确认 AWQ fused QKV / gate-up / MoE int4 decode 的命中率。

### 3.2 执行步骤

#### 3.2.1 ROCm Kernel Trace

```bash
# 1. 基线 e2e benchmark（产出 JSON 报告）
FASTINFERENCE_KV_TYPE=turbo_int4 \
FASTINFERENCE_GEMMA4_ALLOW_INT4_KV=0 \
uv run python tests/e2e_full_benchmark.py \
  --models gemma4_26b_a4b \
  --gemma26b-concurrent 1 \
  --gemma26b-max-new-tokens 24 \
  --json-out /tmp/gemma26b_baseline.json

# 2. ROCm kernel trace（产出 kernel_stats.csv + 原始 trace）
#    用 --gemma26b-max-new-tokens 8 缩短 decode 阶段，减少 profile 数据量。
rocprofv3 --kernel-trace --stats \
  --output-dir /tmp/gemma26b_rocprof \
  -- uv run python tests/e2e_full_benchmark.py \
    --models gemma4_26b_a4b \
    --gemma26b-max-new-tokens 8 \
    --json-out /tmp/gemma26b_rocprof.json

# 3. AWQ fast-path coverage audit（AWQ audit 事件默认记录，无需额外开关）
uv run python tests/e2e_full_benchmark.py \
  --models gemma4_26b_a4b \
  --gemma26b-concurrent 1 \
  --gemma26b-max-new-tokens 24 \
  --json-out /tmp/gemma26b_audit.json \
  > /tmp/gemma26b_awq_audit.log 2>&1
```

**P0 必须产出的 artifacts**：

| Artifact | 路径 | 用途 |
|---|---|---|
| e2e baseline JSON | `/tmp/gemma26b_baseline.json` | TTFT / prefill / decode TPS |
| kernel stats CSV | `/tmp/gemma26b_rocprof/kernel_stats.csv` | Top kernels / calls / avg time |
| profile JSON | `/tmp/gemma26b_rocprof.json` | 必要时二次分析 |
| AWQ audit log | `/tmp/gemma26b_awq_audit.log` | fused QKV / gate-up / MoE int4 命中率 |
| section timer 汇总 | 从 stdout 或 profile JSON 提取 | attention / MoE / router 占比 |

#### 3.2.2 Python Section Timer

`vllm/model_executor/models/gemma4/profiling.py` 已提供 `_gemma4_profile_span`。先确认 section timer 的启用方式：

```bash
# 方式 1：使用真实存在的 Gemma4 layer profile 开关
FASTINFERENCE_GEMMA4_LAYER_PROFILE=1 uv run python tests/e2e_full_benchmark.py --models gemma4_26b_a4b

# 方式 2：若方式 1 不够细，临时在 vllm/model_executor/models/gemma4/layer.py 中
# 将 Gemma4LayerConfig 实例的 profile_enabled 设为 True（P0 阶段专用，不提交）
```

> P0 第一步使用 `FASTINFERENCE_GEMMA4_LAYER_PROFILE=1`；若该开关不够细，改用方式 2 或修改 profiling 开关。

重点观察 section：

- `layer_self_attn`
- `attn_local_decode_kernel` / `attn_global_kernel`
- `layer_moe_router`
- `layer_moe_sparse_experts`
- `moe_materialize_one_expert_awq`
- `moe_int4_decode_attempt`

#### 3.2.3 AWQ Fast-Path Coverage Audit

```bash
uv run python tests/e2e_full_benchmark.py \
  --models gemma4_26b_a4b \
  --gemma26b-concurrent 1 \
  --gemma26b-max-new-tokens 24
```

检查 stdout / 生成的 audit 报告中：

- `qkv_fused_decode` count 与 `qkv_separate_decode` count。
- `gate_up_fused_decode`（如果存在）是否命中。
- `moe_int4_decode_used` count 与 fallback reason。

相关代码位置：

- `vllm/model_executor/models/gemma4/attention.py:223-266`
- `vllm/model_executor/models/_fused_awq_pair.py:168-239`
- `vllm/model_executor/models/gemma4/moe.py:504-567`

### 3.3 P0 完成标准

- [ ] 产出 `/tmp/gemma26b_rocprof/kernel_stats.csv`（或等效 rocprofv3 输出）。
- [ ] 产出 Python section timer 汇总（attention / MoE / router 占比）。
- [ ] 产出 AWQ audit 汇总，明确 fused QKV / gate-up / MoE int4 命中率。
- [ ] 根据数据确认 P1 的优化方向（例如：若 MoE int4 已 100% 命中但 router 占比高，则优先优化 router）。

---

## 4. P1：MoE INT4 Decode Path 优化

### 4.1 目标

在不改变模型语义的前提下，提升 BS=1 decode 吞吐至 >= 13 tok/s。

### 4.2 候选优化点

#### 4.2.1 策略选择

**不要假设默认策略**。P0 的 fast-path coverage audit 必须确认当前 26B e2e 实际命中的策略：

- 检查 `vllm/model_executor/models/gemma4/moe.py` 中 `_int4_kernel_strategy` 的默认值。
- 检查 `tests/e2e_full_benchmark.py` 的 `_GEMMA4_26B_RECOMMENDED_ENV` 是否覆盖策略。
- 通过 AWQ audit / profile 输出确认每次 decode 实际使用的 kernel 路径。

只有在确认当前策略不是最优后，再离线评估候选策略：

- `two_stage`
- `batched`
- `batched_tuned`
- `batched_chunked`
- `batched_grouped`

验证方式（以 `batched_tuned` 为例，实际选择由 P0 数据决定）：

```bash
FASTINFERENCE_GEMMA4_MOE_INT4_KERNEL_STRATEGY=batched_tuned \
uv run python tests/e2e_full_benchmark.py --models gemma4_26b_a4b
```

#### 4.2.2 Tile 调优

`vllm/kernels/triton/gemma4_moe_int4.py` 中 tuning 常量可能未针对 26B 形状优化：

```python
_TUNED_GATE_UP_BLOCK_PACKS = 8
_TUNED_DOWN_BLOCK_PACKS = 8
_TUNED_DOWN_BLOCK_H_SMALL_H = 64
_TUNED_DOWN_BLOCK_H_LARGE_H = 128
_CHUNKED_BLOCK_I = 256
_CHUNKED_BLOCK_H = 64
_CHUNKED_BLOCK_PACKS_H = 8
_CHUNKED_BLOCK_PACKS_I = 8
```

26B 的形状：

- `hidden_dim = 2816`
- `intermediate_dim = 704`
- `top_k = 8`
- `num_experts = 128`

建议用 `benchmarks/kernels/benchmark_moe.py` 或新建 microbench 扫描：

```bash
uv run python benchmarks/kernels/benchmark_moe.py \
  --hidden 2816 --intermediate 704 --top-k 8 --batch-sizes 1,2,4
```

#### 4.2.3 Expert Cache 调优

当前默认 `gemma4_moe_expert_cache_size=32`（`vllm/adapters/gemma4.py:110`）。对 26B 128 experts / top-8，可尝试 64 / 128：

```bash
# P0/P1 阶段可用环境变量做 A/B 实验
FASTINFERENCE_GEMMA4_MOE_EXPERT_CACHE_SIZE=64 \
uv run python tests/e2e_full_benchmark.py --models gemma4_26b_a4b
```

> 最终代码中任何新 tuning 参数必须通过 `RuntimeConfig` / TOML `[tuning_keyvals]` 暴露，不得新增 `os.environ` 读取。A/B 实验用的临时环境变量不得提交。

#### 4.2.4 Router / Top-K 优化

`Gemma4TopKRouterLite.forward`（`vllm/model_executor/models/gemma4/moe.py:92-125`）执行：

1. RMSNorm（无缩放）
2. `self.proj(x)`
3. `F.softmax` + `torch.topk`
4. 可选 `per_expert_scale`

对 decode M=1，top-k 涉及的 tensor 很小，但 30 层 × 每 token 调用一次，可能累积。可尝试：

- 将 router 的 `proj` 也走 AWQ decode GEMV fast path（如果还没走）。
- 若 `per_expert_scale` 为空或全 1，跳过乘法。

### 4.3 P1 完成标准

- [ ] BS=1 decode TPS >= **13 tok/s**（对比基线 11.4）。
- [ ] 所有改动通过 `bash tests/run_regression_suite.sh`。
- [ ] 若涉及 MoE int4 kernel tile 变更，需提供 microbench 对比数据。
- [ ] 最终代码不引入新的环境变量读取（遵循 AGENTS.md：配置走 TOML / `RuntimeConfig`）。P0/P1 的临时 A/B 实验环境变量不得提交。

---

## 5. P2：M=2/M=4 Batched AWQ Decode Kernel

### 5.1 目标

解决 BS>1 退化，支撑未来 speculative verifier 或小批 decode。

### 5.2 范围

优先扩展两个 fusion：

1. **Fused QKV**：`vllm/kernels/triton/awq_fused_gemm.py` 中的 `packed_int4_symmetric_fused_qkv_m1_safe`。
2. **Fused gate-up**：`vllm/kernels/triton/awq_fused_gemm.py` 中的 `packed_int4_symmetric_fused_gate_up_m1_safe`。

### 5.3 实现步骤

#### 5.3.1 Microbench 先验证收益

在修改模型代码前，先建离线 microbench：

```bash
# 建议新增或复用 benchmarks/kernels/benchmark_awq_fused_gemm.py
uv run python benchmarks/kernels/benchmark_awq_fused_gemm.py \
  --m 1,2,4 \
  --n 5376,2816,5632 \
  --k 2816,21504 \
  --group-size 32,128
```

目标：M=2/M=4 的 fused kernel throughput 不低于 M=1 的 80%（即 batch=2 时 aggregate TPS 至少接近 2×11.4）。

#### 5.3.2 Kernel 侧改动

在 `vllm/kernels/triton/awq_fused_gemm.py` 中：

- 新增 `packed_int4_symmetric_fused_qkv_batched_safe(x, q_qweight, k_qweight, v_qweight, ..., m: int)`。
- 新增 `packed_int4_symmetric_fused_gate_up_batched_safe(x, qwg, qwu, ..., m: int)`。
- 内部根据 `m` 选择 BLOCK_M=1/2/4 的 tile。

约束：

- 仅处理 symmetric packed int4（qzeros is None），与现有 M=1 路径一致。
- 保持输出 dtype 和 scale 处理方式不变。
- 若 batched kernel 失败，自动回退到现有 separate LiteLinear 路径。

#### 5.3.3 模型侧改动

1. `vllm/model_executor/models/gemma4/attention.py:216-230`
   - 将 `is_decode_m1` 改为 `is_decode_small_batch`。
   - 当 `M <= 4` 时尝试 fused batched QKV。

2. `vllm/model_executor/models/_fused_awq_pair.py:216-219`
   - 放宽 `x2.shape[0] == 1` 限制到 `x2.shape[0] <= 4`。

3. `vllm/model_executor/models/gemma4/moe.py`
   - 确保 26B 默认使用 batched MoE int4 decode 路径。

### 5.4 P2 完成标准（分阶段）

**第一阶段：消除 BS=2 灾难性退化**

- [ ] M=1 路径性能不退化（与 baseline 差异 < 3%）。
- [ ] BS=2 decode aggregate TPS >= **11–13 tok/s**（对比当前 1.29–4.25，先达到不弱于 BS=1 的水平）。
- [ ] 所有改动通过 `bash tests/run_regression_suite.sh` 和 `bash tests/run_inference_correctness_regression.sh`（Gemma4-26B 部分）。

**第二阶段： stretched goal**

- [ ] BS=1 decode TPS >= **15 tok/s**。
- [ ] BS=2 decode aggregate TPS >= **18–20 tok/s**。
- [ ] 同样通过全部回归测试。

---

## 6. P3：Speculative Decoding Viability 评估

### 6.1 目标

评估在 26B 上引入投机采样的可行性，作为 BS=1 进一步加速的备选。

### 6.2 评估标准

- **Draft 来源优先级**：
  1. n-gram / prompt lookup（无额外模型，tokenizer 天然兼容）。
  2. tokenizer-compatible 的小 Gemma 模型（如 Gemma-2B 级别，若本地存在）。
  3. 其他小模型仅作为最后选项，且必须验证 tokenizer/vocab 匹配。
- **测量指标**：
  - `acceptance_rate`：被 target 模型接受的 draft token 比例。
  - `effective_tps`：考虑 draft + verify 总耗时后的实际 decode TPS。
- **立项门槛**：`effective_tps` 必须显著高于 P2 完成后的 BS=1 baseline（例如 > 15 tok/s），而不仅仅是 acceptance rate 高。

### 6.3 暂不实现

P3 作为独立项目，不在本设计范围内。

---

## 7. 风险与回退策略

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| batched AWQ kernel tile 在 gfx1151 上表现不佳 | P2 目标无法达成 | 保留 M=1 原始路径；batched kernel 失败时自动 fallback |
| MoE int4 策略切换导致精度回归 | correctness 测试失败 | 任何策略变更都跑 `tests/run_inference_correctness_regression.sh` |
| 新 kernel 增加 Triton 编译时间 | 冷启动变慢 | 复用 `FASTINFERENCE_BENCH_COMPILE_CACHE_DIR` 机制 |
| Router 优化收益不明显 | P1 目标延迟 | 优先做有 microbench 数据支撑的点 |

---

## 8. 验证命令清单

### 8.1 基线测量

```bash
uv run python tests/e2e_full_benchmark.py \
  --models gemma4_26b_a4b \
  --gemma26b-concurrent 1 \
  --gemma26b-max-new-tokens 24 \
  --json-out /tmp/gemma26b_baseline.json
```

### 8.2 BS=2 退化验证

```bash
uv run python tests/e2e_full_benchmark.py \
  --models gemma4_26b_a4b \
  --gemma26b-concurrent 2 \
  --gemma26b-max-new-tokens 24 \
  --json-out /tmp/gemma26b_batch2.json
```

### 8.3 回归测试

```bash
bash tests/run_regression_suite.sh
SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh
```

### 8.4 ROCm Profile

```bash
rocprofv3 --kernel-trace --stats \
  --output-dir /tmp/gemma26b_rocprof \
  -- uv run python tests/e2e_full_benchmark.py \
  --models gemma4_26b_a4b \
  --gemma26b-max-new-tokens 8
```

> 注：rocprofv3 的具体 CLI 语法以本机 ROCm 版本为准。若 `-- <cmd>` 形式不支持，改用 profile 输入文件或 `rocprof` v2。

---

## 9. 待决策事项

1. P1 中是否新增一个 `benchmarks/kernels/benchmark_gemma4_moe_26b.py` 专门 microbench，还是复用现有 `benchmark_moe.py`？
2. P2 中 batched AWQ kernel 的 M 上限设为 4 还是 8？建议先 4。
3. ~~是否允许在 P1 阶段临时使用环境变量做 A/B 实验？~~ 已明确：P0/P1 可临时用 env 做实验，最终代码必须走 RuntimeConfig / TOML。

---

## 10. 结论

Gemma4-26B-A4B 不是内存墙模型，当前主要瓶颈是 **M=1 AWQ kernel specialization 与 MoE launch/expert overhead**。本设计通过 **P0 profiling → P1 MoE 优化 → P2 batched AWQ kernel → P3 speculative 评估** 的分阶段路线，目标将 BS=1 decode 从 11.4 tok/s 提升至 15 tok/s，并彻底解决 BS>1 退化问题。
