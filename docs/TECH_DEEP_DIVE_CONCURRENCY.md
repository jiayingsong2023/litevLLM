# LitevLLM 技术深度评估：优缺点、并发与吞吐提升方案

> 面向当前仓库实现（单卡、Python+Triton、`uv` 工作流）的技术层分析。重点回答：
> 1) 项目现阶段技术优缺点；2) 如何提升**高并发性能**与**吞吐能力**；3) 采用何种低风险落地路径。

---

## 1. 技术现状判断（从代码实现出发）

### 1.1 优势（值得保留）

1. **单卡路径聚焦，执行链路短**
   - 引擎直接在 `AsyncLLM -> LiteEngine` 闭环，减少了分布式控制面复杂度，适合快速迭代性能路径。 
2. **已具备连续批处理核心机制**
   - `LiteEngine.step()` 已将请求分为 prefill/decode，并带有 decode-priority 策略、prefill 预算、microbatch 等参数化控制。 
3. **KV Cache 资源模型清晰**
   - 支持 FP8 KV（默认）、max context 对齐、max active requests 限幅，便于在显存约束下做吞吐调优。 
4. **已有端到端性能回归入口**
   - `tests/e2e_full_benchmark.py` 提供并发请求、TTFT、decode TPS 的统计路径，可作为优化前后对比基准。

### 1.2 短板（当前并发上限的关键约束）

1. **异步循环存在 busy-yield 风险**
   - `AsyncLLM` 后台循环里 `engine.step()` 后固定 `await asyncio.sleep(0)`，当请求多、步进耗时高时，容易形成高频空转/调度开销。 
2. **请求超限直接拒绝，缺少等待队列**
   - `LiteEngine.add_request()` 在无空闲 slot 时直接 reject，而不是进入排队队列，这会在高并发流量下放大拒绝率并降低整体吞吐稳定性。 
3. **调度预算按 token 粒度较粗，缺乏自适应反馈**
   - 目前以环境变量配置预算与 prefill/decode 比例，但缺少基于实时指标（TTFT、队列长度、GPU占用）的在线调参闭环。 
4. **并发测试层次仍偏“功能回归”，压测剖析不足**
   - 现有测试更强调正确性和模型对拍，对高并发下的 tail latency、拒绝率、排队时延曲线覆盖不足。

---

## 2. 并发与吞吐瓶颈拆解（系统视角）

### 2.1 控制平面瓶颈
- **入口限流策略偏硬**：超过 `max_active_requests` 就拒绝，缺少“短暂排队 + 过载保护”两级策略。
- **请求编排缺少公平性策略**：decode-priority 对交互体验好，但 prefill backlog 可能积压，导致新请求 TTFT 波动。

### 2.2 数据平面瓶颈
- **KV 预分配与并发数强绑定**：`num_total_blocks = max_active_requests * num_blocks_per_seq`，高并发时显存压力迅速上升。
- **prefill chunk 固定阈值**：不同模型/序列长度下固定 chunk 可能不是最优点，易出现 SM 利用率与内存带宽不匹配。

### 2.3 运行时瓶颈
- **Python 调度开销**：高 QPS 下，过于细粒度的 step 可能引入 Python 层调度成本。
- **缺乏分层指标闭环**：没有系统化输出“排队时长、批次有效token数、拒绝率、重试率、OOM 恢复时长”等关键指标，优化迭代难以量化。

---

## 3. 提升高并发与吞吐的可执行方案（按收益/风险排序）

## 3.1 P0：低风险、快速见效（1~2周）

### A. 增加 admission queue（替代直接 reject）
- 做法
  - 在 `add_request()` 前增加有界队列（如 `pending_queue`），满队列再拒绝。
  - 出队策略：优先短请求或按 FIFO + aging。
- 预期收益
  - 降低短时流量尖峰时的拒绝率，提高吞吐稳定性。
- 风险
  - 队列过长会抬高尾延迟。
- 控制
  - 设置 `max_queue_wait_ms` 与队列长度上限，超时主动失败。

### B. 调度指标落盘（先观测再优化）
- 做法
  - 在 step 级别输出：`queued`, `running`, `prefill_tokens`, `decode_tokens`, `reject_count`, `ttft_p95`。
  - 先 CSV/JSON，后接 prometheus。
- 预期收益
  - 明确瓶颈在“算力不够”还是“调度不佳”。

### C. 统一并发压测口径
- 做法
  - 基于 `tests/e2e_full_benchmark.py` 固定 workload：短输入/长输入/混合三档。
  - 固定报告：吞吐、TTFT p50/p95、E2E p95、拒绝率。

## 3.2 P1：中风险、中高收益（2~5周）

### D. 自适应 prefill/decode 配额
- 做法
  - 基于 backlog 与 recent TTFT 动态调节 `prefill_reserved_tokens` 与 `decode_limit`。
  - 例：当 `ttft_p95` 超阈值，短期提高 prefill 配额；当 decode backlog 上升，回收给 decode。
- 预期收益
  - 降低 tail latency，同时保持 decode 吞吐。

### E. 自适应 chunked prefill
- 做法
  - 根据模型规模、batch 活跃度、OOM 风险动态调整 `prefill_chunk_size`（而非固定 512/1024）。
- 预期收益
  - 更稳定地逼近 GPU 最优吞吐点。

### F. 请求分级（交互优先 vs 批处理优先）
- 做法
  - 增加 QoS 标签：interactive/batch。
  - 调度侧设置最小配额，避免单一流量类型饿死另一类。

## 3.3 P2：高收益但需更严谨验证（5~10周）

### G. 关键路径图捕获（CUDA Graph）与算子融合覆盖提升
- 做法
  - 对稳定 shape 的 decode 热路径优先图捕获，减少 launch 开销。
  - 与现有 Triton kernel 路径配合，逐步扩大覆盖面。
- 预期收益
  - 高并发小 batch 下可显著降低调度和 launch 开销。

### H. 动态批次“有效 token 密度”优化
- 做法
  - 在组批时优先聚合“长度相近请求”，提升每步有效 token 比例。
- 预期收益
  - 减少 padding / 空转 token，提升单位时间真实生成量。

### I. KV 资源策略精细化
- 做法
  - 引入基于请求生命周期的 KV 回收与 reuse 统计；按模型/上下文长度给差异化上限。
- 预期收益
  - 在不增加 OOM 风险前提下提升并发上限。

---

## 4. 建议的性能目标（建议三阶段）

### 阶段1（两周）
- 目标：拒绝率下降 50%，TTFT p95 收敛。
- 手段：Admission queue + 指标落盘 + 压测基线。

### 阶段2（四周）
- 目标：混合负载下 aggregate TPS 提升 15~25%。
- 手段：自适应 prefill/decode 配额 + 自适应 chunk。

### 阶段3（八到十周）
- 目标：在同等质量约束下，E2E p95 再降 15%，吞吐再增 10~20%。
- 手段：图捕获覆盖 + 动态批次密度优化 + KV 策略精细化。

---

## 5. 具体调优参数建议（先从可控参数入手）

> 以下参数均建议在固定压测集上做网格实验，避免“单次跑分偶然性”。

- `FASTINFERENCE_LITE_PREFILL_CHUNK`
  - 建议试验：`256 / 512 / 1024 / 1536`
- `FASTINFERENCE_LITE_PREFILL_MICROBATCH`
  - 建议试验：`1 / 2 / 4`
- `FASTINFERENCE_LITE_PREFILL_RESERVED_TOKENS`
  - 建议试验：`0 / 64 / 128 / 256`
- `FASTINFERENCE_LITE_PREFILL_CATCHUP_RATIO`
  - 建议试验：`0.15 / 0.25 / 0.40`
- `FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS`
  - 建议按显存预算逐级上探，配合拒绝率和OOM联合观察。

---

## 6. uv-first 基准执行建议（可直接落地）

```bash
# 1) 建立基线（固定模型集 + JSON输出）
PYTHONPATH=. uv run python tests/e2e_full_benchmark.py \
  --models tinyllama,qwen35_9b_awq \
  --json-out .tmp_perf_baseline.json

# 2) 进行参数实验（示例）
FASTINFERENCE_LITE_PREFILL_CHUNK=512 \
FASTINFERENCE_LITE_PREFILL_MICROBATCH=2 \
FASTINFERENCE_LITE_PREFILL_CATCHUP_RATIO=0.25 \
PYTHONPATH=. uv run python tests/e2e_full_benchmark.py \
  --models tinyllama,qwen35_9b_awq \
  --json-out .tmp_perf_tuned_a.json

# 3) 可选：AWQ fused GEMM 微基准（非整模）
PYTHONPATH=. uv run python tests/bench_awq_fused_gemm_ab.py
```

---

## 7. 我对项目“优缺点”的结论（直给版）

- **优点**：方向正确（单卡+Triton+性能导向），核心工程能力强（模型适配、KV 与量化实践、回归意识）。
- **缺点**：高并发控制面（排队/调度反馈/可观测性）尚未产品化，导致吞吐上限与稳定性受限。
- **突破口**：不是先“重写内核”，而是先补“调度与观测闭环”；把吞吐提升转化为可重复、可解释的工程收益。


