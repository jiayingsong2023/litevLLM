# FastInference 架构与代码质量评估报告（修订版 v4）

**评估日期**: 2026-05-29  
**评估范围**: vllm/ 核心目录 (554 个 Python 文件, ~81,776 行)  
**可信度**: 方向性评估，关键计数和 lite 引擎路径依赖边界已实测验证。遗留代码的精确删除闭包和 tests/tools 迁移范围仍需专门审计。

---

## 0. 如何使用本报告

本报告定位为**架构讨论稿**，可作为技术评审和重构执行的输入材料。grep/AST 解析/逐行计数已用于验证关键事实；遗留删除闭包和 tests/tools 迁移范围仍需专门审计。

**本轮新增**: §2.5 补充了 lite 引擎路径对上游遗留目录的导入分析——结论是 lite 路径与 worker/core/distributed/third_party 之间**无直接导入依赖**，为分阶段清理提供了边界依据。

---

## 1. 总体评价

FastInference 是从上游 vLLM 向纯 Python + Triton lite-only 路径精简的项目。项目目标明确（见 CLAUDE.md）：**不维护与上游 vLLM 的兼容性**，代码库应保持精简。当前 lite 引擎主路径架构方向清晰，但上游遗留代码（约 15,496 行，占总代码量 ~19%）仍在仓库中堆积，形成理解和维护负担。

**已确认的核心判断**:
- lite engine 控制平面是项目最清晰的部分
- ModelAdapter / RuntimeModelPolicy 方向正确，模型差异隔离机制有效
- 上游遗留代码与 lite 路径之间无直接导入依赖，为分阶段清理提供了边界依据
- 关键架构债务：Gemma4 模块级全局可变状态（9 处）、model_policy 隐式字符串契约、StepScheduler 构造面过大、RuntimeAssemblyContext 类型信息不足、RuntimeConfig 字段膨胀

---

## 2. 架构评估

### 2.1 引擎控制平面 — 良好（项目最成熟的部分）

**组件边界**:
```
StepScheduler（调度决策）
  → PrefillExecutor（硬件 SDPA prefill）
  → DecodeExecutor（Triton PagedAttention decode）
  → SamplingDriver（采样）
  → OutputPipeline（输出组装）
```

**设计亮点**:
- **Observer 模式**：RuntimeObserver 抽象类 + NullRuntimeObserver + InMemoryRuntimeObserver + LoggingRuntimeObserver，符合开闭原则。
- **策略值对象**：SchedulerRuntimePolicy / BackendRuntimePolicy 使用 frozen dataclass 物化调度/后端策略。
- **Factory 装配**：LiteRuntimeFactory.build(RuntimeAssemblyContext) 将 10+ 个组件构造集中在一处。

**实质性问题**:

| 问题 | 精确数据 | 严重度 |
|------|----------|--------|
| StepScheduler.__init__ 参数过多 | 40 个 keyword-only 参数（22 个 LoRA/MultiModal 相关） | 🟡 偏重 |
| RuntimeAssemblyContext 类型信息不足 | 35 个字段，17 个 Any（48.6%） | 🟡 偏重 |
| StepScheduler 单类过大 | 1352 行 | 🟡 偏重 |

### 2.2 配置系统 — 合格（迁移中）

**分层**:
1. `FastInferenceConfig` — TOML 文件层
2. `RuntimeProfile` — 策略解析层
3. `RuntimeConfig` — 运行时扁平层

**已确认**:
- RuntimeConfig 平铺了 14 个 `gemma4_*` 字段，与 Gemma4Adapter.runtime_policy() 返回的 model_policy 形成双层存在——source of truth 不唯一
- env_registry.py 记录了 158 个 FASTINFERENCE_* 环境变量（88 deprecated、69 tool_only、1 public），迁移路线清晰
- RuntimeProfileRegistry._effective_name 接收 model_capabilities 和 gpu_total_gb 参数但当前未使用——动态策略选择尚未实现

### 2.3 Adapter 模式 — 方向正确

ModelAdapter Protocol 定义三个接口契约：detect() → runtime_policy() → install_tuning_config()。Gemma4Adapter / Qwen35Adapter / LlamaAdapter 各自封装模型差异。

**实质性债务**：model_policy 是 `dict[str, object]`，Gemma4Adapter 构建约 19 个字符串键，Qwen35Adapter 构建 5 个——键名没有 TypedDict 或枚举约束。是高价值重构，但不具备全局状态那样直接破坏实例隔离的紧迫性。

### 2.4 模型层 — Gemma4 全局状态是关键风险

**完整清单**（vllm/model_executor/models/gemma4.py）:

| 变量 | 行号 | 类型 | 风险 |
|------|------|------|------|
| `_GEMMA4_TUNING` | 31 | `dict[str, str]` 模块级可变 | 🔴 forward 路径直接依赖 |
| `_GEMMA4_TUNING_LOCKED` | 32 | `bool` 模块级可变 | 🔴 影响 set_gemma4_tuning_config 行为 |
| `_GEMMA4_PROFILE_ENABLED` | 61 | `bool` 模块级可变 | 🔴 在 Gemma4Attention/MLP forward 中读取 |
| `_GEMMA4_ROCTX_PROFILE_ENABLED` | 62 | `bool` 模块级可变 | 🔴 在 forward 的 roctx 中使用 |
| `_GEMMA4_PROFILE_STATS` | 63 | `dict[str, dict]` 模块级可变 | 🟡 跨实例累积统计 |
| `_GEMMA4_PROFILE_PRINTED` | 64 | `bool` 模块级可变 | 🟡 控制 atexit dump 行为 |
| `_GEMMA4_ROPE_CACHE_POOL` | 66 | `OrderedDict` 模块级 LRU | 🔴 跨实例共享缓存 |
| `atexit.register(_dump_gemma4_profile)` | 187 | 进程级回调 | 🟡 模块导入即注册，无实例感知 |
| `_GEMMA4_MOE_MATERIALIZE_BATCH_EXPERTS` | 65 | `int` 常量 | 🟢 不构成风险 |

**核心机理**：`set_gemma4_tuning_config()` 通过 `global` 关键字修改上述变量，在 Gemma4Attention、Gemma4MLP 的 forward 中直接读取。导致：同进程加载两个不同配置的 Gemma4 实例时互相污染、测试间存在隐式依赖。

**P0 两层目标**:
1. **最低目标**：forward 路径不再依赖任何模块级可变配置
2. **完整目标**：tuning、profile stats、rope cache 全部实例隔离；保留的进程级 diagnostic state 需明确标注且不进入 forward 路径

### 2.5 上游遗留代码 — 需按引用闭包分阶段清理

**项目定位**（来自 CLAUDE.md）：
> "heavily trimmed to a pure Python + Triton path"  
> "not maintaining compatibility with upstream vLLM's full feature surface"

**lite 引擎路径导入分析**（实测验证）:

对 `vllm/engine/`、`vllm/serving/`、`vllm/entrypoints/`、`vllm/adapters/` 的导入扫描结论：这四个 lite 核心目录对 worker/core/distributed/third_party **零直接导入**。

**但是**，全仓扫描揭示了大量非 lite 模块对遗留目录的引用：

| 引用方 | 导入的遗留模块 |
|--------|--------------|
| `vllm/forward_context.py` | `vllm.worker.ubatch_utils` |
| `vllm/v1_outputs.py` | `vllm.core.sched.output`, `vllm.distributed.kv_events` |
| `vllm/device_allocator/cumem.py` | `vllm.distributed.device_communicators` |
| `vllm/structured_output/utils.py` | `vllm.core.sched.output`, `vllm.worker.gpu_input_batch` |
| `vllm/attention/backends/utils.py` | `vllm.core.sched.output`, `vllm.worker.gpu_input_batch` |
| `vllm/executor/abstract.py` | `vllm.core.sched.output`, `vllm.worker.worker_base` |
| `vllm/executor/uniproc_executor.py` | `vllm.core.sched.output` |
| `vllm/logging_utils/dump_input.py` | `vllm.core.sched.output` |
| `vllm/model_executor/layers/mamba/*` | `vllm.distributed.parallel_state` 等 |
| `vllm/model_executor/layers/fused_moe/*` | `vllm.distributed`, `vllm.worker.ubatching` |
| `vllm/model_executor/layers/kda.py` | `vllm.distributed` |
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | `vllm.distributed` |
| `vllm/model_executor/layers/quantization/fp8.py` | `vllm.distributed` |
| `vllm/model_executor/model_loader/weight_utils.py` | `vllm.distributed`（lazy import） |
| `vllm/model_executor/warmup/*` | `vllm.worker.gpu_model_runner`, `vllm.worker.gpu_worker` |
| `vllm/utils/import_utils.py` | 动态检测 `vllm.third_party.triton_kernels` |

**结论**：lite 引擎主路径无直接依赖，但全仓仍有 30+ 处引用。**不能一次性删除四个目录**——必须先删除或改造引用方，再按目录分阶段清理。

**遗留目录规模**:

| 目录 | 行数 | 风险 |
|------|------|------|
| vllm/worker/ | 7,184 | 被 forward_context、executor、attention/backends、warmup 引用 |
| vllm/core/ | 2,871 | 被 v1_outputs、structured_output、executor、logging_utils、attention/backends 引用 |
| vllm/third_party/ | 5,127 | 被 utils/import_utils.py 动态检测，影响能力发现 |
| vllm/distributed/ | 46 | 被 model_executor 的 8+ 个模块引用（大部分引用方自身也是可删的遗留层） |
| **合计** | **~15,228** | |

**推荐策略**：引用闭包驱动的分阶段删除。
1. 先产出 DEPENDENCY_CLOSURE.md，列出每个遗留模块的引用者
2. 删除或改造引用方（如 forward_context 的 worker import 可改为条件导入或移除）
3. 每个目录单独删除，每次单独跑 ruff/mypy/regression
4. `vllm/utils/import_utils.py` 对 `vllm.third_party.triton_kernels` 的动态检测需要特殊处理——先改为显式 try/except + 日志后再删

### 2.6 Kernel 层 — 工程实践良好

- 每个 Triton kernel 文件顶部有 ASCII 注释描述内存布局和 tiling
- paged_attention.py 包含寄存器压力分析和低 ILP 回退策略
- awq_fused_gemm.py autotune 优先级链设计合理
- 大规模文件：gemma4_moe_int4.py（3067 行）、awq_fused_gemm.py（2868 行）

### 2.7 测试体系 — 庞杂，需精简

**总规模**: tests/ 目录 31,188 行 Python + 2.4MB 历史数据。其中 **27 个 .py 文件不含任何 `def test_` 函数**（含 `__init__.py`、3 个回归必跑脚本、22 个 tools/ 诊断脚本、1 个未引用压测脚本），这些不是 pytest 测试。

**回归套件引用分析**（实测 `run_inference_correctness_regression.sh` 的依赖）：

| 被引用的文件 | 用途 |
|-------------|------|
| `tests/e2e_full_benchmark.py` | P1 必跑端到端性能回归 |
| `tests/verify_semantic_integrity.py` | 全模型语义完整性检查 |
| `tests/tools/quality_bar_spotcheck.py` | 质量门禁抽查 |
| `tests/tools/gemma4_single_prompt_smoke.py` | Gemma4 A-lite smoke |
| `tests/tools/gemma4_prefill_strict_audit.py` | Gemma4 strict audit |
| `tests/tools/fixtures/*.json` | 正确性测试 prompt 数据 |

**底线（不可删/不可移出 tests/）**:
- `run_regression_suite.sh` — 17 个快速单元/smoke test
- `run_inference_correctness_regression.sh` — 全模型端到端正确性 + 性能回归
- 以上两个脚本引用的**所有**文件

**分类统计**:

| 类别 | 文件数 | 行数 | 被回归套件引用 | 判定 |
|------|--------|------|--------------|------|
| tests/test_*.py | 64 | ~22,700 | 部分（run_regression_suite.sh 列出约 20 个） | 保留 |
| tests/smoke/ | 4 | 116 | 全部 | 保留 |
| `e2e_full_benchmark.py` | 1 | 3,089 | ✅ P1 必跑 | **保留** |
| `verify_semantic_integrity.py` | 1 | 1,405 | ✅ 全模型必跑 | **保留** |
| `tests/tools/quality_bar_spotcheck.py` | 1 | 1,382 | ✅ | **保留** |
| `tests/tools/gemma4_single_prompt_smoke.py` | 1 | 305 | ✅ | **保留** |
| `tests/tools/gemma4_prefill_strict_audit.py` | 1 | 174 | ✅ | **保留** |
| `tests/tools/fixtures/` | - | JSON | ✅ | **保留** |
| `tests/tools/` 其余 17 个脚本 | 17 | ~5,523 | ❌ 未被回归套件引用 | 候选迁移到 `diagnostics/` 或 `benchmarks/`（需先改 CLAUDE.md/AGENTS.md 规范 + 修正动态 import 路径） |
| `tests/tools/` DeepSeek 3 个脚本 | 3 | 1,063 | ❌ DeepSeek 非支持模型 | **删除** |
| `tests/reports/` | 33 | 2.4MB JSON | ❌ | **删除**（历史快照，非代码） |
| `stress_test_64k.py` | 1 | 67 | ❌ | **删除** |

**精简后 tests/ 规模**: 从 31,188 行 + 2.4MB → ~28,500 行（保留所有回归套件引用的文件，删报告数据和非支持模型脚本，迁移独立诊断工具）。核心回归套件完整保留。

**候选迁移的 17 个 scripts（未被回归套件引用，但需先更新仓库规范）**:

| 文件 | 行数 | 建议去向 |
|------|------|----------|
| `_gemma4_diag_utils.py` | 81 | `diagnostics/gemma4/` |
| `gemma4_layer_drift_diagnostic.py` | 262 | `diagnostics/gemma4/` |
| `gemma4_decode_window_ab_report.py` | 362 | `diagnostics/gemma4/` |
| `gemma4_31b_sprint2_matrix.py` | 1,097 | `diagnostics/gemma4/` |
| `profile_gemma4_layer_breakdown.py` | 584 | `diagnostics/gemma4/` |
| `profile_qwen35_layer_breakdown.py` | 621 | `diagnostics/qwen35/` |
| `qwen35_chunk_gated_delta_alignment.py` | 112 | `diagnostics/qwen35/` |
| `qwen35_gated_delta_conv_alignment.py` | 117 | `diagnostics/qwen35/` |
| `qwen35_gguf_alignment_audit.py` | 309 | `diagnostics/qwen35/` |
| `qwen35_moe_packed_lite_logits_audit.py` | 368 | `diagnostics/qwen35/` |
| `verify_qwen35_final_hidden_alignment.py` | 324 | `diagnostics/qwen35/` |
| `report_expected_alignment_metrics.py` | 35 | `diagnostics/qwen35/` |
| `bench_awq_fused_gemm_ab.py` | 127 | `benchmarks/` |
| `bench_gemma4_31b_fused_gemm.py` | 260 | `benchmarks/` |
| `build_awq_fused_profile.py` | 99 | `benchmarks/` |
| `perf_grid_search.py` | 493 | `benchmarks/` |
| `profile_kernel_registers.py` | 272 | `benchmarks/` |

**候选删除（需决策门确认）**:

| 文件 | 行数 | 原因 |
|------|------|------|
| `compare_hf_lite_deepseek_layer_hiddens.py` | 342 | DeepSeek 非当前支持模型；若保留诊断能力则迁移到 diagnostics/ |
| `compare_hf_lite_deepseek_logits.py` | 596 | 同上 |
| `diagnose_deepseek_hf_lite_logits.py` | 125 | 同上 |
| `stress_test_64k.py` | 67 | 未被任何回归套件引用 |

---

## 3. 代码清洁度评估

### 3.1 类型安全

- from __future__ import annotations：33/554 文件（6.0%）
- RuntimeAssemblyContext：35 字段中 17 个 Any（48.6%）
- # type: ignore：engine 层仅 3 处

### 3.2 异常处理

全项目 101 处 `except Exception`。lite_engine.py 中 5 处：3 处有 debug 日志（tuning config 安装），2 处静默 fallback（AWQ stats、KV spec 探测）——后两者建议加 debug 日志。

### 3.3 代码重复

- `_dtype_nbytes()` 在 lite_engine.py:38 和 runtime_planner.py:11 完全重复
- `_truthy()` / `_truthy_string()` 在 adapters/gemma4.py、model_executor/models/gemma4.py、kernels/triton/awq_fused_gemm.py 三处各有实现
- scheduling_constraints.py 单向导入 scheduling_helpers.py（纯函数），边界已拆分

### 3.4 测试质量

- test_step_scheduler.py：40 个测试用例，AAA 结构清晰
- 回归套件完整：run_regression_suite.sh + run_inference_correctness_regression.sh

---

## 4. 优先改进建议

### 🔴 P0 — 阻塞性架构风险

#### 4.1 消除 Gemma4 模块级全局可变状态

**现状**: 9 处模块级可变状态 + 1 处 atexit 注册（详见 §2.4 清单），forward 路径直接依赖 `_GEMMA4_PROFILE_ENABLED`、`_GEMMA4_ROCTX_PROFILE_ENABLED`、`_GEMMA4_TUNING`。

**两层目标**:
1. **最低目标**：forward 路径不再依赖任何模块级可变配置
2. **完整目标**：tuning、profile stats、rope cache 全部实例隔离；保留的进程级 diagnostic state 需明确标注且不进入 forward 路径

**验收标准**:
- 新增测试：同进程创建两个不同 tuning 配置的 Gemma4 实例，各自 forward 读取各自的配置
- 新增测试：测试 tearDown 无需手动重置模块级全局变量
- 相关 regression suite 通过

---

### 🟡 P1 — 高价值重构

#### 4.2 删除上游遗留目录（引用闭包驱动的分阶段删除）

**现状**: worker/core/distributed/third_party 共 ~15,228 行。lite 引擎路径零直接导入，但全仓仍有 30+ 处引用（详见 §2.5 引用方清单）。

**方案**: 分阶段执行，每个阶段单独验证。
1. 产出 DEPENDENCY_CLOSURE.md，列出每个遗留模块的全部引用者
2. 删除或改造引用方（如 forward_context、executor、v1_outputs 等 v1 路径模块中的条件导入）
3. `vllm/utils/import_utils.py` 对 `vllm.third_party.triton_kernels` 的动态检测先改为显式 try/except
4. 每个目录单独删除，单独跑 ruff/mypy/regression
5. 最后在 model_executor 内部逐层清理 mamba/fused_moe 分布式路径等遗留层

**验收标准**:
- 每个阶段 `uv run ruff check vllm` 零错误
- 每个阶段 `uv run mypy vllm` 零新增错误
- 每个阶段 `bash tests/run_regression_suite.sh` 通过
- 全部完成后 `bash tests/run_inference_correctness_regression.sh` 通过

#### 4.3 model_policy / kernel_policy 类型化

**方案**（分步）:
1. 在 adapters/policy_keys.py 定义键名常量，替换裸字符串
2. 建立 TypedDict（Gemma4ModelPolicy, Qwen35ModelPolicy）
3. 最终：每个 model adapter 返回 typed dataclass

**验收标准**: adapter 生成端和主要消费端不再使用裸字符串；新增 policy key 必须出现在常量/TypedDict 定义中。

#### 4.4 RuntimeConfig 的 Gemma4 字段迁移到 model_policy

**现状**: 14 个 `gemma4_*` 字段在 RuntimeConfig 和 model_policy 中双层存在，source of truth 不唯一。

**方案**: 将 source of truth 统一到 Gemma4Adapter.runtime_policy() 返回的 model_policy。RuntimeConfig.gemma4_* 保留为 deprecated fallback。添加 contract test 验证映射关系。

**验收标准**: 模型层 gemma4_* 读取全部走 model_policy；contract test 覆盖所有 14 个字段。

#### 4.5 StepScheduler 构造参数聚合

**方案**: 将 LoRA 约束参数聚合为 LoraSchedulingParams，MultiModal 约束聚合为 MultiModalSchedulingParams。签名从 40 参数降到 ~18 参数。

**验收标准**: StepScheduler.__init__ 签名参数 ≤ 20 个；40 个调度器测试全部通过。

#### 4.6 RuntimeAssemblyContext 类型强化

**方案**: 为高频字段添加具体类型（kv_caches → list[torch.Tensor]、model → nn.Module）。

**验收标准**: Any 字段数从 17 降到 ≤ 10。

---

### 🟢 P2 — 清洁度提升

#### 4.7 测试目录精简

**前提**: 当前 CLAUDE.md/AGENTS.md 将一次性诊断工具放在 `tests/tools/`。如果迁移，需先更新仓库规范。

**动态依赖审计**（实测）:
- `tests/tools/_gemma4_diag_utils.py` 被 `test_gemma4_diagnostics_warn_only.py` 直接 import——此测试在 run_regression_suite.sh 中
- `tests/tools/gemma4_layer_drift_diagnostic.py` 被 `test_gemma4_diagnostics_warn_only.py` 以子进程执行
- `tests/tools/qwen35_moe_packed_lite_logits_audit.py` 被 `test_logits_dump_stats.py` 以 `importlib` 动态加载
- `tests/tools/perf_grid_search.py` 被 `test_perf_grid_search.py` 以 `importlib` 动态加载
- `verify_semantic_integrity.py` 有 DeepSeek 相关常量/函数，删除 DeepSeek 脚本需同步修改

**方案**:
1. 先审计所有 `tests/` 下对 `tests/tools/` 的动态引用（importlib、subprocess、直接 import），产出 `tests/TOOLS_REFERENCE_AUDIT.md`
2. 删除 `tests/reports/`（2.4MB 历史数据）
3. DeepSeek 诊断脚本处置：若正式确认 DeepSeek 不再作为支持/诊断目标，则删除 3 个脚本 + `stress_test_64k.py`，同步修改 `verify_semantic_integrity.py` 中的 DeepSeek 阈值常量和 `tests/tools/README.md`、`docs/INFERENCE_ACCURACY.md`；否则迁移到 `diagnostics/` 而非删除
4. 对于动态加载的 tools（logits_dump_stats、perf_grid_search、gemma4_diag_utils）：先改对应 test 的加载路径，再迁移脚本
5. 如决定迁移：更新 CLAUDE.md 中的诊断工具位置约定

**验收标准**:
- `bash tests/run_regression_suite.sh` 通过
- `bash tests/run_inference_correctness_regression.sh` 通过
- 迁移后所有动态 import 路径有效

#### 4.9 工具函数去重

- `_dtype_nbytes()` → vllm/utils/torch_utils.py
- `_truthy_string()` / `_truthy()` → vllm/utils/text_utils.py

#### 4.10 异常处理细化

- lite_engine.py 中 2 处静默 fallback 添加 logger.debug()

#### 4.11 超大类拆分

- gemma4.py（2770 行）：Gemma4Attention、Gemma4MLP、Gemma4SparseMoeBlock 拆为独立文件
- awq_fused_gemm.py（2868 行）：autotune/profile 部分拆出

#### 4.12 model_executor 内部遗留层清理

删除 worker/core/distributed/third_party 后，model_executor 内部的 mamba、fused_moe 分布式路径、kda、warmup 等遗留层也可逐步清理。建议第二阶段逐文件分析依赖。

---

## 5. 建议 PR 拆分顺序

原则：先建文档锁住边界 → 加测试暴露问题 → 修 P0 全局状态 → 类型和结构重构 → 遗留代码按引用闭包逐目录删除 → 测试精简 → 清洁度提升。

| 序号 | 内容 | 优先级 | 预计改动 | 依赖 |
|------|------|--------|----------|------|
| **PR0** | 依赖闭包文档（DEPENDENCY_CLOSURE.md）+ 自动检查脚本，列出所有 legacy 目录引用者 | P1-4.2 前置 | 文档新增 + CI 脚本 | 无 |
| **PR1** | Gemma4 全局污染隔离测试：先加 **xfail** 测试，证明同进程不同配置互相污染。优先用 minimal config / monkeypatch / 轻量 layer 对象验证配置隔离（不依赖完整模型加载和 GPU），完整模型测试放 correctness regression | P0-4.1 | 新增 ~80 行测试 | 无 |
| **PR2** | 移除 tuning/profile flag 全局状态：将 `_GEMMA4_TUNING`、`_GEMMA4_PROFILE_ENABLED`、`_GEMMA4_ROCTX_PROFILE_ENABLED` 改为实例级，PR1 xfail 通过 | P0-4.1 最低目标 | gemma4.py ~80 行 | PR1 |
| **PR3** | 移除 rope cache 全局状态 + profile stats 实例化：`_GEMMA4_ROPE_CACHE_POOL` 改实例级 LRU；处理 `_GEMMA4_PROFILE_STATS` 和 atexit.register | P0-4.1 完整目标 | gemma4.py ~60 行 + 测试 | PR2 |
| **PR4** | policy key 常量定义（adapters/policy_keys.py），替换裸字符串 | P1-4.3 第一步 | ~50 行新增 + 多处替换 | 无（不阻塞 P0） |
| **PR5** | Gemma4/Qwen policy TypedDict 或集中 accessor | P1-4.3 后续 | adapters/ ~40 行 | PR4 |
| **PR6** | RuntimeConfig 的 Gemma4 字段迁移到 model_policy + contract test | P1-4.4 | runtime_config.py + adapter + 测试 | PR5 |
| **PR7** | RuntimeAssemblyContext 类型强化 | P1-4.6 | runtime_factory.py ~25 行 | 无 |
| **PR8** | StepScheduler 参数聚合（LoraSchedulingParams + MultiModalSchedulingParams） | P1-4.5 | step_scheduler.py + factory ~100 行 | 无 |
| **PR9** | 测试目录引用审计（TOOLS_REFERENCE_AUDIT.md），先更新 README/docs/动态加载测试 | P2-4.7 前置 | 文档 + 测试路径修正 | 无 |
| **PR10** | 删除 `tests/reports/` + `stress_test_64k.py` + DeepSeek 诊断脚本（同步修改 verify_semantic_integrity.py 和 docs/） | P2-4.7 | 删除 ~2.4MB + ~1,130 行 | PR9 |
| **PR11a** | 按 PR0 闭包，处理 worker/ 引用方 + 删除 worker/ + ruff/mypy/regression 验证 | P1-4.2 | 删除 ~7,184 行 + 引用方改造 | PR0 |
| **PR11b** | 按 PR0 闭包，处理 core/ 引用方 + 删除 core/ + 验证 | P1-4.2 | 删除 ~2,871 行 + 引用方改造 | PR11a |
| **PR11c** | 按 PR0 闭包，处理 distributed/ 引用方 + 删除 distributed/ + 验证 | P1-4.2 | 删除 ~46 行 + 引用方改造 | PR11b |
| **PR11d** | 处理 `utils/import_utils.py` 动态检测 + 删除 third_party/ + 验证 | P1-4.2 | 删除 ~5,127 行 + import_utils 改造 | PR11c |
| **P2** | 异常日志补充 + 工具函数去重 + 超大文件拆分 + model_executor 内部遗留层清理 | P2 | 分散 | PR0-PR11d 完成后 |

## 6. 架构亮点

1. **控制平面流水线设计**：engine 层组件边界清晰
2. **Adapter 模式**：方向正确，模型差异隔离有效
3. **env → config 迁移**：158 个变量的 EnvScope 分类规范可追溯
4. **Observer 模式**：开闭原则实践
5. **Triton kernel 注释规范**：内存布局、register pressure、launch config 均文档化

## 7. 关键风险

1. **全局状态**（P0）：Gemma4 模块级 9 处可变状态 + atexit 注册——直接破坏同进程多实例和测试隔离。这是唯一的 P0 项。
2. **遗留代码堆积**（P1）：~15,228 行遗留代码，lite 引擎路径无直接依赖但全仓仍有 30+ 处引用——需引用闭包驱动的分阶段删除，不能一次性删除
3. **测试目录边界模糊**（P2）：29 个文件不含 test 函数，17 个 tools 脚本未被回归套件引用但有动态 import 依赖——精简前需先审计引用关系
4. **双层配置**（P1）：14 个 gemma4_* 字段在 RuntimeConfig 和 model_policy 中同时存在，source of truth 不唯一
5. **隐式契约**（P1）：model_policy 的字符串键名缺少编译时校验
