# P0 组实施规格：基础设施 + Gemma4 全局状态隔离

**日期**: 2026-05-29  
**来源**: architecture_evaluation.md v5, §5 PR 拆分顺序  
**分支策略**: 从 `main` 创建单分支，完成 PR0-PR3 后合并；P1/P2 组后续独立分支处理。

## PR0: 依赖闭包文档

**目标**: 产出 `DEPENDENCY_CLOSURE.md`，从 `vllm/engine/lite_engine.py` 出发递归分析 import 树，标注三类文件。

**产出物**:
- `docs/DEPENDENCY_CLOSURE.md` — 三类文件清单
- `scripts/check_lite_imports.py` — CI 可运行的自动检查脚本

**三类标注**:
1. **必须保留** — lite 路径 transitive import 覆盖的所有模块
2. **安全可删** — worker/core/distributed/third_party 及其依赖链中无 lite 路径引用的模块
3. **兼容性边界** — lite 路径不依赖但外部 API（OpenAI server, CLI）依赖的模块

## PR1: Gemma4 全局污染隔离测试 (xfail)

**目标**: 新增一个用 monkeypatch + 轻量 layer 对象的测试，证明两个不同 tuning 配置的实例在模块级全局状态下配置互相污染。

**不依赖**:
- 完整模型加载
- GPU
- 权重文件

**实现**: 创建两个 `Gemma4DecoderLayer` 最小实例（或更低层的 `Gemma4Attention`/`Gemma4MLP`），通过 `set_gemma4_tuning_config()` 设置不同 tuning flag，在 forward 中读回并断言各自配置独立——当前预期失败（xfail）。

## PR2: 修复 tuning/profile flag 全局状态（P0 最低目标）

**修改文件**: `vllm/model_executor/models/gemma4.py`

**变更**:
- 删除 `global _GEMMA4_TUNING`、`global _GEMMA4_TUNING_LOCKED`
- `_GEMMA4_PROFILE_ENABLED`、`_GEMMA4_ROCTX_PROFILE_ENABLED` 通过 `RuntimeModelPolicy.model_policy` 传入 `Gemma4DecoderLayer.__init__`
- `set_gemma4_tuning_config()` 改为将 tuning 数据存入实例属性而非模块级变量
- PR1 测试从 xfail 改为 pass

**验收**: `tests/run_regression_suite.sh` 通过

## PR3: 修复 rope cache + profile stats + atexit（P0 完整目标）

**修改文件**: `vllm/model_executor/models/gemma4.py`

**变更**:
- `_GEMMA4_ROPE_CACHE_POOL` → 实例级 `OrderedDict`（移至 `Gemma4DecoderLayer`）
- `_GEMMA4_PROFILE_STATS`、`_GEMMA4_PROFILE_PRINTED` → 改为 observer 收集或标注为进程级 diagnostic（不进入 forward 路径）
- `atexit.register(_dump_gemma4_profile)` → 改为实例级生命周期管理
- 新增 rope cache 独立性测试

**验收**: `tests/run_regression_suite.sh` 通过

## 验证节点

P0 组全部完成后运行：
- `bash tests/run_inference_correctness_regression.sh`
- `uv run python tests/e2e_full_benchmark.py`
