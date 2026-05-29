# P0 组：Gemma4 全局状态隔离 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 `vllm/model_executor/models/gemma4.py` 中所有模块级全局可变状态，实现同进程不同 Gemma4 配置实例隔离。

**Architecture:** 将 9 处模块级全局状态（tuning dict、profile flags、profile stats、rope cache pool、atexit hook）从模块级变量转为实例属性或通过 `RuntimeModelPolicy.model_policy` 传递。前置工作：产出 lite 路径依赖闭包文档和自动检查脚本。

**Tech Stack:** Python 3.12, PyTorch, Triton (ROCm), pytest

**Branch:** `refactor/p0-gemma4-global-state-isolation` (从 `main` 创建)

**Verification gate:** 全部完成后运行 `run_inference_correctness_regression.sh` + `e2e_full_benchmark.py`

---

## PR0: 依赖闭包文档 + 自动检查脚本

### Task 0.1: 创建依赖闭包文档

**Files:**
- Create: `docs/DEPENDENCY_CLOSURE.md`

- [ ] **Step 1: Write the dependency closure document**

`docs/DEPENDENCY_CLOSURE.md`:
```markdown
# Lite Engine 依赖闭包

从 `vllm/engine/lite_engine.py` 出发的 transitive import 分析。

## 必须保留（lite 路径 transitive import 覆盖）

- `vllm/engine/*` — 引擎控制平面（lite_engine, step_scheduler, runtime_factory, ...）
- `vllm/serving/config_builder.py` — 配置构建
- `vllm/adapters/*` — 模型适配器（base, gemma4, qwen3_5, llama, registry）
- `vllm/model_executor/models/gemma4.py` — Gemma4 模型定义
- `vllm/model_executor/models/qwen3_5.py` — Qwen3.5 模型定义
- `vllm/model_executor/models/llama.py` — Llama 模型定义
- `vllm/model_executor/layers/lite_linear.py` — Lite 线性层
- `vllm/model_executor/model_loader/*` — 模型加载
- `vllm/kernels/triton/*` — Triton 内核
- `vllm/config/*` — 基础配置 dataclass
- `vllm/entrypoints/*` — API 入口
- `vllm/sample/*` — 采样
- `vllm/inputs/*` — 输入处理
- `vllm/triton_utils/*` — Triton 工具
- `vllm/utils/*` — 通用工具函数
- `vllm/transformers_utils/*` — HF 工具
- `vllm/attention/*` — 注意力后端

## 安全可删（lite 路径无直接/间接依赖）

### 一级目录

| 目录 | 行数 | lite import | 非 lite 引用方 |
|------|------|-------------|---------------|
| `vllm/worker/` | 7,184 | 零 | `vllm/forward_context.py`, `vllm/executor/abstract.py`, `vllm/executor/uniproc_executor.py`, `vllm/attention/backends/utils.py`, `vllm/structured_output/utils.py`, `vllm/logging_utils/dump_input.py`, `vllm/model_executor/warmup/*` |
| `vllm/core/` | 2,871 | 零 | `vllm/v1_outputs.py`, `vllm/executor/abstract.py`, `vllm/executor/uniproc_executor.py`, `vllm/attention/backends/utils.py`, `vllm/structured_output/utils.py`, `vllm/logging_utils/dump_input.py` |
| `vllm/distributed/` | 46 | 零 | `vllm/model_executor/layers/mamba/*`, `vllm/model_executor/layers/fused_moe/*`, `vllm/model_executor/layers/vocab_parallel_embedding.py`, `vllm/model_executor/layers/kda.py`, `vllm/model_executor/layers/quantization/fp8.py`, `vllm/model_executor/model_loader/weight_utils.py`, `vllm/device_allocator/cumem.py` |
| `vllm/third_party/` | 5,127 | 零 | `vllm/utils/import_utils.py`（动态能力检测） |

### 引用方处理清单

- `vllm/forward_context.py:17` — `from vllm.worker.ubatch_utils import UBatchSlices`
- `vllm/executor/abstract.py:14,18` — `from vllm.core.sched.output import ...` + `from vllm.worker.worker_base import WorkerBase`
- `vllm/executor/uniproc_executor.py:7` — `from vllm.core.sched.output import ...`
- `vllm/attention/backends/utils.py:23-24` — `from vllm.core.sched.output import ...` + `from vllm.worker.gpu_input_batch import InputBatch`
- `vllm/structured_output/utils.py:20,29` — `from vllm.core.sched.output import ...` + `from vllm.worker.gpu_input_batch import InputBatch`
- `vllm/v1_outputs.py:13,16-17` — `from vllm.core.sched.output import ...` + `from vllm.distributed.kv_events import ...`
- `vllm/logging_utils/dump_input.py:12` — `from vllm.core.sched.output import SchedulerOutput`
- `vllm/device_allocator/cumem.py:33` — `from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary`
- `vllm/model_executor/layers/mamba/*` — `from vllm.distributed.*`（整个 mamba/ 自身是遗留层）
- `vllm/model_executor/layers/fused_moe/*` — `from vllm.distributed.*`, `from vllm.worker.ubatching import ...`
- `vllm/model_executor/layers/kda.py` — `from vllm.distributed import ...`
- `vllm/model_executor/layers/vocab_parallel_embedding.py` — `from vllm.distributed import ...`
- `vllm/model_executor/layers/quantization/fp8.py` — `from vllm.distributed import ...`
- `vllm/model_executor/model_loader/weight_utils.py:673` — lazy `from vllm.distributed import ...`
- `vllm/model_executor/warmup/kernel_warmup.py:15-16` — `from vllm.worker.gpu_model_runner import GPUModelRunner` + `from vllm.worker.gpu_worker import Worker`
- `vllm/model_executor/warmup/deep_gemm_warmup.py:8` — `from vllm.distributed.parallel_state import ...`
- `vllm/utils/import_utils.py` — 动态检测 `vllm.third_party.triton_kernels`

## 兼容性边界（lite 不依赖但外部 API 需要）

- `vllm/entrypoints/openai/*` — OpenAI API server（如有外部用户依赖）
- `vllm/entrypoints/cli/*` — CLI 入口
- `vllm/grpc/*` — gRPC（如有外部用户依赖）
```

- [ ] **Step 2: Commit**

```bash
git add docs/DEPENDENCY_CLOSURE.md
git commit -m "docs: add lite engine dependency closure document

Lists transitive imports from lite_engine.py, categorizes files as
must-keep / safe-to-delete / compatibility-boundary, and enumerates
all non-lite reference sites for legacy directories.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 0.2: 创建自动检查脚本

**Files:**
- Create: `scripts/check_lite_imports.sh`

- [ ] **Step 1: Write the CI check script**

`scripts/check_lite_imports.sh`:
```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# CI guard: forbid new lite-engine imports from legacy directories.
# Legacy dirs: vllm/worker vllm/core vllm/distributed vllm/third_party
# Lite dirs: vllm/engine vllm/serving vllm/entrypoints vllm/adapters
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LEGACY_DIRS="vllm/worker|vllm/core|vllm/distributed|vllm/third_party"
LITE_DIRS="vllm/engine|vllm/serving|vllm/entrypoints|vllm/adapters"

violations=$(grep -rn "from \(${LEGACY_DIRS//./\\.}\)" \
  $(find vllm/engine vllm/serving vllm/entrypoints vllm/adapters \
    -name "*.py" -not -path "*__pycache__*") 2>/dev/null || true)

if [ -n "$violations" ]; then
  echo "ERROR: Lite engine paths MUST NOT import from legacy directories:"
  echo "$violations"
  echo ""
  echo "Legacy directories: vllm/worker, vllm/core, vllm/distributed, vllm/third_party"
  echo "If this import is intentional, update docs/DEPENDENCY_CLOSURE.md and this script."
  exit 1
fi

echo "PASS: No lite-engine imports from legacy directories."
```

- [ ] **Step 2: Make executable and test**

```bash
chmod +x scripts/check_lite_imports.sh
bash scripts/check_lite_imports.sh
```

Expected: `PASS: No lite-engine imports from legacy directories.`

- [ ] **Step 3: Commit**

```bash
git add scripts/check_lite_imports.sh
git commit -m "feat: add CI guard script for lite-engine legacy imports

Forbids new imports from vllm/worker, vllm/core, vllm/distributed,
vllm/third_party into lite engine core directories.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 0.3: 跑快速回归验证 PR0

- [ ] **Step 1: Run regression suite**

```bash
bash tests/run_regression_suite.sh
```

Expected: all tests pass (PR0 only adds docs + shell script, no code changes).

---

## PR1: Gemma4 全局污染隔离测试 (xfail)

### Task 1.1: 创建隔离测试

**Files:**
- Create: `tests/test_gemma4_global_state_isolation.py`

- [ ] **Step 1: Write the xfail isolation test**

`tests/test_gemma4_global_state_isolation.py`:
```python
# SPDX-License-Identifier: Apache-2.0
"""Verify that Gemma4 tuning/profile flags are instance-isolated.

Before global state remediation (PR2/PR3), these tests are expected to
fail (xfail) — they demonstrate that module-level ``_GEMMA4_TUNING``,
``_GEMMA4_PROFILE_ENABLED``, and ``_GEMMA4_ROCTX_PROFILE_ENABLED``
are currently shared across instances.
"""

import pytest
import torch

from vllm.model_executor.models.gemma4 import (
    set_gemma4_tuning_config,
    _GEMMA4_PROFILE_ENABLED,
    _GEMMA4_ROCTX_PROFILE_ENABLED,
    _GEMMA4_TUNING,
)


@pytest.fixture(autouse=True)
def _reset_gemma4_global_state():
    """Reset module-level globals between tests so failures don't cascade."""
    # Reset to defaults
    set_gemma4_tuning_config(None, locked=False)
    yield
    set_gemma4_tuning_config(None, locked=False)


@pytest.mark.xfail(
    reason="PR2 not yet applied: _GEMMA4_TUNING is module-level mutable state"
)
def test_tuning_config_isolation_across_instances():
    """Two calls to set_gemma4_tuning_config with different profiles should
    be stored independently, not overwrite each other at module level."""
    set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "1"}, locked=False
    )
    tuning_a = dict(_GEMMA4_TUNING)
    assert tuning_a.get("FASTINFERENCE_GEMMA4_LAYER_PROFILE") == "1"

    set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "0"}, locked=False
    )
    tuning_b = dict(_GEMMA4_TUNING)

    # With proper instance isolation, tuning_a should still have "1",
    # tuning_b should have "0". Currently they share module-level dict.
    assert tuning_a.get("FASTINFERENCE_GEMMA4_LAYER_PROFILE") == "1", (
        "Expected tuning_a to retain '1' after second set_gemma4_tuning_config "
        "— but module-level _GEMMA4_TUNING was overwritten to "
        f"'{tuning_a.get('FASTINFERENCE_GEMMA4_LAYER_PROFILE')}'"
    )
    assert tuning_b.get("FASTINFERENCE_GEMMA4_LAYER_PROFILE") == "0"


@pytest.mark.xfail(
    reason="PR2 not yet applied: _GEMMA4_PROFILE_ENABLED is module-level mutable state"
)
def test_profile_flag_isolation():
    """Two different profile configurations should have independent
    _GEMMA4_PROFILE_ENABLED flags."""
    set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "1"}, locked=False
    )
    profile_a = _GEMMA4_PROFILE_ENABLED

    set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "0"}, locked=False
    )
    profile_b = _GEMMA4_PROFILE_ENABLED

    assert profile_a is True
    assert profile_b is False
```

- [ ] **Step 2: Run test to verify it fails (xfail)**

```bash
uv run pytest tests/test_gemma4_global_state_isolation.py -v
```

Expected: 2 xfail (tests run but are marked as expected failures).

- [ ] **Step 3: Commit**

```bash
git add tests/test_gemma4_global_state_isolation.py
git commit -m "test: add xfail Gemma4 global state isolation tests

Demonstrates module-level _GEMMA4_TUNING and _GEMMA4_PROFILE_ENABLED
are shared across instances. Tests are marked xfail pending PR2/PR3.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 1.2: 跑快速回归验证 PR1

- [ ] **Step 1: Run regression suite**

```bash
bash tests/run_regression_suite.sh
```

Expected: all tests pass (new test is xfail so doesn't break the suite).

---

## PR2: 移除 tuning/profile flag 全局状态（P0 最低目标）

### Task 2.1: 定义 Gemma4 实例级配置容器

**Files:**
- Modify: `vllm/model_executor/models/gemma4.py:25-188`

- [ ] **Step 1: Add Gemma4LayerConfig dataclass**

Insert after line 30 (`_GEMMA4_TUNING_LOCKED = False`), before line 35 (`def set_gemma4_tuning_config`):

```python
from dataclasses import dataclass, field


@dataclass
class Gemma4LayerConfig:
    """Per-instance tuning/profile configuration for Gemma4 layers.

    Replaces the module-level globals ``_GEMMA4_TUNING``,
    ``_GEMMA4_PROFILE_ENABLED``, ``_GEMMA4_ROCTX_PROFILE_ENABLED``,
    ``_GEMMA4_PROFILE_STATS``, ``_GEMMA4_PROFILE_PRINTED``, and
    ``_GEMMA4_ROPE_CACHE_POOL``.

    Instance lifetime is tied to the owning ``Gemma4DecoderLayer``.
    """

    profile_enabled: bool = False
    roctx_profile_enabled: bool = False
    profile_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    profile_printed: bool = False
    tuning: dict[str, str] = field(default_factory=dict)
    tuning_locked: bool = False
```

- [ ] **Step 2: Update set_gemma4_tuning_config to return Gemma4LayerConfig**

Replace lines 35-54 (the existing `set_gemma4_tuning_config` function):

```python
def set_gemma4_tuning_config(
    values: dict[str, object] | None, *, locked: bool = False
) -> Gemma4LayerConfig:
    """Build a Gemma4LayerConfig from tuning overrides.

    Returns a new ``Gemma4LayerConfig`` instance — no module-level side effects.
    For backward compatibility, the returned config can be installed on a
    ``Gemma4DecoderLayer`` via ``layer.set_config(config)``.

    The old behaviour of mutating module-level globals is preserved *only*
    when the caller does not pass a target instance — a deprecation warning
    is emitted in that path.
    """
    global _GEMMA4_TUNING, _GEMMA4_TUNING_LOCKED
    global _GEMMA4_PROFILE_ENABLED, _GEMMA4_ROCTX_PROFILE_ENABLED

    tuning = {
        str(key): str(value)
        for key, value in (values or {}).items()
        if str(key) in _GEMMA4_ALLOWED_TUNING_ENV and value is not None
    }
    profile_enabled = _truthy_string(
        tuning.get("FASTINFERENCE_GEMMA4_LAYER_PROFILE")
    )
    roctx_enabled = _truthy_string(
        tuning.get("FASTINFERENCE_GEMMA4_ROCTX_PROFILE")
    )

    # --- NEW: return instance config ---
    return Gemma4LayerConfig(
        tuning=tuning,
        tuning_locked=bool(locked),
        profile_enabled=profile_enabled,
        roctx_profile_enabled=roctx_enabled,
    )


# Backward-compat shim: keep module-level mutation for callers
# that don't yet use the instance config path.
def _apply_global_tuning_config(config: Gemma4LayerConfig) -> None:
    """Apply a Gemma4LayerConfig to the legacy module-level globals.

    Deprecated: new code should pass the config to Gemma4DecoderLayer
    via ``set_config()``.
    """
    global _GEMMA4_TUNING, _GEMMA4_TUNING_LOCKED
    global _GEMMA4_PROFILE_ENABLED, _GEMMA4_ROCTX_PROFILE_ENABLED
    _GEMMA4_TUNING = dict(config.tuning)
    _GEMMA4_TUNING_LOCKED = config.tuning_locked
    _GEMMA4_PROFILE_ENABLED = config.profile_enabled
    _GEMMA4_ROCTX_PROFILE_ENABLED = config.roctx_profile_enabled
```

- [ ] **Step 3: Update Gemma4Adapter.install_tuning_config to use new API**

Modify `vllm/adapters/gemma4.py` lines 154-157:

```python
    def install_tuning_config(self, tuning_env: dict[str, str]) -> None:
        config = set_gemma4_tuning_config(tuning_env, locked=True)
        _apply_global_tuning_config(config)
```

- [ ] **Step 4: Add set_config method to Gemma4DecoderLayer**

In `class Gemma4DecoderLayer.__init__` (around line 2470), add:

```python
        self._layer_config = Gemma4LayerConfig()

    def set_config(self, config: Gemma4LayerConfig) -> None:
        """Install per-instance tuning/profile configuration."""
        self._layer_config = config
        # Propagate to sub-layers
        self.self_attn._layer_config = config
        self.mlp._layer_config = config
```

And in `Gemma4Attention.__init__` and `Gemma4MLP.__init__`, add:

```python
        self._layer_config = Gemma4LayerConfig()
```

- [ ] **Step 5: Replace global reads in forward paths with instance reads**

In `_Gemma4ProfileSpan.__enter__` (lines 134-137), replace:

```python
        if _GEMMA4_ROCTX_PROFILE_ENABLED:
```
with:
```python
        if self._layer_config.roctx_profile_enabled:
```

In `_Gemma4ProfileSpan.__exit__` (lines 142-145), replace:

```python
        if _GEMMA4_PROFILE_ENABLED:
```
with:
```python
        if self._layer_config.profile_enabled:
```

Note: `_Gemma4ProfileSpan` needs a `_layer_config` reference. Add it to `__init__`:

```python
    def __init__(self, scope: str, layer_config: Gemma4LayerConfig | None = None):
        self.scope = scope
        self._layer_config = layer_config or Gemma4LayerConfig()
```

In `Gemma4Attention.forward` (line 2175), replace `_GEMMA4_PROFILE_ENABLED` with `self._layer_config.profile_enabled`.

In `Gemma4DecoderLayer.forward`, replace `_GEMMA4_PROFILE_ENABLED` (line 2178) with `self._layer_config.profile_enabled`.

- [ ] **Step 6: Run regression to verify no breakage**

```bash
bash tests/run_regression_suite.sh
```

Expected: all tests pass.

- [ ] **Step 7: Run isolation test to verify it now passes**

```bash
uv run pytest tests/test_gemma4_global_state_isolation.py -v
```

Expected: `test_tuning_config_isolation_across_instances` and `test_profile_flag_isolation` should now both PASS (we need to adapt them to use the new instance-based API). If they still fail because tests compare module-level globals, update the test in this step to use `Gemma4LayerConfig` instances directly.

Update `tests/test_gemma4_global_state_isolation.py` — remove xfail markers and rewrite to use instance configs:

```python
def test_tuning_config_isolation_across_instances():
    config_a = set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "1"}, locked=False
    )
    assert config_a.profile_enabled is True

    config_b = set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "0"}, locked=False
    )
    assert config_b.profile_enabled is False

    # Instance isolation: config_a unchanged
    assert config_a.profile_enabled is True
    assert config_b.profile_enabled is False


def test_profile_flag_isolation():
    config_a = set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "1"}, locked=False
    )
    config_b = set_gemma4_tuning_config(
        {"FASTINFERENCE_GEMMA4_LAYER_PROFILE": "0"}, locked=False
    )
    assert config_a.profile_enabled is True
    assert config_b.profile_enabled is False
```

- [ ] **Step 8: Commit**

```bash
git add vllm/model_executor/models/gemma4.py vllm/adapters/gemma4.py tests/test_gemma4_global_state_isolation.py
git commit -m "fix: eliminate tuning/profile global state from Gemma4 forward path

Replace module-level _GEMMA4_TUNING, _GEMMA4_PROFILE_ENABLED,
_GEMMA4_ROCTX_PROFILE_ENABLED with per-instance Gemma4LayerConfig.
set_gemma4_tuning_config() now returns an immutable config instance.
Legacy module-level mutation preserved via _apply_global_tuning_config
shim for backward compat.

P0 minimum target: forward path no longer reads module-level mutables.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 2.2: 跑快速回归验证 PR2

- [ ] **Step 1: Run regression suite**

```bash
bash tests/run_regression_suite.sh
```

Expected: all tests pass.

---

## PR3: 移除 rope cache + profile stats + atexit（P0 完整目标）

### Task 3.1: 将 _GEMMA4_ROPE_CACHE_POOL 移入 Gemma4LayerConfig

**Files:**
- Modify: `vllm/model_executor/models/gemma4.py:66-69, 934-952`

- [ ] **Step 1: Add rope_cache_pool to Gemma4LayerConfig**

Add to `Gemma4LayerConfig` dataclass:

```python
    rope_cache_pool: OrderedDict[
        tuple[int, int, float, str, float, str, int, str],
        tuple[torch.Tensor, torch.Tensor],
    ] = field(default_factory=OrderedDict)
```

- [ ] **Step 2: Replace global _GEMMA4_ROPE_CACHE_POOL reads with config reads**

In `Gemma4LayerRotaryEmbedding` (around line 934), replace:

```python
        cached = _GEMMA4_ROPE_CACHE_POOL.get(key)
        if cached is not None:
            _GEMMA4_ROPE_CACHE_POOL.move_to_end(key)
            return cached

        cos, sin = self._compute_rope(positions_expanded, head_dim, rope_theta, scaling)
        _GEMMA4_ROPE_CACHE_POOL[key] = (cos, sin)
        pool_limit = _gemma4_policy_value(runtime_config, "rope_cache_pool_max", 8)
        while len(_GEMMA4_ROPE_CACHE_POOL) > pool_limit:
            _GEMMA4_ROPE_CACHE_POOL.popitem(last=False)
```

with:

```python
        # Use instance-level rope cache from layer config
        layer_config = getattr(self, '_layer_config', None)
        pool = layer_config.rope_cache_pool if layer_config is not None else _GEMMA4_ROPE_CACHE_POOL

        cached = pool.get(key)
        if cached is not None:
            pool.move_to_end(key)
            return cached

        cos, sin = self._compute_rope(positions_expanded, head_dim, rope_theta, scaling)
        pool[key] = (cos, sin)
        pool_limit = _gemma4_policy_value(runtime_config, "rope_cache_pool_max", 8)
        while len(pool) > pool_limit:
            pool.popitem(last=False)
```

- [ ] **Step 3: Delete module-level _GEMMA4_ROPE_CACHE_POOL**

Remove line 66-69 (the global `OrderedDict` declaration). Keep as a fallback in the code path for backward compat (see step 2 above).

### Task 3.2: 将 _GEMMA4_PROFILE_STATS 移入 Gemma4LayerConfig

- [ ] **Step 1: Replace global _GEMMA4_PROFILE_STATS with config field**

In `_Gemma4ProfileSpan.__enter__` (line 123), replace:

```python
        bucket = _GEMMA4_PROFILE_STATS.setdefault(scope, {"time_s": 0.0, "count": 0.0})
```

with:

```python
        cfg = self._layer_config
        bucket = cfg.profile_stats.setdefault(scope, {"time_s": 0.0, "count": 0.0})
```

In `_dump_gemma4_profile` (lines 153-165), replace references to `_GEMMA4_PROFILE_STATS` with an argument:

```python
def _dump_gemma4_profile(config: Gemma4LayerConfig | None = None) -> None:
    """Dump per-instance profile stats. If no config given, use module-level fallback."""
    if config is None:
        return
    if not config.profile_enabled or not config.profile_stats or config.profile_printed:
        return
    config.profile_printed = True
    total_s = sum(v["time_s"] for v in config.profile_stats.values())
    if total_s <= 0:
        return
    for scope, v in sorted(config.profile_stats.items()):
        print(f"[gemma4-profile] {scope}: {v['time_s']:.4f}s ({int(v['count'])} calls, "
              f"{100.0 * v['time_s'] / total_s:.1f}%)")
```

- [ ] **Step 2: Replace atexit.register with instance lifecycle**

Remove line 187 (`atexit.register(_dump_gemma4_profile)`). Instead, add profile dump to `Gemma4ForCausalLM` or let the engine's observer handle it.

In `Gemma4ForCausalLM.__init__`, add a cleanup hook:

```python
        import atexit
        self._gemma4_profile_config: Gemma4LayerConfig | None = None

    def _register_profile_dump(self, config: Gemma4LayerConfig) -> None:
        """Register atexit profile dump for this model instance."""
        self._gemma4_profile_config = config
        import atexit
        atexit.register(self._dump_profile)

    def _dump_profile(self) -> None:
        _dump_gemma4_profile(self._gemma4_profile_config)
```

### Task 3.3: 新增 rope cache 独立性测试

**Files:**
- Modify: `tests/test_gemma4_global_state_isolation.py`

- [ ] **Step 1: Add rope cache isolation test**

```python
def test_rope_cache_isolation_across_configs():
    """Two Gemma4LayerConfig instances should have independent rope caches."""
    config_a = Gemma4LayerConfig()
    config_b = Gemma4LayerConfig()

    # Simulate a cache entry
    key_a = (0, 128, 1.0, "default", 10000.0, "linear", 4096, "llama3")
    config_a.rope_cache_pool[key_a] = (
        torch.randn(1, 128), torch.randn(1, 128)
    )

    assert key_a in config_a.rope_cache_pool
    assert key_a not in config_b.rope_cache_pool, (
        "config_b's rope cache should be independent of config_a's"
    )
```

- [ ] **Step 2: Run all isolation tests**

```bash
uv run pytest tests/test_gemma4_global_state_isolation.py -v
```

Expected: all tests PASS.

### Task 3.4: 跑快速回归验证 PR3

- [ ] **Step 1: Run regression suite**

```bash
bash tests/run_regression_suite.sh
```

Expected: all tests pass.

- [ ] **Step 2: Commit**

```bash
git add vllm/model_executor/models/gemma4.py tests/test_gemma4_global_state_isolation.py
git commit -m "fix: eliminate Gemma4 rope cache, profile stats, atexit global state

- _GEMMA4_ROPE_CACHE_POOL → instance-level OrderedDict in Gemma4LayerConfig
- _GEMMA4_PROFILE_STATS / _GEMMA4_PROFILE_PRINTED → config fields
- atexit.register(_dump_gemma4_profile) → per-instance lifecycle
- Added rope cache independence test

P0 complete target: all Gemma4 mutable global state is instance-isolated.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 验证门

### Gate: 完整回归

- [ ] **Step 1: Run inference correctness regression**

```bash
bash tests/run_inference_correctness_regression.sh
```

Expected: all model tiers pass (TinyLlama A+B, Qwen3.5-9B A+B, Gemma4-31B A-lite+B, Gemma4-26B A-strict+A-lite+B).

- [ ] **Step 2: Run e2e benchmark**

```bash
uv run python tests/e2e_full_benchmark.py
```

Expected: benchmark completes, performance within acceptable range.

- [ ] **Step 3: Run mypy**

```bash
uv run mypy vllm
```

Expected: no new errors vs baseline.

- [ ] **Step 4: Run ruff**

```bash
uv run ruff check vllm
```

Expected: no new errors vs baseline.
