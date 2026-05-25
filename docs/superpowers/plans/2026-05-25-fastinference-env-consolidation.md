# FastInference Env Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce public runtime `FASTINFERENCE_*` environment variables to fewer than 10 while preserving existing Qwen3.5-9B and Gemma4-26B/31B correctness and performance behavior.

**Architecture:** Add a central environment registry, route production env reads through it, replace broad `FASTINFERENCE_*` capture with explicit legacy compatibility, then migrate model/kernel/test tuning to profile/config/CLI controls. Governance tests enforce that no new unmanaged `FASTINFERENCE_*` names enter the tree.

**Tech Stack:** Python 3.12, `uv`, pytest, existing `RuntimeConfig`, `RuntimeProfile`, adapter policy, and Triton kernel policy plumbing.

---

## File Structure

- Create `vllm/engine/env_registry.py`: central allowlist, scope/deprecation metadata, final public env set, legacy compatibility parser.
- Modify `vllm/engine/runtime_profile.py`: replace direct `FASTINFERENCE_PROFILE` read with registry helper.
- Modify `vllm/engine/runtime_config.py`: replace broad `FASTINFERENCE_*` capture with registry-managed compatibility capture.
- Modify `vllm/engine/inference_config.py`: route legacy `from_env` reads through registry helpers before later removing the class-level env surface.
- Modify `vllm/adapters/gemma4.py`: stop deriving production policy from `runtime_config.tuning_env`; use profile/config fields and legacy compatibility object only.
- Modify `vllm/engine/lite_engine.py`: install structured tuning config generated from policy, not arbitrary process env.
- Modify `tests/test_project_governance.py`: enforce registry coverage and production read restrictions.
- Create `tests/test_env_registry.py`: unit tests for registry behavior.
- Modify existing runtime/config policy tests as each compatibility path moves.

## Task 1: Add Registry and Freeze New Env Names

**Files:**
- Create: `vllm/engine/env_registry.py`
- Create: `tests/test_env_registry.py`
- Modify: `tests/test_project_governance.py`

- [ ] **Step 1: Write failing registry tests**

Add `tests/test_env_registry.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.engine.env_registry import (
    FINAL_PUBLIC_FASTINFERENCE_ENV,
    FASTINFERENCE_ENV_REGISTRY,
    EnvScope,
)


def test_final_public_fastinference_env_is_below_limit() -> None:
    assert len(FINAL_PUBLIC_FASTINFERENCE_ENV) < 10
    assert FINAL_PUBLIC_FASTINFERENCE_ENV == {
        "FASTINFERENCE_PROFILE",
        "FASTINFERENCE_CONFIG",
        "FASTINFERENCE_KV_TYPE",
        "FASTINFERENCE_DEBUG",
        "FASTINFERENCE_LOG_LEVEL",
        "FASTINFERENCE_BENCH_PROFILE",
        "FASTINFERENCE_ALLOW_LEGACY_ENV",
    }


def test_kv_fp8_is_deprecated_alias_not_final_public() -> None:
    spec = FASTINFERENCE_ENV_REGISTRY["FASTINFERENCE_KV_FP8"]
    assert spec.scope is EnvScope.DEPRECATED
    assert spec.replacement == "FASTINFERENCE_KV_TYPE=fp8"
    assert "FASTINFERENCE_KV_FP8" not in FINAL_PUBLIC_FASTINFERENCE_ENV


def test_all_final_public_env_names_are_registered_as_public() -> None:
    for name in FINAL_PUBLIC_FASTINFERENCE_ENV:
        assert FASTINFERENCE_ENV_REGISTRY[name].scope is EnvScope.PUBLIC
```

Extend `tests/test_project_governance.py` with:

```python
def test_fastinference_env_names_are_registered() -> None:
    from vllm.engine.env_registry import FASTINFERENCE_ENV_REGISTRY

    ignored_roots = {
        ".git",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
    pattern = re.compile(
        r"(?<![A-Z0-9_])FASTINFERENCE_[A-Z0-9_]*[A-Z0-9](?![A-Z0-9_])"
    )
    found: set[str] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in ignored_roots for part in rel.parts):
            continue
        if rel.parts[:2] == ("tests", "reports"):
            continue
        if path.suffix not in {".py", ".sh", ".md", ".toml", ".yaml", ".yml"}:
            continue
        found.update(pattern.findall(path.read_text(encoding="utf-8", errors="ignore")))

    found.discard("FASTINFERENCE_TUNING_ENV_PREFIX")
    unregistered = sorted(found - set(FASTINFERENCE_ENV_REGISTRY))
    assert not unregistered, "Unregistered FASTINFERENCE_* names: " + ", ".join(unregistered)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest tests/test_env_registry.py tests/test_project_governance.py::test_fastinference_env_names_are_registered -q
```

Expected: import failure for `vllm.engine.env_registry`.

- [ ] **Step 3: Implement registry**

Create `vllm/engine/env_registry.py` with:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EnvScope(str, Enum):
    PUBLIC = "public"
    DEPRECATED = "deprecated"
    TOOL_ONLY = "tool_only"
    REMOVED = "removed"


@dataclass(frozen=True)
class FastInferenceEnvSpec:
    name: str
    scope: EnvScope
    replacement: str | None = None
    description: str = ""


FINAL_PUBLIC_FASTINFERENCE_ENV: frozenset[str] = frozenset(
    {
        "FASTINFERENCE_PROFILE",
        "FASTINFERENCE_CONFIG",
        "FASTINFERENCE_KV_TYPE",
        "FASTINFERENCE_DEBUG",
        "FASTINFERENCE_LOG_LEVEL",
        "FASTINFERENCE_BENCH_PROFILE",
        "FASTINFERENCE_ALLOW_LEGACY_ENV",
    }
)


def _public(name: str, description: str) -> FastInferenceEnvSpec:
    return FastInferenceEnvSpec(name=name, scope=EnvScope.PUBLIC, description=description)


def _deprecated(
    name: str,
    *,
    replacement: str,
    description: str = "",
) -> FastInferenceEnvSpec:
    return FastInferenceEnvSpec(
        name=name,
        scope=EnvScope.DEPRECATED,
        replacement=replacement,
        description=description,
    )


def _tool_only(name: str, description: str = "") -> FastInferenceEnvSpec:
    return FastInferenceEnvSpec(
        name=name,
        scope=EnvScope.TOOL_ONLY,
        description=description,
    )
```

Populate `FASTINFERENCE_ENV_REGISTRY` from the current scan output. Generate the source list with:

```bash
rg --files -g '!**/.venv/**' -g '!**/__pycache__/**' -g '!tests/reports/**' \
  | rg '(^|/)(vllm|tests|docs)(/|$)|(^|/)(README|AGENTS)\.md$|\.sh$|\.py$|\.toml$|\.md$|\.yaml$|\.yml$' \
  | uv run python -c 'import sys,pathlib,re; pat=re.compile(r"(?<![A-Z0-9_])FASTINFERENCE_[A-Z0-9_]*[A-Z0-9](?![A-Z0-9_])"); names=set(); [names.update(pat.findall(pathlib.Path(p.strip()).read_text(encoding="utf-8", errors="ignore"))) for p in sys.stdin if p.strip()]; names.discard("FASTINFERENCE_TUNING_ENV_PREFIX"); print("\n".join(sorted(names)))'
```

Register the 7 final names as `PUBLIC`, `FASTINFERENCE_KV_FP8` as `DEPRECATED` with replacement `FASTINFERENCE_KV_TYPE=fp8`, current production tuning names as `DEPRECATED`, and test/benchmark/doc-only names as `TOOL_ONLY` or `REMOVED`.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_env_registry.py tests/test_project_governance.py::test_fastinference_env_names_are_registered -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/engine/env_registry.py tests/test_env_registry.py tests/test_project_governance.py
git commit -m "feat: register fastinference env surface"
```

## Task 2: Replace Broad Runtime Env Capture

**Files:**
- Modify: `vllm/engine/env_registry.py`
- Modify: `vllm/engine/runtime_config.py`
- Modify: `tests/test_kv_default_policy.py`
- Modify: `tests/test_project_governance.py`

- [ ] **Step 1: Write failing tests**

Add `test_runtime_config_ignores_legacy_env_without_compat` to assert default `RuntimeConfig.from_vllm_config` does not capture arbitrary legacy env:

```python
def test_runtime_config_ignores_legacy_env_without_compat(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_ALLOW_INT4_KV", "1")
    monkeypatch.delenv("FASTINFERENCE_ALLOW_LEGACY_ENV", raising=False)

    cfg = _runtime_config_from_minimal_vllm_config()

    assert cfg.tuning_env == {"FASTINFERENCE_PROFILE": "auto"}
```

Add the compatibility case:

```python
def test_runtime_config_collects_deprecated_env_when_compat_enabled(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_ALLOW_LEGACY_ENV", "1")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_ALLOW_INT4_KV", "1")

    cfg = _runtime_config_from_minimal_vllm_config()

    assert cfg.tuning_env["FASTINFERENCE_GEMMA4_ALLOW_INT4_KV"] == "1"
    assert cfg.tuning_env["FASTINFERENCE_PROFILE"] == "auto"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest tests/test_kv_default_policy.py -q
```

Expected: new default-ignore test fails because current code captures all `FASTINFERENCE_*`.

- [ ] **Step 3: Implement compatibility capture**

Add to `env_registry.py`:

```python
def legacy_env_enabled(environ: Mapping[str, str]) -> bool:
    return environ.get("FASTINFERENCE_ALLOW_LEGACY_ENV", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def collect_runtime_tuning_env(environ: Mapping[str, str]) -> dict[str, str]:
    if not legacy_env_enabled(environ):
        return {}
    return {
        name: value
        for name, value in environ.items()
        if name in FASTINFERENCE_ENV_REGISTRY
        and FASTINFERENCE_ENV_REGISTRY[name].scope is EnvScope.DEPRECATED
    }
```

Change `RuntimeConfig.from_vllm_config` to call `collect_runtime_tuning_env(os.environ)` instead of collecting by prefix. Always set `FASTINFERENCE_PROFILE` to the resolved requested profile.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_kv_default_policy.py tests/test_project_governance.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/engine/env_registry.py vllm/engine/runtime_config.py tests/test_kv_default_policy.py tests/test_project_governance.py
git commit -m "fix: gate legacy fastinference env capture"
```

## Task 3: Route Profile Reads Through Registry

**Files:**
- Modify: `vllm/engine/env_registry.py`
- Modify: `vllm/engine/runtime_profile.py`
- Modify: `tests/test_runtime_profile.py`

- [ ] **Step 1: Write failing tests**

Add tests for profile parsing:

```python
def test_runtime_profile_reads_profile_through_registry(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "accuracy")
    profile = RuntimeProfileRegistry.resolve_from_env(
        model_capabilities=None,
        gpu_total_gb=64.0,
    )
    assert profile.requested_name == "accuracy"
    assert profile.effective_name == "accuracy"


def test_runtime_profile_invalid_profile_falls_back_to_auto(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "unknown")
    profile = RuntimeProfileRegistry.resolve_from_env(
        model_capabilities=None,
        gpu_total_gb=64.0,
    )
    assert profile.requested_name == "auto"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest tests/test_runtime_profile.py -q
```

Expected: direct behavior may already pass; add a governance assertion that `runtime_profile.py` imports `get_public_env` and does not call `os.environ.get("FASTINFERENCE_PROFILE"` directly to force RED.

- [ ] **Step 3: Implement registry helper**

Add to `env_registry.py`:

```python
def get_public_env(
    environ: Mapping[str, str],
    name: str,
    default: str = "",
) -> str:
    spec = FASTINFERENCE_ENV_REGISTRY[name]
    if spec.scope is not EnvScope.PUBLIC:
        raise ValueError(f"{name} is not a public runtime environment variable")
    return environ.get(name, default)
```

Update `RuntimeProfileRegistry.resolve_from_env`:

```python
requested = get_public_env(os.environ, "FASTINFERENCE_PROFILE", "auto").strip().lower()
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_runtime_profile.py tests/test_project_governance.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/engine/env_registry.py vllm/engine/runtime_profile.py tests/test_runtime_profile.py tests/test_project_governance.py
git commit -m "refactor: centralize profile env reads"
```

## Task 4: Move Gemma4 Adapter Off Arbitrary Tuning Env

**Files:**
- Modify: `vllm/adapters/gemma4.py`
- Modify: `tests/test_model_adapter_runtime_policy.py`
- Modify: `tests/test_gemma4_runtime_config_policy.py`

- [ ] **Step 1: Write failing tests**

Assert Gemma4 default policy no longer depends on `runtime_config.tuning_env` unless legacy compatibility was explicitly collected:

```python
def test_gemma4_adapter_ignores_uncollected_process_tuning_env(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_FUSED_STAGE", "off")
    cfg = SimpleNamespace(
        kv_cache_dtype="turbo_int4",
        tuning_env={},
        profile=SimpleNamespace(effective_name="benchmark"),
    )

    policy = Gemma4Adapter().runtime_policy(_gemma4_model_config(), cfg)

    assert policy.kernel_policy["awq_fused_scope"] == "all"
    assert policy.kernel_policy["awq_fused_gemm"] is True
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest tests/test_model_adapter_runtime_policy.py tests/test_gemma4_runtime_config_policy.py -q
```

Expected: if current tests rely on tuning env, failures show which assertions must move behind legacy compatibility.

- [ ] **Step 3: Implement policy source**

Make `Gemma4Adapter.runtime_policy` read only `runtime_config.profile.kernel_policy`, `runtime_config.kernel_policy`, typed `RuntimeConfig` fields, and `runtime_config.tuning_env` entries that came from the compatibility bridge. Keep the current default `fused_stage="all"` and current Gemma4 KV guard behavior.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_model_adapter_runtime_policy.py tests/test_gemma4_runtime_config_policy.py tests/test_e2e_gemma31_overrides.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/adapters/gemma4.py tests/test_model_adapter_runtime_policy.py tests/test_gemma4_runtime_config_policy.py tests/test_e2e_gemma31_overrides.py
git commit -m "refactor: move gemma4 tuning behind runtime policy"
```

## Task 5: Collapse AWQ Env Reads Into Kernel Policy

**Files:**
- Modify: `vllm/model_executor/layers/quantization/tensor.py`
- Modify: `vllm/kernels/triton/awq_fused_gemm.py`
- Modify: `tests/test_awq_fused_scope_policy.py`
- Modify: `tests/test_awq_fused_gemm_heuristics.py`
- Modify: `tests/test_awq_gemm_m1_specialization.py`

- [ ] **Step 1: Write failing tests**

Add `test_awq_kernel_policy_ignores_conflicting_process_env`:

```python
def test_awq_kernel_policy_ignores_conflicting_process_env(monkeypatch) -> None:
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization import tensor

    monkeypatch.setenv("FASTINFERENCE_AWQ_DECODE_GEMV", "0")
    config = SimpleNamespace(kernel_policy={"awq_decode_gemv": True})

    assert tensor._env_awq_decode_gemv_tool_override() is False
    assert tensor._env_awq_decode_gemv(config) is True
```

Add `test_awq_fused_kernel_policy_ignores_conflicting_process_env`:

```python
def test_awq_fused_kernel_policy_ignores_conflicting_process_env(monkeypatch) -> None:
    from types import SimpleNamespace

    from vllm.kernels.triton import awq_fused_gemm

    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K", "0")
    config = SimpleNamespace(kernel_policy={"awq_fused_gemm_split_k": 2})

    assert awq_fused_gemm._env_fused_gemm_split_k_tool_override() is False
    assert awq_fused_gemm._fused_gemm_split_k(config=config) == 2
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest tests/test_awq_fused_scope_policy.py tests/test_awq_fused_gemm_heuristics.py -q
```

Expected: any remaining env-first helper fails when process env conflicts with kernel policy.

- [ ] **Step 3: Implement kernel policy precedence**

For production helpers, resolve values in this order:

1. Explicit `inf_config.kernel_policy`.
2. Runtime profile `kernel_policy`.
3. Built-in defaults.

Keep environment reads only in tool-only helpers and legacy snapshot installers called by tests/tools.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_awq_fused_scope_policy.py tests/test_awq_fused_gemm_heuristics.py tests/test_awq_gemm_m1_specialization.py tests/test_awq_fused_gemm_numerics.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/layers/quantization/tensor.py vllm/kernels/triton/awq_fused_gemm.py tests/test_awq_fused_scope_policy.py tests/test_awq_fused_gemm_heuristics.py tests/test_awq_gemm_m1_specialization.py tests/test_awq_fused_gemm_numerics.py
git commit -m "refactor: route awq tuning through kernel policy"
```

## Task 6: Update Scripts, Docs, and Full Verification

**Files:**
- Modify: `tests/run_inference_correctness_regression.sh`
- Modify: `tests/tools/*.py` affected by removed env names
- Modify: `README.md`
- Modify: `docs/API_REFERENCE.md`
- Modify: `docs/AWQ_POLICY_MATRIX.md`

- [ ] **Step 1: Replace active script env knobs**

Convert active benchmark/test-only env knobs to CLI flags or profile/config usage. Leave historical docs unchanged only if they are clearly archived.

- [ ] **Step 2: Run governance scan**

Run:

```bash
uv run pytest tests/test_env_registry.py tests/test_project_governance.py -q
```

Expected: pass.

- [ ] **Step 3: Run correctness gates**

Run:

```bash
bash tests/run_regression_suite.sh
bash tests/run_inference_correctness_regression.sh
```

Expected: pass.

- [ ] **Step 4: Run performance gate**

Run:

```bash
uv run python tests/tools/profile_gemma4_layer_breakdown.py --model models/gemma-4-31B-it-AWQ-4bit --prompt-tokens 128 --decode-steps 8 --warmup-decode 2 --max-new-tokens 24 --max-num-batched-tokens 512 --max-model-len 512
```

Expected: no material regression from the captured Gemma4-31B baseline.

- [ ] **Step 5: Commit**

```bash
git add tests README.md docs
git commit -m "docs: document consolidated fastinference env"
```

## Self-Review

- Spec coverage: The plan covers registry, broad capture removal, profile routing, Gemma4/model policy migration, AWQ/kernel policy migration, docs/scripts cleanup, and correctness/performance gates.
- Placeholder scan: No `TBD`, `TODO`, or unspecified implementation placeholders remain; later tasks name exact files and verification commands.
- Type consistency: The plan uses existing `RuntimeConfig`, `RuntimeProfile`, `RuntimeModelPolicy`, `model_policy`, and `kernel_policy` names already present in the codebase.
