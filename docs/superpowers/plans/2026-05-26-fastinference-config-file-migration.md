# FastInference Config File Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move production FastInference runtime controls from public `FASTINFERENCE_*` environment variables into explicit config files or API-provided config objects, keeping `FASTINFERENCE_CONFIG` only as a temporary config-file locator.

**Architecture:** Add a small typed config layer in `vllm/engine/fastinference_config.py`, route `RuntimeProfileRegistry` and `RuntimeConfig` through resolved config objects, and update `build_vllm_config()` to accept explicit config inputs. Legacy deprecated env variables remain ignored by default and are admitted only when config sets `legacy_env.enabled = true`.

**Tech Stack:** Python 3.12, stdlib `dataclasses`, stdlib `tomllib`, existing `RuntimeProfile`, `RuntimeConfig`, pytest, uv.

---

## File Structure

- Create `vllm/engine/fastinference_config.py`: typed config dataclasses, TOML loading, strict validation, and resolution precedence.
- Modify `vllm/engine/runtime_profile.py`: add config-based resolver and stop using env in the production path.
- Modify `vllm/engine/runtime_config.py`: read profile/KV/legacy policy from resolved config.
- Modify `vllm/serving/config_builder.py`: accept explicit config object/path and attach resolved config to `VllmConfig`.
- Modify `tests/test_runtime_profile.py`: replace production env selector expectations with config-based expectations.
- Modify `tests/test_kv_default_policy.py`: update RuntimeConfig tests for config-first semantics.
- Create `tests/test_fastinference_config.py`: focused loader, precedence, and validation tests.

## Task 1: Add Config Loader

**Files:**
- Create: `vllm/engine/fastinference_config.py`
- Test: `tests/test_fastinference_config.py`

- [ ] **Step 1: Write failing tests for defaults, TOML loading, precedence, and validation**

Create `tests/test_fastinference_config.py` with:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from vllm.engine.fastinference_config import (
    BenchmarkConfig,
    FastInferenceConfig,
    LegacyEnvConfig,
    load_fastinference_config,
    resolve_fastinference_config,
)


def test_default_fastinference_config_is_benchmark_compatible() -> None:
    cfg = FastInferenceConfig()

    assert cfg.profile == "auto"
    assert cfg.kv_type == "auto"
    assert cfg.debug is False
    assert cfg.log_level == "info"
    assert cfg.benchmark == BenchmarkConfig(profile="default")
    assert cfg.legacy_env == LegacyEnvConfig(enabled=False)


def test_load_fastinference_config_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "fastinference.toml"
    path.write_text(
        '''
profile = "accuracy"
kv_type = "fp8"
debug = true
log_level = "debug"

[benchmark]
profile = "cold"

[legacy_env]
enabled = true
'''.strip(),
        encoding="utf-8",
    )

    cfg = load_fastinference_config(path)

    assert cfg.profile == "accuracy"
    assert cfg.kv_type == "fp8"
    assert cfg.debug is True
    assert cfg.log_level == "debug"
    assert cfg.benchmark.profile == "cold"
    assert cfg.legacy_env.enabled is True


def test_explicit_config_object_wins_over_path_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fastinference.toml"
    path.write_text('profile = "accuracy"\\nkv_type = "fp8"\\n', encoding="utf-8")
    monkeypatch.setenv("FASTINFERENCE_CONFIG", str(path))

    cfg = resolve_fastinference_config(
        config=FastInferenceConfig(profile="latency", kv_type="turbo_int4"),
        path=path,
    )

    assert cfg.profile == "latency"
    assert cfg.kv_type == "turbo_int4"


def test_explicit_path_wins_over_fastinference_config_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit.toml"
    fallback = tmp_path / "fallback.toml"
    explicit.write_text('profile = "throughput"\\n', encoding="utf-8")
    fallback.write_text('profile = "accuracy"\\n', encoding="utf-8")
    monkeypatch.setenv("FASTINFERENCE_CONFIG", str(fallback))

    cfg = resolve_fastinference_config(config=None, path=explicit)

    assert cfg.profile == "throughput"


def test_fastinference_config_env_is_path_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback = tmp_path / "fallback.toml"
    fallback.write_text('profile = "accuracy"\\n', encoding="utf-8")
    monkeypatch.setenv("FASTINFERENCE_CONFIG", str(fallback))

    cfg = resolve_fastinference_config(config=None, path=None)

    assert cfg.profile == "accuracy"


@pytest.mark.parametrize("field,value", [("profile", "bad"), ("kv_type", "bad")])
def test_invalid_enum_values_fail(tmp_path: Path, field: str, value: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(f'{field} = "{value}"\\n', encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_fastinference_config(path)


def test_unknown_top_level_field_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('profile = "auto"\\nunknown = true\\n', encoding="utf-8")

    with pytest.raises(ValueError, match="unknown"):
        load_fastinference_config(path)


def test_missing_config_file_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_fastinference_config(tmp_path / "missing.toml")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_fastinference_config.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vllm.engine.fastinference_config'`.

- [ ] **Step 3: Implement config dataclasses and loader**

Create `vllm/engine/fastinference_config.py` with:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_CONFIG_PROFILES = frozenset(
    {"auto", "latency", "throughput", "accuracy", "benchmark"}
)
SUPPORTED_KV_TYPES = frozenset({"auto", "fp16", "fp8", "turbo_int4"})
_TOP_LEVEL_FIELDS = frozenset(
    {"profile", "kv_type", "debug", "log_level", "benchmark", "legacy_env"}
)


@dataclass(frozen=True)
class BenchmarkConfig:
    profile: str = "default"


@dataclass(frozen=True)
class LegacyEnvConfig:
    enabled: bool = False


@dataclass(frozen=True)
class FastInferenceConfig:
    profile: str = "auto"
    kv_type: str = "auto"
    debug: bool = False
    log_level: str = "info"
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    legacy_env: LegacyEnvConfig = field(default_factory=LegacyEnvConfig)


def _as_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a table")
    return value


def _validate_profile(value: object) -> str:
    profile = str(value or "auto").strip().lower()
    if profile not in SUPPORTED_CONFIG_PROFILES:
        raise ValueError(f"profile must be one of {sorted(SUPPORTED_CONFIG_PROFILES)}")
    return profile


def _validate_kv_type(value: object) -> str:
    kv_type = str(value or "auto").strip().lower()
    if kv_type not in SUPPORTED_KV_TYPES:
        raise ValueError(f"kv_type must be one of {sorted(SUPPORTED_KV_TYPES)}")
    return kv_type


def config_from_mapping(data: Mapping[str, Any]) -> FastInferenceConfig:
    unknown = sorted(set(data) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError("Unknown FastInference config fields: " + ", ".join(unknown))

    benchmark_data = _as_mapping(data.get("benchmark"), "benchmark")
    legacy_data = _as_mapping(data.get("legacy_env"), "legacy_env")

    return FastInferenceConfig(
        profile=_validate_profile(data.get("profile", "auto")),
        kv_type=_validate_kv_type(data.get("kv_type", "auto")),
        debug=bool(data.get("debug", False)),
        log_level=str(data.get("log_level", "info")).strip().lower() or "info",
        benchmark=BenchmarkConfig(
            profile=str(benchmark_data.get("profile", "default")).strip().lower()
            or "default",
        ),
        legacy_env=LegacyEnvConfig(enabled=bool(legacy_data.get("enabled", False))),
    )


def load_fastinference_config(path: str | Path | None) -> FastInferenceConfig:
    if path is None:
        return FastInferenceConfig()
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"FastInference config file not found: {config_path}")
    with config_path.open("rb") as f:
        payload = tomllib.load(f)
    return config_from_mapping(payload)


def resolve_fastinference_config(
    *,
    config: FastInferenceConfig | Mapping[str, Any] | None = None,
    path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> FastInferenceConfig:
    if isinstance(config, FastInferenceConfig):
        return config
    if isinstance(config, Mapping):
        return config_from_mapping(config)

    env = os.environ if environ is None else environ
    resolved_path = path
    if resolved_path is None:
        raw_env_path = str(env.get("FASTINFERENCE_CONFIG", "")).strip()
        resolved_path = raw_env_path or None
    return load_fastinference_config(resolved_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/test_fastinference_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add vllm/engine/fastinference_config.py tests/test_fastinference_config.py
git commit -m "feat: add fastinference config loader"
```

## Task 2: Route RuntimeProfile Through Config

**Files:**
- Modify: `vllm/engine/runtime_profile.py`
- Modify: `tests/test_runtime_profile.py`

- [ ] **Step 1: Update tests for config-based profile resolution**

Modify `tests/test_runtime_profile.py`:

```python
from vllm.engine.fastinference_config import FastInferenceConfig
```

Replace `test_env_profile_is_the_only_fastinference_runtime_selector` with:

```python
def test_config_profile_is_the_runtime_selector(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "accuracy")

    profile = RuntimeProfileRegistry.resolve_from_config(
        FastInferenceConfig(profile="latency"),
        model_capabilities=_caps("llama"),
        gpu_total_gb=24.0,
    )

    assert profile.requested_name == "latency"
    assert profile.effective_name == "latency"
```

Replace `test_unknown_profile_falls_back_to_auto` with:

```python
def test_unknown_profile_falls_back_to_auto_in_pure_resolver() -> None:
    profile = RuntimeProfileRegistry.resolve(
        requested_profile="experimental_local",
        model_capabilities=_caps("llama"),
        gpu_total_gb=24.0,
    )

    assert profile.requested_name == "auto"
    assert profile.effective_name in SUPPORTED_PROFILE_NAMES
```

- [ ] **Step 2: Run updated tests to verify failure**

Run:

```bash
uv run pytest tests/test_runtime_profile.py -q
```

Expected: FAIL because `RuntimeProfileRegistry.resolve_from_config` is missing.

- [ ] **Step 3: Add `resolve_from_config`**

Modify `vllm/engine/runtime_profile.py`:

```python
from vllm.engine.fastinference_config import FastInferenceConfig
```

Add to `RuntimeProfileRegistry`:

```python
    @classmethod
    def resolve_from_config(
        cls,
        config: FastInferenceConfig,
        *,
        model_capabilities: Any | None,
        gpu_total_gb: float,
    ) -> RuntimeProfile:
        return cls.resolve(
            requested_profile=config.profile,
            model_capabilities=model_capabilities,
            gpu_total_gb=gpu_total_gb,
        )
```

Keep `resolve_from_env()` during transition, but production code will stop calling it in Task 3.

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/test_runtime_profile.py tests/test_fastinference_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add vllm/engine/runtime_profile.py tests/test_runtime_profile.py
git commit -m "refactor: resolve runtime profile from config"
```

## Task 3: Route RuntimeConfig Through Config

**Files:**
- Modify: `vllm/engine/runtime_config.py`
- Modify: `tests/test_kv_default_policy.py`

- [ ] **Step 1: Update RuntimeConfig tests for config-first semantics**

Modify `tests/test_kv_default_policy.py`:

```python
from vllm.engine.fastinference_config import FastInferenceConfig
```

In `_mock_vllm_config()`, add:

```python
        fastinference_config=FastInferenceConfig(),
```

Replace public env override tests with config tests:

```python
def test_runtime_config_config_kv_type_overrides_profile(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "fp16")
    vllm_config = _mock_vllm_config()
    vllm_config.fastinference_config = FastInferenceConfig(kv_type="fp8")

    cfg = RuntimeConfig.from_vllm_config(vllm_config)

    assert cfg.profile.requested_name == "auto"
    assert cfg.kv_cache_dtype == "fp8"
    assert cfg.tuning_env == {"FASTINFERENCE_PROFILE": "auto"}
```

```python
def test_runtime_config_config_kv_type_auto_uses_profile(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "benchmark")
    vllm_config = _mock_vllm_config()
    vllm_config.fastinference_config = FastInferenceConfig(
        profile="accuracy",
        kv_type="auto",
    )

    cfg = RuntimeConfig.from_vllm_config(vllm_config)

    assert cfg.profile.requested_name == "accuracy"
    assert cfg.kv_cache_dtype == "fp8"
```

```python
def test_runtime_config_collects_deprecated_env_when_config_compat_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_ALLOW_LEGACY_ENV", "1")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_ALLOW_INT4_KV", "1")
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "fp16")
    vllm_config = _mock_vllm_config()
    vllm_config.fastinference_config = FastInferenceConfig(
        legacy_env=LegacyEnvConfig(enabled=True),
    )

    cfg = RuntimeConfig.from_vllm_config(vllm_config)

    assert cfg.tuning_env["FASTINFERENCE_GEMMA4_ALLOW_INT4_KV"] == "1"
    assert "FASTINFERENCE_KV_TYPE" not in cfg.tuning_env
    assert cfg.tuning_env["FASTINFERENCE_PROFILE"] == "auto"
```

Update profile tests to set `vllm_config.fastinference_config = FastInferenceConfig(profile="accuracy")` instead of setting `FASTINFERENCE_PROFILE`.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_kv_default_policy.py -q
```

Expected: FAIL because `RuntimeConfig.from_vllm_config()` still reads profile/KV/legacy from env.

- [ ] **Step 3: Implement config-first RuntimeConfig**

Modify `vllm/engine/runtime_config.py` imports:

```python
from vllm.engine.env_registry import collect_runtime_tuning_env
from vllm.engine.fastinference_config import (
    FastInferenceConfig,
    resolve_fastinference_config,
)
```

In `from_vllm_config`, resolve config before profile:

```python
        fastinference_config = resolve_fastinference_config(
            config=getattr(vllm_config, "fastinference_config", None),
            path=getattr(vllm_config, "fastinference_config_path", None),
        )
        profile = RuntimeProfileRegistry.resolve_from_config(
            fastinference_config,
            model_capabilities=getattr(vllm_config, "model_capabilities", None),
            gpu_total_gb=get_total_gpu_memory_gb(),
        )
        tuning_env = (
            collect_runtime_tuning_env(os.environ)
            if fastinference_config.legacy_env.enabled
            else {}
        )
        tuning_env["FASTINFERENCE_PROFILE"] = profile.requested_name
        kv_cache_dtype = fastinference_config.kv_type.strip().lower()
        if kv_cache_dtype == "auto":
            kv_cache_dtype = profile.kv_cache_dtype
```

Remove `get_public_env` from imports.

- [ ] **Step 4: Run targeted tests**

Run:

```bash
uv run pytest tests/test_kv_default_policy.py tests/test_runtime_profile.py tests/test_fastinference_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add vllm/engine/runtime_config.py tests/test_kv_default_policy.py
git commit -m "refactor: build runtime config from fastinference config"
```

## Task 4: Add Config Inputs to Config Builder

**Files:**
- Modify: `vllm/serving/config_builder.py`
- Test: add to `tests/test_kv_default_policy.py` or create `tests/test_config_builder_fastinference_config.py`

- [ ] **Step 1: Write tests for explicit object/path and env fallback**

Create `tests/test_config_builder_fastinference_config.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from vllm.engine.fastinference_config import FastInferenceConfig
from vllm.serving import config_builder


def _fake_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="llama",
        max_position_embeddings=1024,
        quantization_config=None,
    )


def test_build_vllm_config_accepts_fastinference_config_object(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config_builder, "load_hf_config", lambda _path: _fake_hf_config())
    monkeypatch.setattr(config_builder.os.path, "isfile", lambda _path: False)
    monkeypatch.setattr(config_builder.os, "listdir", lambda _path: [])

    cfg = config_builder.build_vllm_config(
        "models/mock",
        fastinference_config=FastInferenceConfig(profile="accuracy", kv_type="fp8"),
    )

    assert cfg.fastinference_config.profile == "accuracy"
    assert cfg.runtime_config.profile.requested_name == "accuracy"
    assert cfg.runtime_config.kv_cache_dtype == "fp8"


def test_build_vllm_config_accepts_fastinference_config_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(config_builder, "load_hf_config", lambda _path: _fake_hf_config())
    monkeypatch.setattr(config_builder.os.path, "isfile", lambda _path: False)
    monkeypatch.setattr(config_builder.os, "listdir", lambda _path: [])
    path = tmp_path / "fastinference.toml"
    path.write_text('profile = "latency"\\nkv_type = "auto"\\n', encoding="utf-8")

    cfg = config_builder.build_vllm_config(
        "models/mock",
        fastinference_config_path=path,
    )

    assert cfg.fastinference_config.profile == "latency"
    assert cfg.runtime_config.profile.requested_name == "latency"
    assert cfg.runtime_config.kv_max_active_requests == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_config_builder_fastinference_config.py -q
```

Expected: FAIL because `build_vllm_config()` does not attach the resolved config.

- [ ] **Step 3: Resolve and attach config in `build_vllm_config()`**

Modify `vllm/serving/config_builder.py`:

```python
from vllm.engine.fastinference_config import resolve_fastinference_config
```

Before `RuntimeConfig.from_vllm_config(vllm_config)`:

```python
    fastinference_config = resolve_fastinference_config(
        config=kwargs.get("fastinference_config"),
        path=kwargs.get("fastinference_config_path"),
    )
    vllm_config.fastinference_config = fastinference_config
    vllm_config.fastinference_config_path = kwargs.get("fastinference_config_path")
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
uv run pytest tests/test_config_builder_fastinference_config.py tests/test_kv_default_policy.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add vllm/serving/config_builder.py tests/test_config_builder_fastinference_config.py
git commit -m "feat: accept fastinference config in config builder"
```

## Task 5: Governance and Compatibility Cleanup

**Files:**
- Modify: `tests/test_project_governance.py`
- Modify: `vllm/engine/env_registry.py` if public surface policy needs to reflect only transitional `FASTINFERENCE_CONFIG`.

- [ ] **Step 1: Inspect existing governance tests**

Run:

```bash
sed -n '1,320p' tests/test_project_governance.py
```

Expected: identify tests asserting seven final public env names.

- [ ] **Step 2: Update governance assertions**

Change the expected long-term public runtime env set to:

```python
assert FINAL_PUBLIC_FASTINFERENCE_ENV == {
    "FASTINFERENCE_CONFIG",
}
```

If existing tests need transition naming, use a test name that makes it clear `FASTINFERENCE_CONFIG` is transitional and path-only:

```python
def test_only_transitional_config_locator_remains_public() -> None:
    assert FINAL_PUBLIC_FASTINFERENCE_ENV == {"FASTINFERENCE_CONFIG"}
```

- [ ] **Step 3: Update registry public/deprecated scopes**

In `vllm/engine/env_registry.py`, keep only `FASTINFERENCE_CONFIG` in `FINAL_PUBLIC_FASTINFERENCE_ENV`. Mark previous public control variables as deprecated:

```python
"FASTINFERENCE_PROFILE": _deprecated(
    "FASTINFERENCE_PROFILE",
    replacement="FastInference config field: profile",
),
"FASTINFERENCE_KV_TYPE": _deprecated(
    "FASTINFERENCE_KV_TYPE",
    replacement="FastInference config field: kv_type",
),
"FASTINFERENCE_DEBUG": _deprecated(
    "FASTINFERENCE_DEBUG",
    replacement="FastInference config field: debug",
),
"FASTINFERENCE_LOG_LEVEL": _deprecated(
    "FASTINFERENCE_LOG_LEVEL",
    replacement="FastInference config field: log_level",
),
"FASTINFERENCE_BENCH_PROFILE": _deprecated(
    "FASTINFERENCE_BENCH_PROFILE",
    replacement="FastInference config field: benchmark.profile",
),
"FASTINFERENCE_ALLOW_LEGACY_ENV": _deprecated(
    "FASTINFERENCE_ALLOW_LEGACY_ENV",
    replacement="FastInference config field: legacy_env.enabled",
),
```

Do not collect these deprecated former-public variables into `tuning_env` unless they are needed by existing adapter/kernel snapshots. They are not needed for Task 5.

- [ ] **Step 4: Run governance tests**

Run:

```bash
uv run pytest tests/test_project_governance.py tests/test_fastinference_config.py tests/test_kv_default_policy.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```bash
git add vllm/engine/env_registry.py tests/test_project_governance.py
git commit -m "refactor: keep only config locator public"
```

## Task 6: Verification

**Files:**
- No code changes unless tests expose a real issue.

- [ ] **Step 1: Run fast regression suite**

Run:

```bash
bash tests/run_regression_suite.sh
```

Expected: PASS.

- [ ] **Step 2: Run focused compile check**

Run:

```bash
uv run python -m py_compile \
  vllm/engine/fastinference_config.py \
  vllm/engine/runtime_profile.py \
  vllm/engine/runtime_config.py \
  vllm/serving/config_builder.py
```

Expected: PASS with no output.

- [ ] **Step 3: Commit any verification-only fixes**

If no code changed, skip this step. If a fix was needed:

```bash
git add <changed-files>
git commit -m "fix: stabilize fastinference config migration"
```

- [ ] **Step 4: Decide heavy regression timing**

Run full correctness and performance only after Tasks 1-5 are green:

```bash
bash tests/run_inference_correctness_regression.sh
uv run python tests/e2e_full_benchmark.py --models tinyllama,qwen35_9b_awq,gemma4_26b_a4b,gemma4_31b_q4 --model-process-isolation
```

Expected: correctness PASS; performance JSON generated with no script failure. If Qwen performance remains slow but script completes, record it as a known performance concern rather than a config migration failure.
