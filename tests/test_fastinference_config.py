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
        """
profile = "accuracy"
kv_type = "fp8"
debug = true
log_level = "debug"

[benchmark]
profile = "cold"

[legacy_env]
enabled = true
""".strip(),
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
    path.write_text('profile = "accuracy"\nkv_type = "fp8"\n', encoding="utf-8")
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
    explicit.write_text('profile = "throughput"\n', encoding="utf-8")
    fallback.write_text('profile = "accuracy"\n', encoding="utf-8")
    monkeypatch.setenv("FASTINFERENCE_CONFIG", str(fallback))

    cfg = resolve_fastinference_config(config=None, path=explicit)

    assert cfg.profile == "throughput"


def test_fastinference_config_env_is_path_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback = tmp_path / "fallback.toml"
    fallback.write_text('profile = "accuracy"\n', encoding="utf-8")
    monkeypatch.setenv("FASTINFERENCE_CONFIG", str(fallback))

    cfg = resolve_fastinference_config(config=None, path=None)

    assert cfg.profile == "accuracy"


@pytest.mark.parametrize("field,value", [("profile", "bad"), ("kv_type", "bad")])
def test_invalid_enum_values_fail(tmp_path: Path, field: str, value: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(f'{field} = "{value}"\n', encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_fastinference_config(path)


def test_unknown_top_level_field_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('profile = "auto"\nunknown = true\n', encoding="utf-8")

    with pytest.raises(ValueError, match="unknown"):
        load_fastinference_config(path)


def test_unknown_tuning_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    unknown_key = "FASTINFERENCE_" + "TYP0"
    path.write_text(
        f'[tuning_keyvals]\n{unknown_key} = "1"\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match=unknown_key):
        load_fastinference_config(path)


def test_missing_config_file_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_fastinference_config(tmp_path / "missing.toml")
