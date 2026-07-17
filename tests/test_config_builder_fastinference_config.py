# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.engine.fastinference_config import (
    FastInferenceConfig,
    ModelLoadConfig,
    config_from_mapping,
)
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
    monkeypatch.setattr(
        config_builder, "load_hf_config", lambda _path, **_kwargs: _fake_hf_config()
    )
    monkeypatch.setattr(config_builder.os.path, "isfile", lambda _path: False)
    monkeypatch.setattr(config_builder.os, "listdir", lambda _path: [])

    cfg = config_builder.build_vllm_config(
        "models/mock",
        fastinference_config=FastInferenceConfig(profile="accuracy", kv_type="fp8"),
    )

    assert cfg.fastinference_config.profile == "accuracy"
    assert cfg.runtime_config.profile.requested_name == "accuracy"
    assert cfg.runtime_config.kv_cache_dtype == "fp8"
    assert cfg.cache_config.cache_dtype == "fp8"


def test_build_vllm_config_accepts_fastinference_config_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        config_builder, "load_hf_config", lambda _path, **_kwargs: _fake_hf_config()
    )
    monkeypatch.setattr(config_builder.os.path, "isfile", lambda _path: False)
    monkeypatch.setattr(config_builder.os, "listdir", lambda _path: [])
    path = tmp_path / "fastinference.toml"
    path.write_text('profile = "latency"\nkv_type = "auto"\n', encoding="utf-8")

    cfg = config_builder.build_vllm_config(
        "models/mock",
        fastinference_config_path=path,
    )

    assert cfg.fastinference_config.profile == "latency"
    assert cfg.runtime_config.profile.requested_name == "latency"
    assert cfg.runtime_config.kv_max_active_requests == 1


def test_build_vllm_config_applies_runtime_int4_kv_dtype_to_cache_config(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        config_builder, "load_hf_config", lambda _path, **_kwargs: _fake_hf_config()
    )
    monkeypatch.setattr(config_builder.os.path, "isfile", lambda _path: False)
    monkeypatch.setattr(config_builder.os, "listdir", lambda _path: [])

    cfg = config_builder.build_vllm_config(
        "models/mock",
        fastinference_config=FastInferenceConfig(kv_type="turbo_int4"),
    )

    assert cfg.runtime_config.kv_cache_dtype == "turbo_int4"
    assert cfg.cache_config.cache_dtype == "int4"


def test_remote_code_requires_a_pinned_revision() -> None:
    with pytest.raises(ValueError, match="model.revision"):
        config_from_mapping({"model": {"trust_remote_code": True}})


def test_build_vllm_config_uses_explicit_remote_code_policy(monkeypatch) -> None:
    captured = {}

    def fake_load_hf_config(path: str, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return _fake_hf_config()

    monkeypatch.setattr(config_builder, "load_hf_config", fake_load_hf_config)
    monkeypatch.setattr(config_builder.os.path, "isfile", lambda _path: False)
    monkeypatch.setattr(config_builder.os, "listdir", lambda _path: [])

    cfg = config_builder.build_vllm_config(
        "models/mock",
        fastinference_config=FastInferenceConfig(
            model=ModelLoadConfig(
                trust_remote_code=True,
                revision="deadbeef",
            )
        ),
    )

    assert captured == {
        "path": "models/mock",
        "trust_remote_code": True,
        "revision": "deadbeef",
    }
    assert cfg.model_config.trust_remote_code is True
    assert cfg.model_config.revision == "deadbeef"


def test_build_vllm_config_rejects_unpinned_remote_code(monkeypatch) -> None:
    monkeypatch.setattr(
        config_builder, "load_hf_config", lambda _path, **_kwargs: _fake_hf_config()
    )

    with pytest.raises(ValueError, match="model revision"):
        config_builder.build_vllm_config("models/mock", trust_remote_code=True)
