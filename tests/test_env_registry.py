# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.engine.env_registry import (
    FASTINFERENCE_ENV_REGISTRY,
    FINAL_PUBLIC_FASTINFERENCE_ENV,
    EnvScope,
)


def test_final_public_fastinference_env_is_below_limit() -> None:
    assert len(FINAL_PUBLIC_FASTINFERENCE_ENV) < 10
    assert {"FASTINFERENCE_CONFIG"} == FINAL_PUBLIC_FASTINFERENCE_ENV


def test_kv_fp8_is_deprecated_alias_not_final_public() -> None:
    spec = FASTINFERENCE_ENV_REGISTRY["FASTINFERENCE_KV_FP8"]
    assert spec.scope is EnvScope.DEPRECATED
    assert spec.replacement == "FASTINFERENCE_KV_TYPE=fp8"
    assert "FASTINFERENCE_KV_FP8" not in FINAL_PUBLIC_FASTINFERENCE_ENV


def test_all_final_public_env_names_are_registered_as_public() -> None:
    for name in FINAL_PUBLIC_FASTINFERENCE_ENV:
        assert FASTINFERENCE_ENV_REGISTRY[name].scope is EnvScope.PUBLIC


def test_previous_public_runtime_controls_are_deprecated() -> None:
    expected = {
        "FASTINFERENCE_ALLOW_LEGACY_ENV",
        "FASTINFERENCE_BENCH_PROFILE",
        "FASTINFERENCE_DEBUG",
        "FASTINFERENCE_KV_TYPE",
        "FASTINFERENCE_LOG_LEVEL",
        "FASTINFERENCE_PROFILE",
    }

    for name in expected:
        assert FASTINFERENCE_ENV_REGISTRY[name].scope is EnvScope.DEPRECATED
        assert name not in FINAL_PUBLIC_FASTINFERENCE_ENV
