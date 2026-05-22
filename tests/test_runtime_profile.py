# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from vllm.engine.runtime_profile import (
    RuntimeProfileRegistry,
    SUPPORTED_PROFILE_NAMES,
)


def _caps(model_type: str = "llama") -> SimpleNamespace:
    return SimpleNamespace(
        model_type=model_type,
        supports_moe=False,
        supports_int4_kv=True,
        supports_fp8_kv=True,
        max_model_len=4096,
    )


def test_supported_profile_names_are_stable() -> None:
    assert SUPPORTED_PROFILE_NAMES == (
        "auto",
        "latency",
        "throughput",
        "accuracy",
        "benchmark",
    )


def test_auto_profile_resolves_to_named_effective_profile(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_PROFILE", raising=False)
    profile = RuntimeProfileRegistry.resolve(
        requested_profile=None,
        model_capabilities=_caps("llama"),
        gpu_total_gb=24.0,
    )

    assert profile.requested_name == "auto"
    assert profile.effective_name in SUPPORTED_PROFILE_NAMES
    assert profile.kv_cache_dtype == "turbo_int4"
    assert profile.block_size == 16
    assert profile.backend_policy.gpu_greedy_sampling is True


def test_env_profile_is_the_only_fastinference_runtime_selector(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "accuracy")
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "fp16")

    profile = RuntimeProfileRegistry.resolve_from_env(
        model_capabilities=_caps("llama"),
        gpu_total_gb=24.0,
    )

    assert profile.requested_name == "accuracy"
    assert profile.kv_cache_dtype == "fp8"


def test_unknown_profile_falls_back_to_auto(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "experimental_local")
    profile = RuntimeProfileRegistry.resolve_from_env(
        model_capabilities=_caps("llama"),
        gpu_total_gb=24.0,
    )

    assert profile.requested_name == "auto"
    assert profile.effective_name in SUPPORTED_PROFILE_NAMES
