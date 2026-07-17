# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from vllm.engine.fastinference_config import FastInferenceConfig
from vllm.engine.runtime_profile import (
    SUPPORTED_PROFILE_NAMES,
    RuntimeProfile,
    RuntimeProfileRegistry,
)


def test_supported_profile_names_are_stable() -> None:
    assert SUPPORTED_PROFILE_NAMES == (
        "auto",
        "balanced",
        "latency",
        "throughput",
        "accuracy",
        "benchmark",
    )


def test_auto_profile_resolves_to_named_effective_profile(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_PROFILE", raising=False)
    profile = RuntimeProfileRegistry.resolve(requested_profile=None)

    assert profile.requested_name == "auto"
    assert profile.effective_name == "balanced"
    assert profile.kv_cache_dtype == "turbo_int4"
    assert profile.block_size == 16
    assert profile.backend_policy.gpu_greedy_sampling is True


def test_config_profile_is_the_runtime_selector(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "accuracy")

    profile = RuntimeProfileRegistry.resolve_from_config(
        FastInferenceConfig(profile="latency")
    )

    assert profile.requested_name == "latency"
    assert profile.effective_name == "latency"


def test_unknown_profile_falls_back_to_auto_in_pure_resolver() -> None:
    profile = RuntimeProfileRegistry.resolve(requested_profile="experimental_local")

    assert profile.requested_name == "auto"
    assert profile.effective_name in SUPPORTED_PROFILE_NAMES


def test_latency_profile_caps_active_work_and_enables_gpu_greedy() -> None:
    profile = RuntimeProfileRegistry.resolve(requested_profile="latency")

    assert profile.requested_name == "latency"
    assert profile.effective_name == "latency"
    assert profile.kv_max_active_requests == 1
    assert profile.kv_max_model_len == 512
    assert profile.backend_policy.gpu_greedy_sampling is True
    assert profile.backend_policy.gpu_greedy_max_tokens_only is True
    assert profile.backend_policy.gpu_greedy_bypass_cpu_policies is True


def test_throughput_profile_uses_batched_defaults() -> None:
    profile = RuntimeProfileRegistry.resolve(requested_profile="throughput")

    assert profile.requested_name == "throughput"
    assert profile.effective_name == "throughput"
    assert profile.kv_max_active_requests == 16
    assert profile.prefill_microbatch_size == 4


def test_balanced_profile_is_the_default_service_policy() -> None:
    profile = RuntimeProfileRegistry.resolve(requested_profile="balanced")

    assert profile.effective_name == "balanced"
    assert profile.backend_policy.gpu_greedy_sampling is True


def test_benchmark_profile_uses_turbo_int4_and_gpu_greedy() -> None:
    profile = RuntimeProfileRegistry.resolve(requested_profile="benchmark")

    assert profile.requested_name == "benchmark"
    assert profile.effective_name == "benchmark"
    assert profile.kv_cache_dtype == "turbo_int4"
    assert profile.backend_policy.gpu_greedy_sampling is True
    assert profile.backend_policy.gpu_greedy_max_tokens_only is True
    assert profile.backend_policy.gpu_greedy_bypass_cpu_policies is True


def test_runtime_profile_policy_dicts_are_immutable_snapshots() -> None:
    model_policy = {"family": "llama"}
    kernel_policy = {"paged_attention": "default"}
    profile = RuntimeProfile(
        requested_name="benchmark",
        effective_name="benchmark",
        description="test",
        model_policy=model_policy,
        kernel_policy=kernel_policy,
    )

    model_policy["family"] = "mutated"
    kernel_policy["paged_attention"] = "mutated"

    assert profile.model_policy["family"] == "llama"
    assert profile.kernel_policy["paged_attention"] == "default"
    with pytest.raises(TypeError):
        profile.model_policy["family"] = "mutated"
    with pytest.raises(TypeError):
        profile.kernel_policy["paged_attention"] = "mutated"


def test_runtime_profile_nested_policy_values_are_immutable_snapshots() -> None:
    model_nested = {"y": 1}
    model_list = ["prefill"]
    model_set = {"dense"}
    kernel_nested = {"tile": [16, 16]}
    model_policy = {
        "nested": model_nested,
        "list": model_list,
        "set": model_set,
    }
    kernel_policy = {"nested": kernel_nested}
    profile = RuntimeProfile(
        requested_name="benchmark",
        effective_name="benchmark",
        description="test",
        model_policy=model_policy,
        kernel_policy=kernel_policy,
    )

    model_nested["y"] = 2
    model_list.append("decode")
    model_set.add("moe")
    kernel_nested["tile"].append(32)

    assert profile.model_policy["nested"]["y"] == 1
    assert profile.model_policy["list"] == ("prefill",)
    assert profile.model_policy["set"] == frozenset({"dense"})
    assert profile.kernel_policy["nested"]["tile"] == (16, 16)
    with pytest.raises(TypeError):
        profile.model_policy["nested"]["y"] = 2
    with pytest.raises(AttributeError):
        profile.model_policy["list"].append("decode")
    with pytest.raises(AttributeError):
        profile.model_policy["set"].add("moe")
    with pytest.raises(TypeError):
        profile.kernel_policy["nested"]["tile"][0] = 32
