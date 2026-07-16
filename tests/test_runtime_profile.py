# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.engine.fastinference_config import FastInferenceConfig
from vllm.engine.runtime_policy import BackendRuntimePolicy, SchedulerRuntimePolicy
from vllm.engine.runtime_profile import (
    SUPPORTED_PROFILE_NAMES,
    RuntimeProfile,
    RuntimeProfileRegistry,
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
        "balanced",
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
    assert profile.effective_name == "balanced"
    assert profile.kv_cache_dtype == "turbo_int4"
    assert profile.block_size == 16
    assert profile.backend_policy.gpu_greedy_sampling is True


def test_config_profile_is_the_runtime_selector(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "accuracy")

    profile = RuntimeProfileRegistry.resolve_from_config(
        FastInferenceConfig(profile="latency"),
        model_capabilities=_caps("llama"),
        gpu_total_gb=24.0,
    )

    assert profile.requested_name == "latency"
    assert profile.effective_name == "latency"


def test_unknown_profile_falls_back_to_auto_in_pure_resolver() -> None:
    profile = RuntimeProfileRegistry.resolve(
        requested_profile="experimental_local",
        model_capabilities=_caps("llama"),
        gpu_total_gb=24.0,
    )

    assert profile.requested_name == "auto"
    assert profile.effective_name in SUPPORTED_PROFILE_NAMES


def test_latency_profile_caps_active_work_and_enables_gpu_greedy() -> None:
    profile = RuntimeProfileRegistry.resolve(
        requested_profile="latency",
        model_capabilities=_caps("llama"),
        gpu_total_gb=24.0,
    )

    assert profile.requested_name == "latency"
    assert profile.effective_name == "latency"
    assert profile.kv_max_active_requests == 1
    assert profile.kv_max_model_len == 512
    assert profile.backend_policy.gpu_greedy_sampling is True
    assert profile.backend_policy.gpu_greedy_max_tokens_only is True
    assert profile.backend_policy.gpu_greedy_bypass_cpu_policies is True


def test_throughput_profile_uses_batched_defaults() -> None:
    profile = RuntimeProfileRegistry.resolve(
        requested_profile="throughput",
        model_capabilities=_caps("llama"),
        gpu_total_gb=24.0,
    )

    assert profile.requested_name == "throughput"
    assert profile.effective_name == "throughput"
    assert profile.kv_max_active_requests == 16
    assert profile.prefill_microbatch_size == 4


def test_balanced_profile_is_the_default_service_policy() -> None:
    profile = RuntimeProfileRegistry.resolve(
        requested_profile="balanced",
        model_capabilities=_caps("llama"),
        gpu_total_gb=24.0,
    )

    assert profile.effective_name == "balanced"
    assert profile.backend_policy.gpu_greedy_sampling is True


def test_benchmark_profile_uses_turbo_int4_and_gpu_greedy() -> None:
    profile = RuntimeProfileRegistry.resolve(
        requested_profile="benchmark",
        model_capabilities=_caps("llama"),
        gpu_total_gb=24.0,
    )

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


def test_runtime_policy_collections_are_immutable_snapshots() -> None:
    service_weights = {"gold": 2}
    fairness_classes = {"gold"}
    preemptible_classes = {"silver"}
    scheduler_policy = SchedulerRuntimePolicy(
        service_class_weights=service_weights,
        fairness_guardrail_service_classes=fairness_classes,
    )
    backend_policy = BackendRuntimePolicy(
        preemptible_service_classes=preemptible_classes,
    )

    service_weights["gold"] = 4
    fairness_classes.add("silver")
    preemptible_classes.add("bronze")

    assert scheduler_policy.service_class_weights == {"gold": 2}
    assert scheduler_policy.fairness_guardrail_service_classes == frozenset({"gold"})
    assert backend_policy.preemptible_service_classes == frozenset({"silver"})
    with pytest.raises(TypeError):
        scheduler_policy.service_class_weights["gold"] = 4
    with pytest.raises(AttributeError):
        scheduler_policy.fairness_guardrail_service_classes.add("silver")
    with pytest.raises(AttributeError):
        backend_policy.preemptible_service_classes.add("bronze")
