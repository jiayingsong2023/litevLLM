# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from vllm.engine.env_registry import get_public_env
from vllm.engine.fastinference_config import FastInferenceConfig
from vllm.engine.runtime_policy import BackendRuntimePolicy, SchedulerRuntimePolicy

SUPPORTED_PROFILE_NAMES = (
    "auto",
    "latency",
    "throughput",
    "accuracy",
    "benchmark",
)


def _freeze_policy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_policy_value(nested) for key, nested in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_policy_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_policy_value(item) for item in value)
    return value


@dataclass(frozen=True)
class RuntimeProfile:
    requested_name: str
    effective_name: str
    description: str
    kv_cache_dtype: str = "turbo_int4"
    block_size: int = 16
    kv_max_model_len: int | None = None
    kv_max_active_requests: int = 4
    fusion_level: int = 2
    enable_decode_priority: bool = True
    prefill_chunk_size: int = 0
    prefill_reserved_tokens: int = 0
    prefill_reserve_backlog: int = 2
    prefill_catchup_ratio: float = 0.25
    prefill_microbatch_size: int = 2
    min_prefill_chunk_size: int = 128
    max_prefill_chunk_size: int | None = None
    prefill_sla_ttft_ms: float = 2000.0
    default_min_new_tokens: int = 0
    queue_timeout_s: float = 30.0
    memory_audit_topn: int = 20
    k_scale: float = 1.0
    v_scale: float = 1.0
    use_prompt_guard: bool = True
    scheduler_policy: SchedulerRuntimePolicy = field(
        default_factory=SchedulerRuntimePolicy
    )
    backend_policy: BackendRuntimePolicy = field(default_factory=BackendRuntimePolicy)
    model_policy: Mapping[str, Any] = field(default_factory=dict)
    kernel_policy: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_policy",
            _freeze_policy_value(self.model_policy),
        )
        object.__setattr__(
            self,
            "kernel_policy",
            _freeze_policy_value(self.kernel_policy),
        )

    def stats(self) -> dict[str, Any]:
        return {
            "requested": self.requested_name,
            "effective": self.effective_name,
            "description": self.description,
            "kv_cache_dtype": self.kv_cache_dtype,
            "block_size": self.block_size,
            "kv_max_model_len": self.kv_max_model_len,
            "kv_max_active_requests": self.kv_max_active_requests,
            "fusion_level": self.fusion_level,
        }


class RuntimeProfileRegistry:
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

    @classmethod
    def resolve_from_env(
        cls,
        *,
        model_capabilities: Any | None,
        gpu_total_gb: float,
    ) -> RuntimeProfile:
        requested = (
            get_public_env(os.environ, "FASTINFERENCE_PROFILE", "auto").strip().lower()
        )
        return cls.resolve(
            requested_profile=requested,
            model_capabilities=model_capabilities,
            gpu_total_gb=gpu_total_gb,
        )

    @classmethod
    def resolve(
        cls,
        *,
        requested_profile: str | None,
        model_capabilities: Any | None,
        gpu_total_gb: float,
    ) -> RuntimeProfile:
        requested = (requested_profile or "auto").strip().lower()
        if requested not in SUPPORTED_PROFILE_NAMES:
            requested = "auto"
        effective = cls._effective_name(requested, model_capabilities, gpu_total_gb)
        return cls._build(requested_name=requested, effective_name=effective)

    @staticmethod
    def _effective_name(
        requested: str,
        model_capabilities: Any | None,
        gpu_total_gb: float,
    ) -> str:
        del model_capabilities, gpu_total_gb
        if requested == "auto":
            return "benchmark"
        return requested

    @staticmethod
    def _build(*, requested_name: str, effective_name: str) -> RuntimeProfile:
        if effective_name == "accuracy":
            return RuntimeProfile(
                requested_name=requested_name,
                effective_name=effective_name,
                description="Accuracy-first profile with conservative KV policy.",
                kv_cache_dtype="fp8",
            )
        if effective_name == "latency":
            return RuntimeProfile(
                requested_name=requested_name,
                effective_name=effective_name,
                description="Single-request and low-TPOT profile.",
                kv_cache_dtype="turbo_int4",
                kv_max_active_requests=1,
                kv_max_model_len=512,
                backend_policy=BackendRuntimePolicy(
                    gpu_greedy_sampling=True,
                    gpu_greedy_max_tokens_only=True,
                    gpu_greedy_bypass_cpu_policies=True,
                ),
            )
        if effective_name == "throughput":
            return RuntimeProfile(
                requested_name=requested_name,
                effective_name=effective_name,
                description="Batch throughput profile for supported dense shapes.",
                kv_cache_dtype="turbo_int4",
                kv_max_active_requests=16,
                prefill_microbatch_size=4,
            )
        return RuntimeProfile(
            requested_name=requested_name,
            effective_name=effective_name,
            description="Benchmark-recommended stable production profile.",
            kv_cache_dtype="turbo_int4",
            backend_policy=BackendRuntimePolicy(
                gpu_greedy_sampling=True,
                gpu_greedy_max_tokens_only=True,
                gpu_greedy_bypass_cpu_policies=True,
            ),
        )
