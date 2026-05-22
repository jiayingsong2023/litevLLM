# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from types import MappingProxyType


def _freeze_optional_mapping(
    value: Mapping[str, int] | None,
) -> Mapping[str, int] | None:
    if value is None:
        return None
    return MappingProxyType(dict(value))


def _freeze_optional_set(
    value: AbstractSet[str] | None,
) -> frozenset[str] | None:
    if value is None:
        return None
    return frozenset(value)


@dataclass(frozen=True)
class SchedulerRuntimePolicy:
    max_decode_streak: int = 4
    queue_aging_threshold_s: float = 2.0
    max_prefill_deferrals: int = 2
    service_class_weights: Mapping[str, int] | None = None
    admission_service_class_quotas: Mapping[str, int] | None = None
    decode_service_class_quotas: Mapping[str, int] | None = None
    fairness_guardrail_queue_wait_s: float = 0.0
    fairness_guardrail_service_classes: AbstractSet[str] | None = None
    max_admit_lora_adapters_per_step: int = 0
    max_prefill_lora_adapters_per_batch: int = 0
    max_decode_lora_adapters_per_batch: int = 0
    lora_fairness_relax_threshold: float = 0.0
    lora_locality_tighten_threshold: float = 0.0
    lora_limit_relax_delta: int = 1
    lora_limit_tighten_delta: int = 1
    max_admit_multimodal_per_step: int = 0
    max_prefill_multimodal_requests_per_batch: int = 0
    max_decode_multimodal_requests_per_batch: int = 0
    max_admit_multimodal_lora_per_step: int = 0
    max_prefill_multimodal_lora_requests_per_batch: int = 0
    max_decode_multimodal_lora_requests_per_batch: int = 0
    multimodal_prefix_cache_relax_threshold: float = 0.0
    multimodal_prefix_cache_tighten_threshold: float = 0.0
    multimodal_prefill_limit_relax_delta: int = 1
    multimodal_prefill_limit_tighten_delta: int = 1
    multimodal_lora_prefill_limit_relax_delta: int = 1
    multimodal_lora_prefill_limit_tighten_delta: int = 1
    multimodal_lora_fairness_relax_threshold: float = 0.0
    multimodal_lora_locality_tighten_threshold: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "service_class_weights",
            _freeze_optional_mapping(self.service_class_weights),
        )
        object.__setattr__(
            self,
            "admission_service_class_quotas",
            _freeze_optional_mapping(self.admission_service_class_quotas),
        )
        object.__setattr__(
            self,
            "decode_service_class_quotas",
            _freeze_optional_mapping(self.decode_service_class_quotas),
        )
        object.__setattr__(
            self,
            "fairness_guardrail_service_classes",
            _freeze_optional_set(self.fairness_guardrail_service_classes),
        )


@dataclass(frozen=True)
class BackendRuntimePolicy:
    max_prefix_cache_entries: int = 8
    preemption_mode: str = "defer_prefill"
    preemption_min_backlog: int = 1
    preemption_min_decodes: int = 1
    preemption_max_queue_wait_s: float = 0.0
    preemptible_service_classes: AbstractSet[str] | None = None
    preempt_multimodal_prefills: bool = False
    preempt_multimodal_max_queue_wait_s: float = 0.0
    multimodal_prefix_cache_protect_threshold: float = 0.0
    gpu_greedy_sampling: bool = False
    gpu_greedy_max_tokens_only: bool = False
    gpu_greedy_bypass_cpu_policies: bool = False
    gpu_greedy_ignore_eos: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preemptible_service_classes",
            _freeze_optional_set(self.preemptible_service_classes),
        )
