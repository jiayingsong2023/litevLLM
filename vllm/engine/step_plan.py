# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass


@dataclass(frozen=True)
class AdmissionPlan:
    request_ids: list[str]


@dataclass(frozen=True)
class PrefillPlan:
    request_ids: list[str]
    chunk_len: int
    token_budget: int


@dataclass(frozen=True)
class DecodePlan:
    request_ids: list[str]
    token_budget: int
    use_fast_path: bool


@dataclass(frozen=True)
class StepPlan:
    admissions: AdmissionPlan | None
    prefills: PrefillPlan | None
    decodes: DecodePlan | None
    step_token_budget: int
    queued_before: int = 0
    running_before: int = 0
    multimodal_prefix_cache_hit_rate: float = 0.0
    prefill_starvation_protected: bool = False
    aged_admission_count: int = 0
    admitted_service_classes: dict[str, int] | None = None
    admitted_multimodal_requests: int = 0
    admitted_multimodal_lora_requests: int = 0
    admitted_lora_adapters: dict[str, int] | None = None
    prefill_service_classes: dict[str, int] | None = None
    prefill_multimodal_requests: int = 0
    prefill_multimodal_lora_requests: int = 0
    prefill_lora_adapters: dict[str, int] | None = None
    decode_service_classes: dict[str, int] | None = None
    decode_multimodal_requests: int = 0
    decode_multimodal_lora_requests: int = 0
    decode_lora_adapters: dict[str, int] | None = None
    effective_admit_multimodal_lora_request_limit: int = 0
    effective_prefill_multimodal_request_limit: int = 0
    effective_prefill_multimodal_lora_request_limit: int = 0
    effective_decode_multimodal_lora_request_limit: int = 0
    effective_admit_lora_adapter_limit: int = 0
    effective_prefill_lora_adapter_limit: int = 0
    effective_decode_lora_adapter_limit: int = 0
    admit_multimodal_lora_limit_triggered: bool = False
    prefill_multimodal_limit_relaxed: bool = False
    prefill_multimodal_limit_tightened: bool = False
    prefill_multimodal_limit_triggered: bool = False
    prefill_multimodal_lora_limit_relaxed: bool = False
    prefill_multimodal_lora_limit_tightened: bool = False
    prefill_multimodal_lora_limit_triggered: bool = False
    prefill_multimodal_lora_limit_relaxed_by_fairness: bool = False
    prefill_multimodal_lora_limit_tightened_by_locality: bool = False
    prefill_multimodal_lora_max_fairness_gap: float = 0.0
    decode_multimodal_lora_limit_relaxed: bool = False
    decode_multimodal_lora_limit_tightened: bool = False
    decode_multimodal_lora_limit_triggered: bool = False
    decode_multimodal_lora_limit_relaxed_by_fairness: bool = False
    decode_multimodal_lora_limit_tightened_by_locality: bool = False
    decode_multimodal_lora_max_fairness_gap: float = 0.0
    admit_lora_limit_relaxed: bool = False
    admit_lora_limit_tightened: bool = False
    prefill_lora_limit_relaxed: bool = False
    prefill_lora_limit_tightened: bool = False
    decode_lora_limit_relaxed: bool = False
    decode_lora_limit_tightened: bool = False
    admitted_lora_fairness_gap: dict[str, float] | None = None
    prefill_lora_fairness_gap: dict[str, float] | None = None
    decode_lora_fairness_gap: dict[str, float] | None = None
    admitted_max_lora_fairness_gap: float = 0.0
    prefill_max_lora_fairness_gap: float = 0.0
    decode_max_lora_fairness_gap: float = 0.0
    queued_service_classes: dict[str, int] | None = None
    queued_multimodal_requests: int = 0
    queued_multimodal_lora_requests: int = 0
    queued_lora_adapters: dict[str, int] | None = None
    queued_avg_wait_s: float = 0.0
    queued_max_wait_s: float = 0.0
    queued_p95_wait_s: float = 0.0
    queued_multimodal_avg_wait_s: float = 0.0
    queued_multimodal_max_wait_s: float = 0.0
    queued_multimodal_p95_wait_s: float = 0.0
    queued_service_class_avg_wait_s: dict[str, float] | None = None
    queued_service_class_max_wait_s: dict[str, float] | None = None
    queued_service_class_p95_wait_s: dict[str, float] | None = None
    fairness_guardrail_triggered: bool = False
