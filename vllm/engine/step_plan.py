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
    prefill_starvation_protected: bool = False
    aged_admission_count: int = 0
    admitted_service_classes: dict[str, int] | None = None
    prefill_service_classes: dict[str, int] | None = None
    decode_service_classes: dict[str, int] | None = None
    queued_service_classes: dict[str, int] | None = None
    queued_avg_wait_s: float = 0.0
    queued_max_wait_s: float = 0.0
    queued_p95_wait_s: float = 0.0
    queued_service_class_avg_wait_s: dict[str, float] | None = None
    queued_service_class_max_wait_s: dict[str, float] | None = None
    queued_service_class_p95_wait_s: dict[str, float] | None = None
    fairness_guardrail_triggered: bool = False
