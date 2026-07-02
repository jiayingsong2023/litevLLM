# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

from vllm.engine.step_plan import AdmissionPlan, DecodePlan, PrefillPlan


@dataclass(frozen=True)
class AdmissionResult:
    plan: AdmissionPlan | None
    aged_count: int
    service_classes: dict[str, int]
    lora_gap: dict[str, float]
    effective_lora_limit: int
    effective_multimodal_lora_limit: int
    multimodal_lora_limit_triggered: bool
    lora_relaxed: bool
    lora_tightened: bool


@dataclass(frozen=True)
class BudgetResult:
    prefill_budget: int
    decode_limit: int


@dataclass(frozen=True)
class PrefillPlanResult:
    """Immutable result of prefill-phase planning, including the prefill plan
    and metadata about multimodal/lora limit relaxations and fairness gaps."""

    plan: PrefillPlan | None
    fairness_gap: dict[str, float]
    effective_multimodal_request_limit: int
    effective_lora_adapter_limit: int
    effective_multimodal_lora_request_limit: int
    multimodal_limit_triggered: bool
    multimodal_lora_limit_triggered: bool
    multimodal_limit_relaxed: bool
    multimodal_limit_tightened: bool
    multimodal_lora_limit_relaxed: bool
    multimodal_lora_limit_tightened: bool
    multimodal_lora_relaxed_by_fairness: bool
    multimodal_lora_tightened_by_locality: bool
    multimodal_lora_max_fairness_gap: float
    lora_limit_relaxed: bool
    lora_limit_tightened: bool


@dataclass(frozen=True)
class DecodePlanResult:
    """Immutable result of decode-phase planning, including the decode plan
    and metadata about lora/multimodal limits and fairness relaxations."""

    plan: DecodePlan | None
    fairness_gap: dict[str, float]
    effective_lora_adapter_limit: int
    effective_multimodal_lora_request_limit: int
    multimodal_lora_limit_triggered: bool
    multimodal_lora_limit_relaxed: bool
    multimodal_lora_limit_tightened: bool
    multimodal_lora_relaxed_by_fairness: bool
    multimodal_lora_tightened_by_locality: bool
    multimodal_lora_max_fairness_gap: float
    lora_limit_relaxed: bool
    lora_limit_tightened: bool
