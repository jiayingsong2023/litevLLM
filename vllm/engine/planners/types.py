# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

from vllm.engine.step_plan import AdmissionPlan


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
