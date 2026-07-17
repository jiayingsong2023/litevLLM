# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

from vllm.engine.step_plan import AdmissionPlan, DecodePlan, PrefillPlan


@dataclass(frozen=True)
class AdmissionResult:
    plan: AdmissionPlan | None


@dataclass(frozen=True)
class BudgetResult:
    prefill_budget: int
    decode_limit: int


@dataclass(frozen=True)
class PrefillPlanResult:
    plan: PrefillPlan | None


@dataclass(frozen=True)
class DecodePlanResult:
    plan: DecodePlan | None
