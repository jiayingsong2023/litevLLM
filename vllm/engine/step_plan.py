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
class StepPlanMetrics:
    queued_before: int = 0
    running_before: int = 0
    prefill_starvation_protected: bool = False


@dataclass(frozen=True)
class StepPlan:
    admissions: AdmissionPlan | None
    prefills: PrefillPlan | None
    decodes: DecodePlan | None
    step_token_budget: int
    metrics: StepPlanMetrics | None = None
