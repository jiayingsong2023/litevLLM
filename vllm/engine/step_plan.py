# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass


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
    prefills: PrefillPlan | None
    decodes: DecodePlan | None
    step_token_budget: int
