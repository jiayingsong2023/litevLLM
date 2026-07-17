# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.engine.planners.types import AdmissionResult
from vllm.engine.step_plan import AdmissionPlan


class AdmissionPlanner:
    """Admit queued requests in FIFO order."""

    def __init__(self, *, max_admit_per_step: int = 2) -> None:
        self.max_admit_per_step = max(1, int(max_admit_per_step))

    def plan(self, scheduler) -> AdmissionResult:
        if not scheduler.has_capacity() or scheduler.queued_request_count == 0:
            return AdmissionResult(plan=None)
        admit_limit = min(
            scheduler.queued_request_count,
            scheduler.available_slots,
            self.max_admit_per_step,
        )
        return AdmissionResult(
            plan=AdmissionPlan(request_ids=scheduler.queued_ids[:admit_limit])
        )
