# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time

from vllm.engine.planners.types import AdmissionResult
from vllm.engine.scheduling_constraints import (
    LoRAConstraint,
    MultiModalConstraint,
    MultiModalLoRAConstraint,
    ServiceClassSelector,
)
from vllm.engine.scheduling_helpers import lora_adapter_key
from vllm.engine.step_plan import AdmissionPlan


class AdmissionPlanner:
    """Select which queued requests are admitted into the running set each step."""

    def __init__(
        self,
        *,
        max_admit_per_step: int = 2,
        queue_aging_threshold_s: float = 2.0,
        service_class_weights: dict[str, int] | None = None,
        admission_service_class_quotas: dict[str, int] | None = None,
        max_admit_lora_adapters_per_step: int = 0,
        max_admit_multimodal_per_step: int = 0,
        max_admit_multimodal_lora_per_step: int = 0,
        lora_fairness_relax_threshold: float = 0.0,
        lora_locality_tighten_threshold: float = 0.0,
        lora_limit_relax_delta: int = 1,
        lora_limit_tighten_delta: int = 1,
    ) -> None:
        self.max_admit_per_step = max(1, int(max_admit_per_step))
        self.queue_aging_threshold_s = max(0.0, float(queue_aging_threshold_s))
        self.service_class_weights = {
            key: max(1, int(value))
            for key, value in (
                service_class_weights
                or {
                    "latency": 4,
                    "interactive": 4,
                    "balanced": 2,
                    "throughput": 1,
                    "background": 1,
                }
            ).items()
        }
        self.admission_service_class_quotas = self._normalize_quotas(
            admission_service_class_quotas or {}
        )
        self.max_admit_lora_adapters_per_step = max(
            0, int(max_admit_lora_adapters_per_step)
        )
        self.max_admit_multimodal_per_step = max(0, int(max_admit_multimodal_per_step))
        self.max_admit_multimodal_lora_per_step = max(
            0, int(max_admit_multimodal_lora_per_step)
        )
        self.lora_fairness_relax_threshold = max(
            0.0, float(lora_fairness_relax_threshold)
        )
        self.lora_locality_tighten_threshold = max(
            0.0, float(lora_locality_tighten_threshold)
        )
        self.lora_limit_relax_delta = max(1, int(lora_limit_relax_delta))
        self.lora_limit_tighten_delta = max(1, int(lora_limit_tighten_delta))

        self._service_class_selector = ServiceClassSelector()
        self._lora_constraint = LoRAConstraint()
        self._multimodal_constraint = MultiModalConstraint()
        self._multimodal_lora_constraint = MultiModalLoRAConstraint()

        self._admission_service_cursor = 0
        self._admission_lora_cursor = 0
        self._admission_multimodal_cursor = 0
        self._admission_multimodal_lora_cursor = 0

    @staticmethod
    def _normalize_quotas(quotas: dict[str, int]) -> dict[str, int]:
        return {key: max(0, int(value)) for key, value in quotas.items()}

    @staticmethod
    def _count_service_classes(
        scheduler,
        request_ids: list[str],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rid in request_ids:
            service_class = str(scheduler.get_request(rid).service_class or "latency")
            counts[service_class] = counts.get(service_class, 0) + 1
        return counts

    @staticmethod
    def _count_lora_adapters(
        scheduler,
        request_ids: list[str],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rid in request_ids:
            key = lora_adapter_key(scheduler.get_request(rid))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def plan(
        self,
        scheduler,
        now: float | None = None,
    ) -> AdmissionResult:
        if not scheduler.has_capacity() or scheduler.queued_request_count == 0:
            return AdmissionResult(
                plan=None,
                aged_count=0,
                service_classes={},
                lora_gap={},
                effective_lora_limit=0,
                effective_multimodal_lora_limit=0,
                multimodal_lora_limit_triggered=False,
                lora_relaxed=False,
                lora_tightened=False,
            )
        if now is None:
            now = time.perf_counter()
        queued_ids = scheduler.queued_ids
        queued_requests = {rid: scheduler.get_request(rid) for rid in queued_ids}
        aged_ids = [
            rid
            for rid in queued_ids
            if (now - float(queued_requests[rid].queued_at or now))
            >= self.queue_aging_threshold_s
        ]
        aged_set = set(aged_ids)
        non_aged_ids = [rid for rid in queued_ids if rid not in aged_set]
        admit_limit = min(
            scheduler.queued_request_count,
            scheduler.available_slots,
            self.max_admit_per_step,
        )
        request_ids = self._service_class_selector.select_weighted_requests(
            scheduler=scheduler,
            step_scheduler=self,
            request_ids=aged_ids,
            limit=admit_limit,
            cursor_attr="_admission_service_cursor",
            prefer_short_prompts=False,
            quotas=self.admission_service_class_quotas,
        )
        if len(request_ids) < admit_limit:
            selected_set = set(request_ids)
            request_ids.extend(
                self._service_class_selector.select_weighted_requests(
                    scheduler=scheduler,
                    step_scheduler=self,
                    request_ids=[
                        rid for rid in non_aged_ids if rid not in selected_set
                    ],
                    limit=admit_limit - len(request_ids),
                    cursor_attr="_admission_service_cursor",
                    prefer_short_prompts=True,
                    quotas=self.admission_service_class_quotas,
                )
            )
        (
            request_ids,
            effective_lora_limit,
            fairness_gap,
            relaxed,
            tightened,
        ) = self._lora_constraint.shape_lora_batch(
            scheduler=scheduler,
            step_scheduler=self,
            request_ids=request_ids,
            baseline_counts=self._count_lora_adapters(scheduler, queued_ids),
            max_lora_adapters=self.max_admit_lora_adapters_per_step,
            cursor_attr="_admission_lora_cursor",
        )
        (
            request_ids,
            effective_multimodal_lora_limit,
            multimodal_lora_limit_triggered,
            _admit_multimodal_lora_relaxed,
            _admit_multimodal_lora_tightened,
            _admit_multimodal_lora_relaxed_by_fairness,
            _admit_multimodal_lora_tightened_by_locality,
            _admit_multimodal_lora_max_fairness_gap,
        ) = self._multimodal_lora_constraint.apply_multimodal_lora_request_limit(
            scheduler=scheduler,
            step_scheduler=self,
            request_ids=request_ids,
            max_multimodal_lora_requests=self.max_admit_multimodal_lora_per_step,
            cursor_attr="_admission_multimodal_lora_cursor",
        )
        request_ids = self._multimodal_constraint.apply_multimodal_request_limit(
            scheduler=scheduler,
            step_scheduler=self,
            request_ids=request_ids,
            max_multimodal_requests=self.max_admit_multimodal_per_step,
            cursor_attr="_admission_multimodal_cursor",
        )
        if not request_ids:
            return AdmissionResult(
                plan=None,
                aged_count=0,
                service_classes={},
                lora_gap=fairness_gap,
                effective_lora_limit=effective_lora_limit,
                effective_multimodal_lora_limit=effective_multimodal_lora_limit,
                multimodal_lora_limit_triggered=multimodal_lora_limit_triggered,
                lora_relaxed=relaxed,
                lora_tightened=tightened,
            )
        aged_admission_count = sum(1 for rid in request_ids if rid in aged_set)
        return AdmissionResult(
            plan=AdmissionPlan(request_ids=request_ids),
            aged_count=aged_admission_count,
            service_classes=self._count_service_classes(scheduler, request_ids),
            lora_gap=fairness_gap,
            effective_lora_limit=effective_lora_limit,
            effective_multimodal_lora_limit=effective_multimodal_lora_limit,
            multimodal_lora_limit_triggered=multimodal_lora_limit_triggered,
            lora_relaxed=relaxed,
            lora_tightened=tightened,
        )
