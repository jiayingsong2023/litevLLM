# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from vllm.engine.planners.types import PrefillPlanResult
from vllm.engine.scheduling_constraints import (
    LoRAConstraint,
    MultiModalConstraint,
    MultiModalLoRAConstraint,
    ServiceClassSelector,
)
from vllm.engine.scheduling_helpers import lora_adapter_key, rotate_candidates
from vllm.engine.step_plan import PrefillPlan


class DecodePrefillPlanner:
    """Build PrefillPlan and DecodePlan for a single engine step."""

    def __init__(
        self,
        *,
        service_class_weights: dict[str, int],
        decode_service_class_quotas: dict[str, int],
        max_prefill_lora_adapters_per_batch: int,
        max_decode_lora_adapters_per_batch: int,
        max_prefill_multimodal_requests_per_batch: int,
        max_decode_multimodal_requests_per_batch: int,
        max_prefill_multimodal_lora_requests_per_batch: int,
        max_decode_multimodal_lora_requests_per_batch: int,
        lora_fairness_relax_threshold: float,
        lora_locality_tighten_threshold: float,
        lora_limit_relax_delta: int,
        lora_limit_tighten_delta: int,
        multimodal_prefix_cache_relax_threshold: float,
        multimodal_prefix_cache_tighten_threshold: float,
        multimodal_prefill_limit_relax_delta: int,
        multimodal_prefill_limit_tighten_delta: int,
        multimodal_lora_prefill_limit_relax_delta: int,
        multimodal_lora_prefill_limit_tighten_delta: int,
        multimodal_lora_fairness_relax_threshold: float,
        multimodal_lora_locality_tighten_threshold: float,
        prefill_chunk_size: int,
        prefill_microbatch_size: int,
    ) -> None:
        """Initialize the planner with scheduling limits and cursor state.

        Stores the configuration parameters used to shape prefill and decode
        batches, initializes round-robin cursors, and creates the constraint
        helpers for service classes, LoRA adapters, and multimodal requests.
        """
        self.service_class_weights = service_class_weights
        self.decode_service_class_quotas = decode_service_class_quotas
        self.max_prefill_lora_adapters_per_batch = max_prefill_lora_adapters_per_batch
        self.max_decode_lora_adapters_per_batch = max_decode_lora_adapters_per_batch
        self.max_prefill_multimodal_requests_per_batch = (
            max_prefill_multimodal_requests_per_batch
        )
        self.max_decode_multimodal_requests_per_batch = (
            max_decode_multimodal_requests_per_batch
        )
        self.max_prefill_multimodal_lora_requests_per_batch = (
            max_prefill_multimodal_lora_requests_per_batch
        )
        self.max_decode_multimodal_lora_requests_per_batch = (
            max_decode_multimodal_lora_requests_per_batch
        )
        self.lora_fairness_relax_threshold = lora_fairness_relax_threshold
        self.lora_locality_tighten_threshold = lora_locality_tighten_threshold
        self.lora_limit_relax_delta = lora_limit_relax_delta
        self.lora_limit_tighten_delta = lora_limit_tighten_delta
        self.multimodal_prefix_cache_relax_threshold = (
            multimodal_prefix_cache_relax_threshold
        )
        self.multimodal_prefix_cache_tighten_threshold = (
            multimodal_prefix_cache_tighten_threshold
        )
        self.multimodal_prefill_limit_relax_delta = multimodal_prefill_limit_relax_delta
        self.multimodal_prefill_limit_tighten_delta = (
            multimodal_prefill_limit_tighten_delta
        )
        self.multimodal_lora_prefill_limit_relax_delta = (
            multimodal_lora_prefill_limit_relax_delta
        )
        self.multimodal_lora_prefill_limit_tighten_delta = (
            multimodal_lora_prefill_limit_tighten_delta
        )
        self.multimodal_lora_fairness_relax_threshold = (
            multimodal_lora_fairness_relax_threshold
        )
        self.multimodal_lora_locality_tighten_threshold = (
            multimodal_lora_locality_tighten_threshold
        )
        self.prefill_chunk_size = prefill_chunk_size
        self.prefill_microbatch_size = prefill_microbatch_size

        self._multimodal_prefix_cache_hit_rate_feedback = 0.0
        self._prefill_rr_cursor = 0
        self._prefill_lora_cursor = 0
        self._prefill_multimodal_cursor = 0
        self._prefill_multimodal_lora_cursor = 0
        self._decode_rr_cursor = 0
        self._decode_service_cursor = 0
        self._decode_lora_cursor = 0
        self._decode_multimodal_cursor = 0
        self._decode_multimodal_lora_cursor = 0

        self._service_class_selector = ServiceClassSelector()
        self._lora_constraint = LoRAConstraint()
        self._multimodal_constraint = MultiModalConstraint()
        self._multimodal_lora_constraint = MultiModalLoRAConstraint()

    def update_runtime_feedback(self, feedback: dict[str, Any]) -> None:
        """Consume runtime feedback to adjust future planning decisions.

        Extracts the multimodal prefix cache hit rate from the observer
        feedback so that multimodal prefill limits can be relaxed or tightened
        in subsequent steps.
        """
        self._multimodal_prefix_cache_hit_rate_feedback = float(
            feedback.get("observer", {})
            .get("multimodal", {})
            .get("prefix_cache_hit_rate", 0.0)
        )

    @staticmethod
    def _count_lora_adapters(scheduler, request_ids: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rid in request_ids:
            key = lora_adapter_key(scheduler.get_request(rid))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _shape_lora_batch(
        self,
        *,
        scheduler,
        request_ids: list[str],
        baseline_counts: dict[str, int],
        max_lora_adapters: int,
        cursor_attr: str,
    ):
        return self._lora_constraint.shape_lora_batch(
            scheduler=scheduler,
            step_scheduler=self,
            request_ids=request_ids,
            baseline_counts=baseline_counts,
            max_lora_adapters=max_lora_adapters,
            cursor_attr=cursor_attr,
        )

    def _apply_multimodal_lora_request_limit(
        self,
        *,
        scheduler,
        request_ids: list[str],
        max_multimodal_lora_requests: int,
        cursor_attr: str,
        candidate_request_ids: list[str] | None = None,
    ):
        return self._multimodal_lora_constraint.apply_multimodal_lora_request_limit(
            scheduler=scheduler,
            step_scheduler=self,
            request_ids=request_ids,
            max_multimodal_lora_requests=max_multimodal_lora_requests,
            cursor_attr=cursor_attr,
            candidate_request_ids=candidate_request_ids,
        )

    def _shape_multimodal_prefill_batch(
        self,
        *,
        scheduler,
        request_ids: list[str],
        candidate_prefills: list[str],
    ):
        return self._multimodal_constraint.shape_multimodal_prefill_batch(
            scheduler=scheduler,
            step_scheduler=self,
            request_ids=request_ids,
            candidate_prefills=candidate_prefills,
        )

    @staticmethod
    def _rotate_candidates(request_ids: list[str], cursor: int) -> list[str]:
        return rotate_candidates(request_ids, cursor)

    def build_prefill_plan(
        self,
        scheduler,
        prefills: list[str],
        token_budget: int,
    ) -> PrefillPlanResult:
        """Assemble a prefill plan for the next engine step.

        Selects prefill candidates that share the same processed sequence length,
        rotates them by the round-robin cursor, applies multimodal and LoRA
        constraints, and computes a per-request chunk length bounded by the
        token budget and ``prefill_chunk_size``.

        Args:
            scheduler: The request scheduler holding request states.
            prefills: Ordered list of request IDs waiting for prefill.
            token_budget: Total tokens available for prefill this step.

        Returns:
            A ``PrefillPlanResult`` containing the plan (or ``None`` when no
            requests can be scheduled) and metadata about effective limits and
            fairness/relaxation state.
        """
        if not prefills or token_budget <= 0:
            return PrefillPlanResult(
                plan=None,
                fairness_gap={},
                effective_multimodal_request_limit=0,
                effective_lora_adapter_limit=0,
                effective_multimodal_lora_request_limit=0,
                multimodal_limit_triggered=False,
                multimodal_lora_limit_triggered=False,
                multimodal_limit_relaxed=False,
                multimodal_limit_tightened=False,
                multimodal_lora_limit_relaxed=False,
                multimodal_lora_limit_tightened=False,
                multimodal_lora_relaxed_by_fairness=False,
                multimodal_lora_tightened_by_locality=False,
                multimodal_lora_max_fairness_gap=0.0,
                lora_limit_relaxed=False,
                lora_limit_tightened=False,
            )
        base_processed_len = scheduler.get_request(prefills[0]).seq_len
        candidate_prefills = [
            rid
            for rid in prefills
            if scheduler.get_request(rid).seq_len == base_processed_len
        ]
        if not candidate_prefills:
            return PrefillPlanResult(
                plan=None,
                fairness_gap={},
                effective_multimodal_request_limit=0,
                effective_lora_adapter_limit=0,
                effective_multimodal_lora_request_limit=0,
                multimodal_limit_triggered=False,
                multimodal_lora_limit_triggered=False,
                multimodal_limit_relaxed=False,
                multimodal_limit_tightened=False,
                multimodal_lora_limit_relaxed=False,
                multimodal_lora_limit_tightened=False,
                multimodal_lora_relaxed_by_fairness=False,
                multimodal_lora_tightened_by_locality=False,
                multimodal_lora_max_fairness_gap=0.0,
                lora_limit_relaxed=False,
                lora_limit_tightened=False,
            )
        request_ids = self._rotate_candidates(
            candidate_prefills,
            self._prefill_rr_cursor,
        )[: min(self.prefill_microbatch_size, len(candidate_prefills))]
        self._prefill_rr_cursor = (self._prefill_rr_cursor + len(request_ids)) % max(
            1, len(candidate_prefills)
        )
        (
            request_ids,
            effective_multimodal_limit,
            multimodal_limit_triggered,
            multimodal_relaxed,
            multimodal_tightened,
        ) = self._shape_multimodal_prefill_batch(
            scheduler=scheduler,
            request_ids=request_ids,
            candidate_prefills=candidate_prefills,
        )
        (
            request_ids,
            effective_lora_limit,
            fairness_gap,
            relaxed,
            tightened,
        ) = self._shape_lora_batch(
            scheduler=scheduler,
            request_ids=request_ids,
            baseline_counts=self._count_lora_adapters(scheduler, candidate_prefills),
            max_lora_adapters=self.max_prefill_lora_adapters_per_batch,
            cursor_attr="_prefill_lora_cursor",
        )
        (
            request_ids,
            effective_multimodal_lora_limit,
            multimodal_lora_limit_triggered,
            multimodal_lora_relaxed,
            multimodal_lora_tightened,
            multimodal_lora_relaxed_by_fairness,
            multimodal_lora_tightened_by_locality,
            multimodal_lora_max_fairness_gap,
        ) = self._apply_multimodal_lora_request_limit(
            scheduler=scheduler,
            request_ids=request_ids,
            max_multimodal_lora_requests=(
                self.max_prefill_multimodal_lora_requests_per_batch
            ),
            cursor_attr="_prefill_multimodal_lora_cursor",
            candidate_request_ids=candidate_prefills,
        )
        if not request_ids:
            return PrefillPlanResult(
                plan=None,
                fairness_gap=fairness_gap,
                effective_multimodal_request_limit=effective_multimodal_limit,
                effective_lora_adapter_limit=effective_lora_limit,
                effective_multimodal_lora_request_limit=effective_multimodal_lora_limit,
                multimodal_limit_triggered=multimodal_limit_triggered,
                multimodal_lora_limit_triggered=multimodal_lora_limit_triggered,
                multimodal_limit_relaxed=multimodal_relaxed,
                multimodal_limit_tightened=multimodal_tightened,
                multimodal_lora_limit_relaxed=multimodal_lora_relaxed,
                multimodal_lora_limit_tightened=multimodal_lora_tightened,
                multimodal_lora_relaxed_by_fairness=multimodal_lora_relaxed_by_fairness,
                multimodal_lora_tightened_by_locality=multimodal_lora_tightened_by_locality,
                multimodal_lora_max_fairness_gap=multimodal_lora_max_fairness_gap,
                lora_limit_relaxed=relaxed,
                lora_limit_tightened=tightened,
            )
        min_remaining = min(
            len(scheduler.get_request(rid).input_ids)
            - scheduler.get_request(rid).seq_len
            for rid in request_ids
        )
        per_req_budget = max(1, token_budget // max(1, len(request_ids)))
        chunk_len = min(min_remaining, self.prefill_chunk_size, per_req_budget)
        if chunk_len <= 0:
            chunk_len = 1
        return PrefillPlanResult(
            plan=PrefillPlan(
                request_ids=request_ids,
                chunk_len=chunk_len,
                token_budget=token_budget,
            ),
            fairness_gap=fairness_gap,
            effective_multimodal_request_limit=effective_multimodal_limit,
            effective_lora_adapter_limit=effective_lora_limit,
            effective_multimodal_lora_request_limit=effective_multimodal_lora_limit,
            multimodal_limit_triggered=multimodal_limit_triggered,
            multimodal_lora_limit_triggered=multimodal_lora_limit_triggered,
            multimodal_limit_relaxed=multimodal_relaxed,
            multimodal_limit_tightened=multimodal_tightened,
            multimodal_lora_limit_relaxed=multimodal_lora_relaxed,
            multimodal_lora_limit_tightened=multimodal_lora_tightened,
            multimodal_lora_relaxed_by_fairness=multimodal_lora_relaxed_by_fairness,
            multimodal_lora_tightened_by_locality=multimodal_lora_tightened_by_locality,
            multimodal_lora_max_fairness_gap=multimodal_lora_max_fairness_gap,
            lora_limit_relaxed=relaxed,
            lora_limit_tightened=tightened,
        )
