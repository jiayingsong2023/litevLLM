# SPDX-License-Identifier: Apache-2.0
import time
from dataclasses import dataclass

from vllm.engine.planners import AdmissionPlanner, BudgetComputer, DecodePrefillPlanner
from vllm.engine.scheduling_constraints import (
    LoRAConstraint,
    MultiModalConstraint,
    MultiModalLoRAConstraint,
    ServiceClassSelector,
)
from vllm.engine.scheduling_helpers import (
    is_multimodal,
    is_multimodal_lora,
    lora_adapter_key,
    max_abs_share_gap,
    normalize_quotas,
    normalized_share_map,
    percentile,
    rotate_candidates,
    service_class_priority,
    share_gap_map,
)
from vllm.engine.step_plan import (
    DecodePlan,
    PrefillPlan,
    StepPlan,
    StepPlanMetrics,
)


@dataclass(frozen=True)
class LoraSchedulingParams:
    """LoRA scheduling constraints as a frozen value object."""

    max_admit_lora_adapters_per_step: int = 0
    max_prefill_lora_adapters_per_batch: int = 0
    max_decode_lora_adapters_per_batch: int = 0
    lora_fairness_relax_threshold: float = 0.0
    lora_locality_tighten_threshold: float = 0.0
    lora_limit_relax_delta: int = 1
    lora_limit_tighten_delta: int = 1


@dataclass(frozen=True)
class MultiModalSchedulingParams:
    """MultiModal scheduling constraints as a frozen value object."""

    max_admit_multimodal_per_step: int = 0
    max_prefill_multimodal_requests_per_batch: int = 0
    max_decode_multimodal_requests_per_batch: int = 0
    max_admit_multimodal_lora_per_step: int = 0
    max_prefill_multimodal_lora_requests_per_batch: int = 0
    max_decode_multimodal_lora_requests_per_batch: int = 0
    multimodal_prefix_cache_relax_threshold: float = 0.0
    multimodal_prefix_cache_tighten_threshold: float = 0.0
    multimodal_prefill_limit_relax_delta: int = 1
    multimodal_prefill_limit_tighten_delta: int = 1
    multimodal_lora_prefill_limit_relax_delta: int = 1
    multimodal_lora_prefill_limit_tighten_delta: int = 1
    multimodal_lora_fairness_relax_threshold: float = 0.0
    multimodal_lora_locality_tighten_threshold: float = 0.0


class StepScheduler:
    DEFAULT_SERVICE_CLASS_WEIGHTS = {
        "latency": 4,
        "interactive": 4,
        "balanced": 2,
        "throughput": 1,
        "background": 1,
    }
    DEFAULT_ADMISSION_SERVICE_CLASS_QUOTAS: dict[str, int] = {}
    DEFAULT_DECODE_SERVICE_CLASS_QUOTAS: dict[str, int] = {}
    BASE_LORA_ADAPTER = "__base__"

    def __init__(
        self,
        *,
        step_token_budget: int,
        decode_priority_enabled: bool,
        prefill_chunk_size: int,
        prefill_reserved_tokens: int,
        prefill_reserve_backlog: int,
        prefill_catchup_ratio: float,
        prefill_microbatch_size: int,
        max_admit_per_step: int = 2,
        max_decode_streak: int = 4,
        queue_aging_threshold_s: float = 2.0,
        max_prefill_deferrals: int = 2,
        service_class_weights: dict[str, int] | None = None,
        admission_service_class_quotas: dict[str, int] | None = None,
        decode_service_class_quotas: dict[str, int] | None = None,
        fairness_guardrail_queue_wait_s: float = 0.0,
        fairness_guardrail_service_classes: set[str] | None = None,
        lora_params: LoraSchedulingParams | None = None,
        multimodal_params: MultiModalSchedulingParams | None = None,
        min_prefill_chunk_size: int | None = None,
        max_prefill_chunk_size: int | None = None,
        prefill_sla_ttft_ms: float | None = None,
    ) -> None:
        self.step_token_budget = max(1, int(step_token_budget))
        self.decode_priority_enabled = decode_priority_enabled
        self.prefill_chunk_size = max(1, int(prefill_chunk_size))

        self.max_prefill_chunk_size = int(
            max_prefill_chunk_size
            if max_prefill_chunk_size is not None
            else self.prefill_chunk_size
        )
        self.min_prefill_chunk_size = int(
            min_prefill_chunk_size
            if min_prefill_chunk_size is not None
            else min(self.max_prefill_chunk_size, 128)
        )
        self.prefill_sla_ttft_ms = float(
            prefill_sla_ttft_ms if prefill_sla_ttft_ms is not None else 2000.0
        )

        self.prefill_reserved_tokens = max(0, int(prefill_reserved_tokens))
        self.prefill_reserve_backlog = max(1, int(prefill_reserve_backlog))
        self.prefill_catchup_ratio = min(1.0, max(0.0, float(prefill_catchup_ratio)))
        self.prefill_microbatch_size = min(4, max(1, int(prefill_microbatch_size)))
        self.max_admit_per_step = max(1, int(max_admit_per_step))
        self.max_decode_streak = max(1, int(max_decode_streak))
        self.queue_aging_threshold_s = max(0.0, float(queue_aging_threshold_s))
        self.max_prefill_deferrals = max(1, int(max_prefill_deferrals))
        self.service_class_weights = {
            key: max(1, int(value))
            for key, value in (
                service_class_weights or self.DEFAULT_SERVICE_CLASS_WEIGHTS
            ).items()
        }
        self.admission_service_class_quotas = self._normalize_quotas(
            admission_service_class_quotas
            or self.DEFAULT_ADMISSION_SERVICE_CLASS_QUOTAS
        )
        self.decode_service_class_quotas = self._normalize_quotas(
            decode_service_class_quotas or self.DEFAULT_DECODE_SERVICE_CLASS_QUOTAS
        )
        self.fairness_guardrail_queue_wait_s = max(
            0.0, float(fairness_guardrail_queue_wait_s)
        )
        self.fairness_guardrail_service_classes = {
            str(item).strip()
            for item in (fairness_guardrail_service_classes or set())
            if str(item).strip()
        }
        _lora = lora_params or LoraSchedulingParams()
        _mm = multimodal_params or MultiModalSchedulingParams()
        self.max_admit_lora_adapters_per_step = max(
            0, int(_lora.max_admit_lora_adapters_per_step)
        )
        self.max_prefill_lora_adapters_per_batch = max(
            0, int(_lora.max_prefill_lora_adapters_per_batch)
        )
        self.max_decode_lora_adapters_per_batch = max(
            0, int(_lora.max_decode_lora_adapters_per_batch)
        )
        self.lora_fairness_relax_threshold = max(
            0.0, float(_lora.lora_fairness_relax_threshold)
        )
        self.lora_locality_tighten_threshold = max(
            0.0, float(_lora.lora_locality_tighten_threshold)
        )
        self.lora_limit_relax_delta = max(1, int(_lora.lora_limit_relax_delta))
        self.lora_limit_tighten_delta = max(1, int(_lora.lora_limit_tighten_delta))
        self.max_admit_multimodal_per_step = max(
            0, int(_mm.max_admit_multimodal_per_step)
        )
        self.max_prefill_multimodal_requests_per_batch = max(
            0, int(_mm.max_prefill_multimodal_requests_per_batch)
        )
        self.max_decode_multimodal_requests_per_batch = max(
            0, int(_mm.max_decode_multimodal_requests_per_batch)
        )
        self.max_admit_multimodal_lora_per_step = max(
            0, int(_mm.max_admit_multimodal_lora_per_step)
        )
        self.max_prefill_multimodal_lora_requests_per_batch = max(
            0, int(_mm.max_prefill_multimodal_lora_requests_per_batch)
        )
        self.max_decode_multimodal_lora_requests_per_batch = max(
            0, int(_mm.max_decode_multimodal_lora_requests_per_batch)
        )
        self.multimodal_prefix_cache_relax_threshold = max(
            0.0, float(_mm.multimodal_prefix_cache_relax_threshold)
        )
        self.multimodal_prefix_cache_tighten_threshold = max(
            0.0, float(_mm.multimodal_prefix_cache_tighten_threshold)
        )
        self.multimodal_prefill_limit_relax_delta = max(
            1, int(_mm.multimodal_prefill_limit_relax_delta)
        )
        self.multimodal_prefill_limit_tighten_delta = max(
            1, int(_mm.multimodal_prefill_limit_tighten_delta)
        )
        self.multimodal_lora_prefill_limit_relax_delta = max(
            1, int(_mm.multimodal_lora_prefill_limit_relax_delta)
        )
        self.multimodal_lora_prefill_limit_tighten_delta = max(
            1, int(_mm.multimodal_lora_prefill_limit_tighten_delta)
        )
        self.multimodal_lora_fairness_relax_threshold = max(
            0.0, float(_mm.multimodal_lora_fairness_relax_threshold)
        )
        self.multimodal_lora_locality_tighten_threshold = max(
            0.0, float(_mm.multimodal_lora_locality_tighten_threshold)
        )
        self._decode_only_streak = 0
        self._prefill_deferrals: dict[str, int] = {}
        self._multimodal_prefix_cache_hit_rate_feedback = 0.0

        self._admission_planner = AdmissionPlanner(
            max_admit_per_step=max_admit_per_step,
            queue_aging_threshold_s=queue_aging_threshold_s,
            service_class_weights=self.service_class_weights,
            admission_service_class_quotas=self.admission_service_class_quotas,
            max_admit_lora_adapters_per_step=self.max_admit_lora_adapters_per_step,
            max_admit_multimodal_per_step=self.max_admit_multimodal_per_step,
            max_admit_multimodal_lora_per_step=(
                self.max_admit_multimodal_lora_per_step
            ),
            lora_fairness_relax_threshold=self.lora_fairness_relax_threshold,
            lora_locality_tighten_threshold=self.lora_locality_tighten_threshold,
            lora_limit_relax_delta=self.lora_limit_relax_delta,
            lora_limit_tighten_delta=self.lora_limit_tighten_delta,
        )
        self._budget_computer = BudgetComputer(
            step_token_budget=step_token_budget,
            prefill_chunk_size=self.prefill_chunk_size,
            decode_priority_enabled=decode_priority_enabled,
            prefill_reserved_tokens=prefill_reserved_tokens,
            prefill_reserve_backlog=prefill_reserve_backlog,
            prefill_catchup_ratio=prefill_catchup_ratio,
        )
        self._decode_prefill_planner = DecodePrefillPlanner(
            service_class_weights=self.service_class_weights,
            decode_service_class_quotas=self.decode_service_class_quotas,
            max_prefill_lora_adapters_per_batch=self.max_prefill_lora_adapters_per_batch,
            max_decode_lora_adapters_per_batch=self.max_decode_lora_adapters_per_batch,
            max_prefill_multimodal_requests_per_batch=self.max_prefill_multimodal_requests_per_batch,
            max_decode_multimodal_requests_per_batch=self.max_decode_multimodal_requests_per_batch,
            max_prefill_multimodal_lora_requests_per_batch=self.max_prefill_multimodal_lora_requests_per_batch,
            max_decode_multimodal_lora_requests_per_batch=self.max_decode_multimodal_lora_requests_per_batch,
            lora_fairness_relax_threshold=self.lora_fairness_relax_threshold,
            lora_locality_tighten_threshold=self.lora_locality_tighten_threshold,
            lora_limit_relax_delta=self.lora_limit_relax_delta,
            lora_limit_tighten_delta=self.lora_limit_tighten_delta,
            multimodal_prefix_cache_relax_threshold=self.multimodal_prefix_cache_relax_threshold,
            multimodal_prefix_cache_tighten_threshold=self.multimodal_prefix_cache_tighten_threshold,
            multimodal_prefill_limit_relax_delta=self.multimodal_prefill_limit_relax_delta,
            multimodal_prefill_limit_tighten_delta=self.multimodal_prefill_limit_tighten_delta,
            multimodal_lora_prefill_limit_relax_delta=self.multimodal_lora_prefill_limit_relax_delta,
            multimodal_lora_prefill_limit_tighten_delta=self.multimodal_lora_prefill_limit_tighten_delta,
            multimodal_lora_fairness_relax_threshold=self.multimodal_lora_fairness_relax_threshold,
            multimodal_lora_locality_tighten_threshold=self.multimodal_lora_locality_tighten_threshold,
            prefill_chunk_size=self.prefill_chunk_size,
            prefill_microbatch_size=self.prefill_microbatch_size,
        )

        self._service_class_selector = ServiceClassSelector()
        self._lora_constraint = LoRAConstraint()
        self._multimodal_constraint = MultiModalConstraint()
        self._multimodal_lora_constraint = MultiModalLoRAConstraint()

    def update_runtime_feedback(self, stats: dict[str, object] | None) -> None:
        observer = stats.get("observer") if isinstance(stats, dict) else None
        multimodal = observer.get("multimodal") if isinstance(observer, dict) else None
        if isinstance(multimodal, dict):
            self._multimodal_prefix_cache_hit_rate_feedback = float(
                multimodal.get("prefix_cache_hit_rate", 0.0) or 0.0
            )
        else:
            self._multimodal_prefix_cache_hit_rate_feedback = 0.0
        self._decode_prefill_planner.update_runtime_feedback(stats)

    def build_plan(self, scheduler) -> StepPlan:
        fast_decode_plan = self._build_single_request_decode_fast_path(scheduler)
        if fast_decode_plan is not None:
            decode_ids = (
                fast_decode_plan.decodes.request_ids if fast_decode_plan.decodes else []
            )
            self._update_prefill_deferrals(
                [],
                decode_ids,
                None,
            )
            self._update_decode_streak(fast_decode_plan)
            return fast_decode_plan

        # Determine the maximum sequence length among all active requests
        max_active_seq_len = 0
        for rid in list(scheduler.running_ids) + list(scheduler.queued_ids):
            try:
                req = scheduler.get_request(rid)
                active_len = len(req.input_ids) if req.is_prefill else req.seq_len
                if active_len > max_active_seq_len:
                    max_active_seq_len = active_len
            except KeyError:
                continue

        # Self-Adaptive Chunk Sizing
        if max_active_seq_len > 16384:
            chunk_size = 256
        elif max_active_seq_len > 8192:
            chunk_size = 512
        elif max_active_seq_len > 4096:
            chunk_size = 1024
        else:
            chunk_size = self.max_prefill_chunk_size

        self.prefill_chunk_size = max(
            self.min_prefill_chunk_size, min(chunk_size, self.max_prefill_chunk_size)
        )

        fast_plan = self._build_single_request_fast_path(scheduler)
        if fast_plan is not None:
            self._update_prefill_deferrals(
                [rid for rid in fast_plan.prefills.request_ids]
                if fast_plan.prefills
                else [],
                [rid for rid in fast_plan.decodes.request_ids]
                if fast_plan.decodes
                else [],
                fast_plan.prefills,
            )
            self._update_decode_streak(fast_plan)
            return fast_plan

        now = time.perf_counter()
        queued_metrics = self._compute_queued_metrics(scheduler)
        admission = self._admission_planner.plan(scheduler, now)
        admissions = admission.plan
        aged_admission_count = admission.aged_count
        admitted_service_classes = admission.service_classes
        admitted_lora_gap = admission.lora_gap
        effective_admit_lora_limit = admission.effective_lora_limit
        effective_admit_multimodal_lora_limit = (
            admission.effective_multimodal_lora_limit
        )
        admit_multimodal_lora_limit_triggered = (
            admission.multimodal_lora_limit_triggered
        )
        admit_lora_relaxed = admission.lora_relaxed
        admit_lora_tightened = admission.lora_tightened
        prefills, decodes = scheduler.classify_requests()
        fairness_guardrail_triggered = self._is_fairness_guardrail_triggered(
            queued_metrics
        )
        starvation_protected = self._should_protect_prefills(
            prefills,
            decodes,
            fairness_guardrail_triggered=fairness_guardrail_triggered,
        )
        budget = self._budget_computer.compute(
            num_prefills=len(prefills),
            num_decodes=len(decodes),
            starvation_protected=starvation_protected,
        )
        prefill_budget = budget.prefill_budget
        decode_limit = budget.decode_limit

        prefill_result = self._decode_prefill_planner.build_prefill_plan(
            scheduler, prefills, prefill_budget
        )
        prefill_plan = prefill_result.plan
        prefill_lora_gap = prefill_result.fairness_gap
        effective_prefill_multimodal_limit = (
            prefill_result.effective_multimodal_request_limit
        )
        effective_prefill_lora_limit = prefill_result.effective_lora_adapter_limit
        effective_prefill_multimodal_lora_limit = (
            prefill_result.effective_multimodal_lora_request_limit
        )
        prefill_multimodal_limit_triggered = prefill_result.multimodal_limit_triggered
        prefill_multimodal_lora_limit_triggered = (
            prefill_result.multimodal_lora_limit_triggered
        )
        prefill_multimodal_relaxed = prefill_result.multimodal_limit_relaxed
        prefill_multimodal_tightened = prefill_result.multimodal_limit_tightened
        prefill_multimodal_lora_relaxed = prefill_result.multimodal_lora_limit_relaxed
        prefill_multimodal_lora_tightened = (
            prefill_result.multimodal_lora_limit_tightened
        )
        prefill_multimodal_lora_relaxed_by_fairness = (
            prefill_result.multimodal_lora_relaxed_by_fairness
        )
        prefill_multimodal_lora_tightened_by_locality = (
            prefill_result.multimodal_lora_tightened_by_locality
        )
        prefill_multimodal_lora_max_fairness_gap = (
            prefill_result.multimodal_lora_max_fairness_gap
        )
        prefill_lora_relaxed = prefill_result.lora_limit_relaxed
        prefill_lora_tightened = prefill_result.lora_limit_tightened
        prefill_request_ids = prefill_plan.request_ids if prefill_plan else []
        prefill_service_classes = self._count_service_classes(
            scheduler, prefill_request_ids
        )
        prefill_lora_adapters = self._count_lora_adapters(
            scheduler, prefill_request_ids
        )
        decode_result = self._decode_prefill_planner.build_decode_plan(
            scheduler,
            decodes,
            decode_limit,
            prefill_plan is None,
        )
        decode_plan = decode_result.plan
        decode_lora_gap = decode_result.fairness_gap
        effective_decode_lora_limit = decode_result.effective_lora_adapter_limit
        effective_decode_multimodal_lora_limit = (
            decode_result.effective_multimodal_lora_request_limit
        )
        decode_multimodal_lora_limit_triggered = (
            decode_result.multimodal_lora_limit_triggered
        )
        decode_multimodal_lora_relaxed = decode_result.multimodal_lora_limit_relaxed
        decode_multimodal_lora_tightened = decode_result.multimodal_lora_limit_tightened
        decode_multimodal_lora_relaxed_by_fairness = (
            decode_result.multimodal_lora_relaxed_by_fairness
        )
        decode_multimodal_lora_tightened_by_locality = (
            decode_result.multimodal_lora_tightened_by_locality
        )
        decode_multimodal_lora_max_fairness_gap = (
            decode_result.multimodal_lora_max_fairness_gap
        )
        decode_lora_relaxed = decode_result.lora_limit_relaxed
        decode_lora_tightened = decode_result.lora_limit_tightened
        decode_request_ids = decode_plan.request_ids if decode_plan else []
        decode_service_classes = self._count_service_classes(
            scheduler, decode_request_ids
        )
        decode_lora_adapters = self._count_lora_adapters(scheduler, decode_request_ids)
        admitted_request_ids = admissions.request_ids if admissions else []
        plan = StepPlan(
            admissions=admissions,
            prefills=prefill_plan,
            decodes=decode_plan,
            step_token_budget=self.step_token_budget,
            metrics=StepPlanMetrics(
                queued_before=scheduler.queued_request_count,
                running_before=scheduler.running_request_count,
                multimodal_prefix_cache_hit_rate=(
                    self._multimodal_prefix_cache_hit_rate_feedback
                ),
                prefill_starvation_protected=bool(
                    starvation_protected and prefill_plan is not None
                ),
                aged_admission_count=aged_admission_count,
                admitted_service_classes=admitted_service_classes,
                admitted_multimodal_requests=self._count_multimodal_requests(
                    scheduler, admitted_request_ids
                ),
                admitted_multimodal_lora_requests=self._count_multimodal_lora_requests(
                    scheduler, admitted_request_ids
                ),
                effective_admit_multimodal_lora_request_limit=(
                    effective_admit_multimodal_lora_limit
                ),
                admit_multimodal_lora_limit_triggered=(
                    admit_multimodal_lora_limit_triggered
                ),
                admitted_lora_adapters=self._count_lora_adapters(
                    scheduler, admitted_request_ids
                ),
                effective_admit_lora_adapter_limit=effective_admit_lora_limit,
                admit_lora_limit_relaxed=admit_lora_relaxed,
                admit_lora_limit_tightened=admit_lora_tightened,
                admitted_lora_fairness_gap=admitted_lora_gap,
                admitted_max_lora_fairness_gap=self._max_abs_share_gap(
                    admitted_lora_gap
                ),
                prefill_service_classes=prefill_service_classes,
                prefill_multimodal_requests=self._count_multimodal_requests(
                    scheduler, prefill_request_ids
                ),
                prefill_multimodal_lora_requests=self._count_multimodal_lora_requests(
                    scheduler, prefill_request_ids
                ),
                effective_prefill_multimodal_request_limit=(
                    effective_prefill_multimodal_limit
                ),
                prefill_multimodal_limit_relaxed=prefill_multimodal_relaxed,
                prefill_multimodal_limit_tightened=prefill_multimodal_tightened,
                prefill_multimodal_limit_triggered=prefill_multimodal_limit_triggered,
                effective_prefill_multimodal_lora_request_limit=(
                    effective_prefill_multimodal_lora_limit
                ),
                prefill_multimodal_lora_limit_relaxed=(prefill_multimodal_lora_relaxed),
                prefill_multimodal_lora_limit_tightened=(
                    prefill_multimodal_lora_tightened
                ),
                prefill_multimodal_lora_limit_relaxed_by_fairness=(
                    prefill_multimodal_lora_relaxed_by_fairness
                ),
                prefill_multimodal_lora_limit_tightened_by_locality=(
                    prefill_multimodal_lora_tightened_by_locality
                ),
                prefill_multimodal_lora_max_fairness_gap=(
                    prefill_multimodal_lora_max_fairness_gap
                ),
                prefill_multimodal_lora_limit_triggered=(
                    prefill_multimodal_lora_limit_triggered
                ),
                prefill_lora_adapters=prefill_lora_adapters,
                effective_prefill_lora_adapter_limit=effective_prefill_lora_limit,
                prefill_lora_limit_relaxed=prefill_lora_relaxed,
                prefill_lora_limit_tightened=prefill_lora_tightened,
                prefill_lora_fairness_gap=prefill_lora_gap,
                prefill_max_lora_fairness_gap=self._max_abs_share_gap(prefill_lora_gap),
                decode_service_classes=decode_service_classes,
                decode_multimodal_requests=self._count_multimodal_requests(
                    scheduler, decode_request_ids
                ),
                decode_multimodal_lora_requests=self._count_multimodal_lora_requests(
                    scheduler, decode_request_ids
                ),
                effective_decode_multimodal_lora_request_limit=(
                    effective_decode_multimodal_lora_limit
                ),
                decode_multimodal_lora_limit_relaxed=decode_multimodal_lora_relaxed,
                decode_multimodal_lora_limit_tightened=decode_multimodal_lora_tightened,
                decode_multimodal_lora_limit_triggered=(
                    decode_multimodal_lora_limit_triggered
                ),
                decode_multimodal_lora_limit_relaxed_by_fairness=(
                    decode_multimodal_lora_relaxed_by_fairness
                ),
                decode_multimodal_lora_limit_tightened_by_locality=(
                    decode_multimodal_lora_tightened_by_locality
                ),
                decode_multimodal_lora_max_fairness_gap=(
                    decode_multimodal_lora_max_fairness_gap
                ),
                decode_lora_adapters=decode_lora_adapters,
                effective_decode_lora_adapter_limit=effective_decode_lora_limit,
                decode_lora_limit_relaxed=decode_lora_relaxed,
                decode_lora_limit_tightened=decode_lora_tightened,
                decode_lora_fairness_gap=decode_lora_gap,
                decode_max_lora_fairness_gap=self._max_abs_share_gap(decode_lora_gap),
                queued_service_classes=queued_metrics["service_classes"],
                queued_multimodal_requests=queued_metrics["multimodal_requests"],
                queued_multimodal_lora_requests=queued_metrics[
                    "multimodal_lora_requests"
                ],
                queued_lora_adapters=queued_metrics["lora_adapters"],
                queued_avg_wait_s=queued_metrics["avg_wait_s"],
                queued_max_wait_s=queued_metrics["max_wait_s"],
                queued_p95_wait_s=queued_metrics["p95_wait_s"],
                queued_multimodal_avg_wait_s=queued_metrics["multimodal_avg_wait_s"],
                queued_multimodal_max_wait_s=queued_metrics["multimodal_max_wait_s"],
                queued_multimodal_p95_wait_s=queued_metrics["multimodal_p95_wait_s"],
                queued_service_class_avg_wait_s=queued_metrics[
                    "service_class_avg_wait_s"
                ],
                queued_service_class_max_wait_s=queued_metrics[
                    "service_class_max_wait_s"
                ],
                queued_service_class_p95_wait_s=queued_metrics[
                    "service_class_p95_wait_s"
                ],
                fairness_guardrail_triggered=fairness_guardrail_triggered,
            ),
        )
        self._update_prefill_deferrals(prefills, decodes, prefill_plan)
        self._update_decode_streak(plan)
        return plan

    def _build_single_request_decode_fast_path(self, scheduler) -> StepPlan | None:
        if scheduler.queued_request_count != 0 or scheduler.running_request_count != 1:
            return None
        rid = scheduler.running_ids[0]
        req = scheduler.get_request(rid)
        if req.is_prefill:
            return None
        return StepPlan(
            admissions=None,
            prefills=None,
            decodes=DecodePlan(
                request_ids=[rid],
                token_budget=1,
                use_fast_path=True,
            ),
            step_token_budget=self.step_token_budget,
            metrics=StepPlanMetrics(
                queued_before=0,
                running_before=1,
            ),
        )

    def _build_single_request_fast_path(self, scheduler) -> StepPlan | None:
        """Fast scheduler path for the common single-running-request case.

        This avoids fairness/admission bookkeeping when no queue exists and there is
        exactly one active running request. Numerical behavior is unchanged.
        """
        if scheduler.queued_request_count != 0 or scheduler.running_request_count != 1:
            return None

        rid = scheduler.running_ids[0]
        req = scheduler.get_request(rid)
        running_before = 1
        queued_before = 0

        if req.is_prefill:
            remaining = max(1, int(len(req.input_ids) - int(req.seq_len)))
            chunk_len = min(remaining, self.prefill_chunk_size, self.step_token_budget)
            prefill = PrefillPlan(
                request_ids=[rid],
                chunk_len=max(1, int(chunk_len)),
                token_budget=self.step_token_budget,
            )
            return StepPlan(
                admissions=None,
                prefills=prefill,
                decodes=None,
                step_token_budget=self.step_token_budget,
                metrics=StepPlanMetrics(
                    queued_before=queued_before,
                    running_before=running_before,
                ),
            )

        decode = DecodePlan(
            request_ids=[rid],
            token_budget=1,
            use_fast_path=True,
        )
        return StepPlan(
            admissions=None,
            prefills=None,
            decodes=decode,
            step_token_budget=self.step_token_budget,
            metrics=StepPlanMetrics(
                queued_before=queued_before,
                running_before=running_before,
            ),
        )

    def _should_protect_prefills(
        self,
        prefills: list[str],
        decodes: list[str],
        *,
        fairness_guardrail_triggered: bool,
    ) -> bool:
        if not (prefills and decodes):
            return False
        if fairness_guardrail_triggered:
            return True
        if self._decode_only_streak >= self.max_decode_streak:
            return True
        return any(
            self._prefill_deferrals.get(rid, 0) >= self.max_prefill_deferrals
            for rid in prefills
        )

    def _update_decode_streak(self, plan: StepPlan) -> None:
        if plan.prefills is not None:
            self._decode_only_streak = 0
        elif plan.decodes is not None:
            self._decode_only_streak += 1
        else:
            self._decode_only_streak = 0

    @staticmethod
    def _rotate_candidates(request_ids: list[str], cursor: int) -> list[str]:
        return rotate_candidates(request_ids, cursor)

    def _count_multimodal_lora_adapters(
        self,
        scheduler,
        request_ids: list[str],
    ) -> dict[str, int]:
        return self._multimodal_lora_constraint.count_multimodal_lora_adapters(
            scheduler, request_ids
        )

    def _shape_lora_batch(
        self,
        *,
        scheduler,
        request_ids: list[str],
        baseline_counts: dict[str, int],
        max_lora_adapters: int,
        cursor_attr: str,
    ) -> tuple[list[str], int, dict[str, float], bool, bool]:
        return self._lora_constraint.shape_lora_batch(
            scheduler=scheduler,
            step_scheduler=self,
            request_ids=request_ids,
            baseline_counts=baseline_counts,
            max_lora_adapters=max_lora_adapters,
            cursor_attr=cursor_attr,
        )

    def _lora_adapters_for_ids(
        self,
        scheduler,
        request_ids: list[str],
    ) -> list[str]:
        return self._lora_constraint.lora_adapters_for_ids(scheduler, request_ids)

    def _take_candidates(
        self,
        *,
        candidates: list[str],
        take: int,
        intra_class_rotate: bool,
    ) -> list[str]:
        return self._service_class_selector._take_candidates(
            step_scheduler=self,
            candidates=candidates,
            take=take,
            intra_class_rotate=intra_class_rotate,
        )

    def _compute_queued_metrics(
        self,
        scheduler,
    ) -> dict[str, object]:
        now = time.perf_counter()
        queued_ids = scheduler.queued_ids
        if not queued_ids:
            return {
                "service_classes": {},
                "multimodal_requests": 0,
                "multimodal_lora_requests": 0,
                "lora_adapters": {},
                "avg_wait_s": 0.0,
                "max_wait_s": 0.0,
                "p95_wait_s": 0.0,
                "multimodal_avg_wait_s": 0.0,
                "multimodal_max_wait_s": 0.0,
                "multimodal_p95_wait_s": 0.0,
                "service_class_avg_wait_s": {},
                "service_class_max_wait_s": {},
                "service_class_p95_wait_s": {},
            }
        counts: dict[str, int] = {}
        lora_counts: dict[str, int] = {}
        multimodal_waits: list[float] = []
        multimodal_lora_requests = 0
        wait_sums: dict[str, float] = {}
        wait_max: dict[str, float] = {}
        wait_values: dict[str, list[float]] = {}
        all_waits: list[float] = []
        total_wait = 0.0
        max_wait = 0.0
        for rid in queued_ids:
            request = scheduler.get_request(rid)
            service_class = str(request.service_class or "latency")
            lora_id = self._lora_adapter_key(request)
            queue_wait_s = max(0.0, now - float(request.queued_at or now))
            counts[service_class] = counts.get(service_class, 0) + 1
            lora_counts[lora_id] = lora_counts.get(lora_id, 0) + 1
            if self._is_multimodal(request):
                multimodal_waits.append(queue_wait_s)
                if self._is_multimodal_lora(request):
                    multimodal_lora_requests += 1
            wait_sums[service_class] = wait_sums.get(service_class, 0.0) + queue_wait_s
            wait_max[service_class] = max(
                wait_max.get(service_class, 0.0), queue_wait_s
            )
            wait_values.setdefault(service_class, []).append(queue_wait_s)
            total_wait += queue_wait_s
            max_wait = max(max_wait, queue_wait_s)
            all_waits.append(queue_wait_s)
        return {
            "service_classes": counts,
            "multimodal_requests": len(multimodal_waits),
            "multimodal_lora_requests": multimodal_lora_requests,
            "lora_adapters": lora_counts,
            "avg_wait_s": total_wait / max(1, len(queued_ids)),
            "max_wait_s": max_wait,
            "p95_wait_s": self._percentile(all_waits, 0.95),
            "multimodal_avg_wait_s": (
                sum(multimodal_waits) / max(1, len(multimodal_waits))
                if multimodal_waits
                else 0.0
            ),
            "multimodal_max_wait_s": max(multimodal_waits) if multimodal_waits else 0.0,
            "multimodal_p95_wait_s": self._percentile(multimodal_waits, 0.95),
            "service_class_avg_wait_s": {
                key: wait_sums[key] / max(1, counts[key]) for key in counts
            },
            "service_class_max_wait_s": wait_max,
            "service_class_p95_wait_s": {
                key: self._percentile(values, 0.95)
                for key, values in wait_values.items()
            },
        }

    @staticmethod
    def _normalize_quotas(quotas: dict[str, int]) -> dict[str, int]:
        return normalize_quotas(quotas)

    def _is_fairness_guardrail_triggered(
        self,
        queued_metrics: dict[str, object],
    ) -> bool:
        threshold = self.fairness_guardrail_queue_wait_s
        if threshold <= 0:
            return False
        if float(queued_metrics.get("p95_wait_s", 0.0) or 0.0) >= threshold:
            return True
        if not self.fairness_guardrail_service_classes:
            return False
        per_class_p95 = queued_metrics.get("service_class_p95_wait_s", {}) or {}
        return any(
            float(per_class_p95.get(service_class, 0.0) or 0.0) >= threshold
            for service_class in self.fairness_guardrail_service_classes
        )

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        return percentile(values, q)

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

    def _count_multimodal_requests(
        self,
        scheduler,
        request_ids: list[str],
    ) -> int:
        return self._multimodal_constraint.count_multimodal_requests(
            scheduler, request_ids
        )

    def _count_multimodal_lora_requests(
        self,
        scheduler,
        request_ids: list[str],
    ) -> int:
        return self._multimodal_lora_constraint.count_multimodal_lora_requests(
            scheduler, request_ids
        )

    def _count_lora_adapters(
        self,
        scheduler,
        request_ids: list[str],
    ) -> dict[str, int]:
        return self._lora_constraint.count_lora_adapters(scheduler, request_ids)

    @staticmethod
    def _is_multimodal(request) -> bool:
        return is_multimodal(request)

    @classmethod
    def _is_multimodal_lora(cls, request) -> bool:
        return is_multimodal_lora(request)

    @classmethod
    def _is_multimodal_request(cls, scheduler, request_id: str) -> bool:
        return is_multimodal(scheduler.get_request(request_id))

    @classmethod
    def _is_multimodal_lora_request(cls, scheduler, request_id: str) -> bool:
        return is_multimodal_lora(scheduler.get_request(request_id))

    @staticmethod
    def _normalized_share_map(counts: dict[str, int]) -> dict[str, float]:
        return normalized_share_map(counts)

    @staticmethod
    def _share_gap_map(
        target_share: dict[str, float],
        baseline_share: dict[str, float],
    ) -> dict[str, float]:
        return share_gap_map(target_share, baseline_share)

    @staticmethod
    def _max_abs_share_gap(gaps: dict[str, float]) -> float:
        return max_abs_share_gap(gaps)

    @classmethod
    def _lora_adapter_key(cls, request) -> str:
        return lora_adapter_key(request, base_lora_adapter=cls.BASE_LORA_ADAPTER)

    @staticmethod
    def _service_class_priority(service_class: object) -> int:
        return service_class_priority(service_class)

    def _update_prefill_deferrals(
        self,
        prefills: list[str],
        decodes: list[str],
        prefill_plan: PrefillPlan | None,
    ) -> None:
        current_prefills = set(prefills)
        for rid in list(self._prefill_deferrals):
            if rid not in current_prefills:
                self._prefill_deferrals.pop(rid, None)
        if not prefills:
            return
        selected = set(prefill_plan.request_ids) if prefill_plan is not None else set()
        if prefill_plan is not None:
            for rid in selected:
                self._prefill_deferrals[rid] = 0
        if not decodes:
            for rid in current_prefills - selected:
                self._prefill_deferrals.setdefault(rid, 0)
            return
        for rid in current_prefills - selected:
            self._prefill_deferrals[rid] = self._prefill_deferrals.get(rid, 0) + 1
