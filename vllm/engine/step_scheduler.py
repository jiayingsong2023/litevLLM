# SPDX-License-Identifier: Apache-2.0
import time

from vllm.engine.step_plan import AdmissionPlan, DecodePlan, PrefillPlan, StepPlan


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
        max_admit_lora_adapters_per_step: int = 0,
        max_prefill_lora_adapters_per_batch: int = 0,
        max_decode_lora_adapters_per_batch: int = 0,
        lora_fairness_relax_threshold: float = 0.0,
        lora_locality_tighten_threshold: float = 0.0,
        lora_limit_relax_delta: int = 1,
        lora_limit_tighten_delta: int = 1,
        max_admit_multimodal_per_step: int = 0,
        max_prefill_multimodal_requests_per_batch: int = 0,
        max_decode_multimodal_requests_per_batch: int = 0,
        max_admit_multimodal_lora_per_step: int = 0,
        max_prefill_multimodal_lora_requests_per_batch: int = 0,
        max_decode_multimodal_lora_requests_per_batch: int = 0,
        multimodal_prefix_cache_relax_threshold: float = 0.0,
        multimodal_prefix_cache_tighten_threshold: float = 0.0,
        multimodal_prefill_limit_relax_delta: int = 1,
        multimodal_prefill_limit_tighten_delta: int = 1,
        multimodal_lora_prefill_limit_relax_delta: int = 1,
        multimodal_lora_prefill_limit_tighten_delta: int = 1,
        multimodal_lora_fairness_relax_threshold: float = 0.0,
        multimodal_lora_locality_tighten_threshold: float = 0.0,
    ) -> None:
        self.step_token_budget = max(1, int(step_token_budget))
        self.decode_priority_enabled = decode_priority_enabled
        self.prefill_chunk_size = max(1, int(prefill_chunk_size))
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
            admission_service_class_quotas or self.DEFAULT_ADMISSION_SERVICE_CLASS_QUOTAS
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
        self.max_admit_lora_adapters_per_step = max(0, int(max_admit_lora_adapters_per_step))
        self.max_prefill_lora_adapters_per_batch = max(
            0, int(max_prefill_lora_adapters_per_batch)
        )
        self.max_decode_lora_adapters_per_batch = max(
            0, int(max_decode_lora_adapters_per_batch)
        )
        self.lora_fairness_relax_threshold = max(
            0.0, float(lora_fairness_relax_threshold)
        )
        self.lora_locality_tighten_threshold = max(
            0.0, float(lora_locality_tighten_threshold)
        )
        self.lora_limit_relax_delta = max(1, int(lora_limit_relax_delta))
        self.lora_limit_tighten_delta = max(1, int(lora_limit_tighten_delta))
        self.max_admit_multimodal_per_step = max(0, int(max_admit_multimodal_per_step))
        self.max_prefill_multimodal_requests_per_batch = max(
            0, int(max_prefill_multimodal_requests_per_batch)
        )
        self.max_decode_multimodal_requests_per_batch = max(
            0, int(max_decode_multimodal_requests_per_batch)
        )
        self.max_admit_multimodal_lora_per_step = max(
            0, int(max_admit_multimodal_lora_per_step)
        )
        self.max_prefill_multimodal_lora_requests_per_batch = max(
            0, int(max_prefill_multimodal_lora_requests_per_batch)
        )
        self.max_decode_multimodal_lora_requests_per_batch = max(
            0, int(max_decode_multimodal_lora_requests_per_batch)
        )
        self.multimodal_prefix_cache_relax_threshold = max(
            0.0, float(multimodal_prefix_cache_relax_threshold)
        )
        self.multimodal_prefix_cache_tighten_threshold = max(
            0.0, float(multimodal_prefix_cache_tighten_threshold)
        )
        self.multimodal_prefill_limit_relax_delta = max(
            1, int(multimodal_prefill_limit_relax_delta)
        )
        self.multimodal_prefill_limit_tighten_delta = max(
            1, int(multimodal_prefill_limit_tighten_delta)
        )
        self.multimodal_lora_prefill_limit_relax_delta = max(
            1, int(multimodal_lora_prefill_limit_relax_delta)
        )
        self.multimodal_lora_prefill_limit_tighten_delta = max(
            1, int(multimodal_lora_prefill_limit_tighten_delta)
        )
        self.multimodal_lora_fairness_relax_threshold = max(
            0.0, float(multimodal_lora_fairness_relax_threshold)
        )
        self.multimodal_lora_locality_tighten_threshold = max(
            0.0, float(multimodal_lora_locality_tighten_threshold)
        )
        self._decode_only_streak = 0
        self._decode_rr_cursor = 0
        self._prefill_rr_cursor = 0
        self._prefill_deferrals: dict[str, int] = {}
        self._admission_service_cursor = 0
        self._decode_service_cursor = 0
        self._admission_lora_cursor = 0
        self._prefill_lora_cursor = 0
        self._decode_lora_cursor = 0
        self._admission_multimodal_cursor = 0
        self._prefill_multimodal_cursor = 0
        self._decode_multimodal_cursor = 0
        self._admission_multimodal_lora_cursor = 0
        self._prefill_multimodal_lora_cursor = 0
        self._decode_multimodal_lora_cursor = 0
        self._multimodal_prefix_cache_hit_rate_feedback = 0.0

    def update_runtime_feedback(self, stats: dict[str, object] | None) -> None:
        observer = stats.get("observer") if isinstance(stats, dict) else None
        multimodal = observer.get("multimodal") if isinstance(observer, dict) else None
        if isinstance(multimodal, dict):
            self._multimodal_prefix_cache_hit_rate_feedback = float(
                multimodal.get("prefix_cache_hit_rate", 0.0) or 0.0
            )
        else:
            self._multimodal_prefix_cache_hit_rate_feedback = 0.0

    def build_plan(self, scheduler) -> StepPlan:
        fast_plan = self._build_single_request_fast_path(scheduler)
        if fast_plan is not None:
            self._update_prefill_deferrals(
                [rid for rid in fast_plan.prefills.request_ids] if fast_plan.prefills else [],
                [rid for rid in fast_plan.decodes.request_ids] if fast_plan.decodes else [],
                fast_plan.prefills,
            )
            self._update_decode_streak(fast_plan)
            return fast_plan

        queued_metrics = self._compute_queued_metrics(scheduler)
        (
            admissions,
            aged_admission_count,
            admitted_service_classes,
            admitted_lora_gap,
            effective_admit_lora_limit,
            effective_admit_multimodal_lora_limit,
            admit_multimodal_lora_limit_triggered,
            admit_lora_relaxed,
            admit_lora_tightened,
        ) = self._build_admission_plan(
            scheduler
        )
        prefills, decodes = scheduler.classify_requests()
        fairness_guardrail_triggered = self._is_fairness_guardrail_triggered(queued_metrics)
        starvation_protected = self._should_protect_prefills(
            prefills,
            decodes,
            fairness_guardrail_triggered=fairness_guardrail_triggered,
        )
        prefill_budget, decode_limit = self._compute_budgets(
            num_prefills=len(prefills),
            num_decodes=len(decodes),
            starvation_protected=starvation_protected,
        )

        (
            prefill_plan,
            prefill_lora_gap,
            effective_prefill_multimodal_limit,
            effective_prefill_lora_limit,
            effective_prefill_multimodal_lora_limit,
            prefill_multimodal_limit_triggered,
            prefill_multimodal_lora_limit_triggered,
            prefill_multimodal_relaxed,
            prefill_multimodal_tightened,
            prefill_multimodal_lora_relaxed,
            prefill_multimodal_lora_tightened,
            prefill_multimodal_lora_relaxed_by_fairness,
            prefill_multimodal_lora_tightened_by_locality,
            prefill_multimodal_lora_max_fairness_gap,
            prefill_lora_relaxed,
            prefill_lora_tightened,
        ) = self._build_prefill_plan(scheduler, prefills, prefill_budget)
        prefill_request_ids = prefill_plan.request_ids if prefill_plan else []
        prefill_service_classes = self._count_service_classes(scheduler, prefill_request_ids)
        prefill_lora_adapters = self._count_lora_adapters(scheduler, prefill_request_ids)
        (
            decode_plan,
            decode_lora_gap,
            effective_decode_lora_limit,
            effective_decode_multimodal_lora_limit,
            decode_multimodal_lora_limit_triggered,
            decode_multimodal_lora_relaxed,
            decode_multimodal_lora_tightened,
            decode_multimodal_lora_relaxed_by_fairness,
            decode_multimodal_lora_tightened_by_locality,
            decode_multimodal_lora_max_fairness_gap,
            decode_lora_relaxed,
            decode_lora_tightened,
        ) = self._build_decode_plan(
            scheduler,
            decodes,
            decode_limit,
            prefill_plan is None,
        )
        decode_request_ids = decode_plan.request_ids if decode_plan else []
        decode_service_classes = self._count_service_classes(scheduler, decode_request_ids)
        decode_lora_adapters = self._count_lora_adapters(scheduler, decode_request_ids)
        admitted_request_ids = admissions.request_ids if admissions else []
        plan = StepPlan(
            admissions=admissions,
            prefills=prefill_plan,
            decodes=decode_plan,
            step_token_budget=self.step_token_budget,
            queued_before=scheduler.queued_request_count,
            running_before=scheduler.running_request_count,
            multimodal_prefix_cache_hit_rate=self._multimodal_prefix_cache_hit_rate_feedback,
            prefill_starvation_protected=bool(starvation_protected and prefill_plan is not None),
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
            admitted_lora_adapters=self._count_lora_adapters(scheduler, admitted_request_ids),
            effective_admit_lora_adapter_limit=effective_admit_lora_limit,
            admit_lora_limit_relaxed=admit_lora_relaxed,
            admit_lora_limit_tightened=admit_lora_tightened,
            admitted_lora_fairness_gap=admitted_lora_gap,
            admitted_max_lora_fairness_gap=self._max_abs_share_gap(admitted_lora_gap),
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
            prefill_multimodal_lora_limit_relaxed=(
                prefill_multimodal_lora_relaxed
            ),
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
            queued_multimodal_lora_requests=queued_metrics["multimodal_lora_requests"],
            queued_lora_adapters=queued_metrics["lora_adapters"],
            queued_avg_wait_s=queued_metrics["avg_wait_s"],
            queued_max_wait_s=queued_metrics["max_wait_s"],
            queued_p95_wait_s=queued_metrics["p95_wait_s"],
            queued_multimodal_avg_wait_s=queued_metrics["multimodal_avg_wait_s"],
            queued_multimodal_max_wait_s=queued_metrics["multimodal_max_wait_s"],
            queued_multimodal_p95_wait_s=queued_metrics["multimodal_p95_wait_s"],
            queued_service_class_avg_wait_s=queued_metrics["service_class_avg_wait_s"],
            queued_service_class_max_wait_s=queued_metrics["service_class_max_wait_s"],
            queued_service_class_p95_wait_s=queued_metrics["service_class_p95_wait_s"],
            fairness_guardrail_triggered=fairness_guardrail_triggered,
        )
        self._update_prefill_deferrals(prefills, decodes, prefill_plan)
        self._update_decode_streak(plan)
        return plan

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

        if bool(req.get("is_prefill", False)):
            remaining = max(1, int(len(req["input_ids"]) - int(req["seq_len"])))
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
                queued_before=queued_before,
                running_before=running_before,
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
            queued_before=queued_before,
            running_before=running_before,
        )

    def _build_admission_plan(
        self,
        scheduler,
    ) -> tuple[
        AdmissionPlan | None,
        int,
        dict[str, int],
        dict[str, float],
        int,
        int,
        bool,
        bool,
        bool,
    ]:
        if not scheduler.has_capacity() or scheduler.queued_request_count == 0:
            return None, 0, {}, {}, 0, 0, False, False, False
        now = time.perf_counter()
        queued_ids = scheduler.queued_ids
        aged_ids = [
            rid
            for rid in queued_ids
            if (now - float(scheduler.get_request(rid).get("queued_at") or now))
            >= self.queue_aging_threshold_s
        ]
        non_aged_ids = [rid for rid in queued_ids if rid not in set(aged_ids)]
        admit_limit = min(
            scheduler.queued_request_count,
            scheduler.available_slots,
            self.max_admit_per_step,
        )
        request_ids = self._select_weighted_requests(
            scheduler=scheduler,
            request_ids=aged_ids,
            limit=admit_limit,
            cursor_attr="_admission_service_cursor",
            prefer_short_prompts=False,
            quotas=self.admission_service_class_quotas,
        )
        if len(request_ids) < admit_limit:
            request_ids.extend(
                self._select_weighted_requests(
                    scheduler=scheduler,
                    request_ids=[rid for rid in non_aged_ids if rid not in set(request_ids)],
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
        ) = self._shape_lora_batch(
            scheduler=scheduler,
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
        ) = self._apply_multimodal_lora_request_limit(
            scheduler=scheduler,
            request_ids=request_ids,
            max_multimodal_lora_requests=self.max_admit_multimodal_lora_per_step,
            cursor_attr="_admission_multimodal_lora_cursor",
        )
        request_ids = self._apply_multimodal_request_limit(
            scheduler=scheduler,
            request_ids=request_ids,
            max_multimodal_requests=self.max_admit_multimodal_per_step,
            cursor_attr="_admission_multimodal_cursor",
        )
        if not request_ids:
            return (
                None,
                0,
                {},
                {},
                effective_lora_limit,
                effective_multimodal_lora_limit,
                multimodal_lora_limit_triggered,
                relaxed,
                tightened,
            )
        aged_admission_count = sum(1 for rid in request_ids if rid in set(aged_ids))
        return (
            AdmissionPlan(request_ids=request_ids),
            aged_admission_count,
            self._count_service_classes(scheduler, request_ids),
            fairness_gap,
            effective_lora_limit,
            effective_multimodal_lora_limit,
            multimodal_lora_limit_triggered,
            relaxed,
            tightened,
        )

    def _compute_budgets(
        self,
        *,
        num_prefills: int,
        num_decodes: int,
        starvation_protected: bool,
    ) -> tuple[int, int]:
        if self.decode_priority_enabled:
            prefill_budget = 0
            if starvation_protected and num_prefills:
                reserve_tokens = max(
                    1,
                    self.prefill_reserved_tokens,
                    int(self.step_token_budget * max(self.prefill_catchup_ratio, 0.25)),
                )
                prefill_budget = min(self.step_token_budget, reserve_tokens)
            elif num_prefills and not num_decodes:
                prefill_budget = min(self.prefill_chunk_size, self.step_token_budget)
            elif num_prefills and num_prefills >= self.prefill_reserve_backlog:
                reserve_tokens = max(
                    self.prefill_reserved_tokens,
                    int(self.step_token_budget * self.prefill_catchup_ratio),
                )
                prefill_budget = min(self.step_token_budget, max(1, reserve_tokens))
            decode_limit = min(num_decodes, max(0, self.step_token_budget - prefill_budget))
        else:
            if num_prefills:
                reserve_tokens = max(1, self.prefill_reserved_tokens or 1)
                decode_limit = max(
                    0, min(num_decodes, self.step_token_budget - reserve_tokens)
                )
            else:
                decode_limit = min(num_decodes, self.step_token_budget)
            prefill_budget = max(0, self.step_token_budget - decode_limit)
        return prefill_budget, decode_limit

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

    def _build_prefill_plan(
        self,
        scheduler,
        prefills: list[str],
        token_budget: int,
    ) -> tuple[
        PrefillPlan | None,
        dict[str, float],
        int,
        int,
        int,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        float,
        bool,
        bool,
    ]:
        if not prefills or token_budget <= 0:
            return (
                None,
                {},
                0,
                0,
                0,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                0.0,
                False,
                False,
            )
        base_processed_len = scheduler.get_request(prefills[0])["seq_len"]
        candidate_prefills = [
            rid for rid in prefills if scheduler.get_request(rid)["seq_len"] == base_processed_len
        ]
        if not candidate_prefills:
            return (
                None,
                {},
                0,
                0,
                0,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                0.0,
                False,
                False,
            )
        request_ids = self._rotate_candidates(
            candidate_prefills,
            self._prefill_rr_cursor,
        )[: min(self.prefill_microbatch_size, len(candidate_prefills))]
        self._prefill_rr_cursor = (
            self._prefill_rr_cursor + len(request_ids)
        ) % max(1, len(candidate_prefills))
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
            return (
                None,
                fairness_gap,
                effective_multimodal_limit,
                effective_lora_limit,
                effective_multimodal_lora_limit,
                multimodal_limit_triggered,
                multimodal_lora_limit_triggered,
                multimodal_relaxed,
                multimodal_tightened,
                multimodal_lora_relaxed,
                multimodal_lora_tightened,
                multimodal_lora_relaxed_by_fairness,
                multimodal_lora_tightened_by_locality,
                multimodal_lora_max_fairness_gap,
                relaxed,
                tightened,
            )
        min_remaining = min(
            len(scheduler.get_request(rid)["input_ids"]) - scheduler.get_request(rid)["seq_len"]
            for rid in request_ids
        )
        per_req_budget = max(1, token_budget // max(1, len(request_ids)))
        chunk_len = min(min_remaining, self.prefill_chunk_size, per_req_budget)
        if chunk_len <= 0:
            chunk_len = 1
        return (
            PrefillPlan(
                request_ids=request_ids,
                chunk_len=chunk_len,
                token_budget=token_budget,
            ),
            fairness_gap,
            effective_multimodal_limit,
            effective_lora_limit,
            effective_multimodal_lora_limit,
            multimodal_limit_triggered,
            multimodal_lora_limit_triggered,
            multimodal_relaxed,
            multimodal_tightened,
            multimodal_lora_relaxed,
            multimodal_lora_tightened,
            multimodal_lora_relaxed_by_fairness,
            multimodal_lora_tightened_by_locality,
            multimodal_lora_max_fairness_gap,
            relaxed,
            tightened,
        )

    def _effective_prefill_multimodal_lora_limit(
        self,
        *,
        scheduler,
        candidate_request_ids: list[str],
        current_request_ids: list[str],
    ) -> tuple[int, bool, bool, float]:
        base_limit = self.max_prefill_multimodal_lora_requests_per_batch
        if base_limit <= 0:
            return 0, False, False, 0.0
        hit_rate = self._multimodal_prefix_cache_hit_rate_feedback
        candidate_count = self._count_multimodal_lora_requests(
            scheduler, candidate_request_ids
        )
        effective_limit = base_limit
        baseline_counts = self._count_multimodal_lora_adapters(
            scheduler, candidate_request_ids
        )
        current_counts = self._count_multimodal_lora_adapters(
            scheduler, current_request_ids
        )
        fairness_gap = self._share_gap_map(
            self._normalized_share_map(current_counts),
            self._normalized_share_map(baseline_counts),
        )
        max_gap = self._max_abs_share_gap(fairness_gap)
        relaxed_by_fairness = False
        tightened_by_locality = False
        if (
            self.multimodal_lora_fairness_relax_threshold > 0
            and max_gap >= self.multimodal_lora_fairness_relax_threshold
        ):
            effective_limit = min(
                candidate_count,
                effective_limit + self.multimodal_lora_prefill_limit_relax_delta,
            )
            relaxed_by_fairness = effective_limit > base_limit
        elif (
            self.multimodal_prefix_cache_relax_threshold > 0
            and hit_rate <= self.multimodal_prefix_cache_relax_threshold
        ):
            effective_limit = min(
                candidate_count,
                effective_limit + self.multimodal_lora_prefill_limit_relax_delta,
            )
        elif (
            self.multimodal_prefix_cache_tighten_threshold > 0
            and hit_rate >= self.multimodal_prefix_cache_tighten_threshold
            and (
                self.multimodal_lora_locality_tighten_threshold <= 0
                or max_gap <= self.multimodal_lora_locality_tighten_threshold
            )
            and effective_limit > 1
        ):
            effective_limit = max(
                1,
                effective_limit - self.multimodal_lora_prefill_limit_tighten_delta,
            )
            tightened_by_locality = (
                self.multimodal_lora_locality_tighten_threshold > 0
                and max_gap <= self.multimodal_lora_locality_tighten_threshold
                and effective_limit < base_limit
            )
        return effective_limit, relaxed_by_fairness, tightened_by_locality, max_gap

    def _effective_decode_multimodal_lora_limit(
        self,
        *,
        scheduler,
        candidate_request_ids: list[str],
        current_request_ids: list[str],
    ) -> tuple[int, bool, bool, float]:
        base_limit = self.max_decode_multimodal_lora_requests_per_batch
        if base_limit <= 0:
            return 0, False, False, 0.0
        hit_rate = self._multimodal_prefix_cache_hit_rate_feedback
        candidate_count = self._count_multimodal_lora_requests(
            scheduler, candidate_request_ids
        )
        effective_limit = base_limit
        baseline_counts = self._count_multimodal_lora_adapters(
            scheduler, candidate_request_ids
        )
        current_counts = self._count_multimodal_lora_adapters(
            scheduler, current_request_ids
        )
        fairness_gap = self._share_gap_map(
            self._normalized_share_map(current_counts),
            self._normalized_share_map(baseline_counts),
        )
        max_gap = self._max_abs_share_gap(fairness_gap)
        relaxed_by_fairness = False
        tightened_by_locality = False
        if (
            self.multimodal_lora_fairness_relax_threshold > 0
            and max_gap >= self.multimodal_lora_fairness_relax_threshold
        ):
            effective_limit = min(
                candidate_count,
                effective_limit + self.multimodal_lora_prefill_limit_relax_delta,
            )
            relaxed_by_fairness = effective_limit > base_limit
        elif (
            self.multimodal_prefix_cache_relax_threshold > 0
            and hit_rate <= self.multimodal_prefix_cache_relax_threshold
        ):
            effective_limit = min(
                candidate_count,
                effective_limit + self.multimodal_lora_prefill_limit_relax_delta,
            )
        elif (
            self.multimodal_prefix_cache_tighten_threshold > 0
            and hit_rate >= self.multimodal_prefix_cache_tighten_threshold
            and (
                self.multimodal_lora_locality_tighten_threshold <= 0
                or max_gap <= self.multimodal_lora_locality_tighten_threshold
            )
            and effective_limit > 1
        ):
            effective_limit = max(
                1,
                effective_limit - self.multimodal_lora_prefill_limit_tighten_delta,
            )
            tightened_by_locality = (
                self.multimodal_lora_locality_tighten_threshold > 0
                and max_gap <= self.multimodal_lora_locality_tighten_threshold
                and effective_limit < base_limit
            )
        return effective_limit, relaxed_by_fairness, tightened_by_locality, max_gap

    def _shape_multimodal_prefill_batch(
        self,
        *,
        scheduler,
        request_ids: list[str],
        candidate_prefills: list[str],
    ) -> tuple[list[str], int, bool, bool, bool]:
        effective_limit = self.max_prefill_multimodal_requests_per_batch
        relaxed = False
        tightened = False
        if effective_limit > 0:
            candidate_mm_count = self._count_multimodal_requests(scheduler, candidate_prefills)
            hit_rate = self._multimodal_prefix_cache_hit_rate_feedback
            if (
                self.multimodal_prefix_cache_relax_threshold > 0
                and hit_rate <= self.multimodal_prefix_cache_relax_threshold
            ):
                effective_limit = min(
                    candidate_mm_count,
                    effective_limit + self.multimodal_prefill_limit_relax_delta,
                )
                relaxed = effective_limit > self.max_prefill_multimodal_requests_per_batch
            elif (
                self.multimodal_prefix_cache_tighten_threshold > 0
                and hit_rate >= self.multimodal_prefix_cache_tighten_threshold
                and effective_limit > 1
            ):
                effective_limit = max(
                    1,
                    effective_limit - self.multimodal_prefill_limit_tighten_delta,
                )
                tightened = effective_limit < self.max_prefill_multimodal_requests_per_batch
        shaped = self._apply_multimodal_request_limit(
            scheduler=scheduler,
            request_ids=request_ids,
            max_multimodal_requests=effective_limit,
            cursor_attr="_prefill_multimodal_cursor",
        )
        triggered = shaped != request_ids
        return shaped, effective_limit, triggered, relaxed, tightened

    def _build_decode_plan(
        self,
        scheduler,
        decodes: list[str],
        decode_limit: int,
        no_prefills_selected: bool,
    ) -> tuple[
        DecodePlan | None,
        dict[str, float],
        int,
        int,
        bool,
        bool,
        bool,
        bool,
        float,
        bool,
        bool,
    ]:
        if decode_limit < len(decodes):
            request_ids = self._select_weighted_requests(
                scheduler=scheduler,
                request_ids=decodes,
                limit=decode_limit,
                cursor_attr="_decode_service_cursor",
                prefer_short_prompts=False,
                intra_class_rotate=True,
                quotas=self.decode_service_class_quotas,
            )
        else:
            request_ids = decodes[:decode_limit]
            self._decode_rr_cursor = 0
            self._decode_service_cursor = 0
        (
            request_ids,
            effective_lora_limit,
            fairness_gap,
            relaxed,
            tightened,
        ) = self._shape_lora_batch(
            scheduler=scheduler,
            request_ids=request_ids,
            baseline_counts=self._count_lora_adapters(scheduler, decodes),
            max_lora_adapters=self.max_decode_lora_adapters_per_batch,
            cursor_attr="_decode_lora_cursor",
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
                self.max_decode_multimodal_lora_requests_per_batch
            ),
            cursor_attr="_decode_multimodal_lora_cursor",
            candidate_request_ids=decodes,
        )
        request_ids = self._apply_multimodal_request_limit(
            scheduler=scheduler,
            request_ids=request_ids,
            max_multimodal_requests=self.max_decode_multimodal_requests_per_batch,
            cursor_attr="_decode_multimodal_cursor",
        )
        if not request_ids:
            return (
                None,
                fairness_gap,
                effective_lora_limit,
                effective_multimodal_lora_limit,
                multimodal_lora_limit_triggered,
                multimodal_lora_relaxed,
                multimodal_lora_tightened,
                multimodal_lora_relaxed_by_fairness,
                multimodal_lora_tightened_by_locality,
                multimodal_lora_max_fairness_gap,
                relaxed,
                tightened,
            )
        use_fast_path = bool(no_prefills_selected and len(request_ids) == len(decodes))
        return (
            DecodePlan(
                request_ids=request_ids,
                token_budget=decode_limit,
                use_fast_path=use_fast_path,
            ),
            fairness_gap,
            effective_lora_limit,
            effective_multimodal_lora_limit,
            multimodal_lora_limit_triggered,
            multimodal_lora_relaxed,
            multimodal_lora_tightened,
            multimodal_lora_relaxed_by_fairness,
            multimodal_lora_tightened_by_locality,
            multimodal_lora_max_fairness_gap,
            relaxed,
            tightened,
        )

    @staticmethod
    def _rotate_candidates(request_ids: list[str], cursor: int) -> list[str]:
        if not request_ids:
            return []
        offset = cursor % len(request_ids)
        return request_ids[offset:] + request_ids[:offset]

    def _service_classes_for_ids(
        self,
        scheduler,
        request_ids: list[str],
    ) -> list[str]:
        present = {
            str(scheduler.get_request(rid).get("service_class") or "latency")
            for rid in request_ids
        }
        return sorted(
            present,
            key=lambda service_class: (
                self._service_class_priority(service_class),
                service_class,
            ),
        )

    def _apply_lora_adapter_batch_limit(
        self,
        *,
        scheduler,
        request_ids: list[str],
        max_lora_adapters: int,
        cursor_attr: str,
    ) -> list[str]:
        if max_lora_adapters <= 0 or len(request_ids) <= 1:
            return request_ids
        adapter_order = self._lora_adapters_for_ids(scheduler, request_ids)
        if len(adapter_order) <= max_lora_adapters:
            return request_ids
        cursor = getattr(self, cursor_attr) % max(1, len(adapter_order))
        allowed_adapters = set(
            self._rotate_candidates(adapter_order, cursor)[:max_lora_adapters]
        )
        setattr(
            self,
            cursor_attr,
            (getattr(self, cursor_attr) + 1) % max(1, len(adapter_order)),
        )
        shaped = [
            rid
            for rid in request_ids
            if self._lora_adapter_key(scheduler.get_request(rid)) in allowed_adapters
        ]
        return shaped or request_ids[:1]

    def _apply_multimodal_request_limit(
        self,
        *,
        scheduler,
        request_ids: list[str],
        max_multimodal_requests: int,
        cursor_attr: str,
    ) -> list[str]:
        if max_multimodal_requests <= 0 or len(request_ids) <= 1:
            return request_ids
        multimodal_ids = [
            rid for rid in request_ids if self._is_multimodal_request(scheduler, rid)
        ]
        if len(multimodal_ids) <= max_multimodal_requests:
            return request_ids
        cursor = getattr(self, cursor_attr) % max(1, len(multimodal_ids))
        allowed_multimodal = set(
            self._rotate_candidates(multimodal_ids, cursor)[:max_multimodal_requests]
        )
        setattr(
            self,
            cursor_attr,
            (getattr(self, cursor_attr) + 1) % max(1, len(multimodal_ids)),
        )
        shaped = [
            rid
            for rid in request_ids
            if (not self._is_multimodal_request(scheduler, rid)) or rid in allowed_multimodal
        ]
        return shaped or request_ids[:1]

    def _apply_multimodal_lora_request_limit(
        self,
        *,
        scheduler,
        request_ids: list[str],
        max_multimodal_lora_requests: int,
        cursor_attr: str,
        candidate_request_ids: list[str] | None = None,
    ) -> tuple[list[str], int, bool, bool, bool, bool, bool, float]:
        if cursor_attr == "_prefill_multimodal_lora_cursor":
            base_limit = self.max_prefill_multimodal_lora_requests_per_batch
        elif cursor_attr == "_decode_multimodal_lora_cursor":
            base_limit = self.max_decode_multimodal_lora_requests_per_batch
        else:
            base_limit = max_multimodal_lora_requests
        relaxed = (
            max_multimodal_lora_requests > 0 and max_multimodal_lora_requests > base_limit
        )
        tightened = (
            max_multimodal_lora_requests > 0 and base_limit > 0 and max_multimodal_lora_requests < base_limit
        )
        effective_limit, relaxed_by_fairness, tightened_by_locality, max_fairness_gap = (
            self._effective_prefill_multimodal_lora_limit(
                scheduler=scheduler,
                candidate_request_ids=candidate_request_ids or request_ids,
                current_request_ids=request_ids,
            )
            if cursor_attr == "_prefill_multimodal_lora_cursor"
            else self._effective_decode_multimodal_lora_limit(
                scheduler=scheduler,
                candidate_request_ids=candidate_request_ids or request_ids,
                current_request_ids=request_ids,
            )
            if cursor_attr == "_decode_multimodal_lora_cursor"
            else (max_multimodal_lora_requests, False, False, 0.0)
        )
        if cursor_attr in {
            "_prefill_multimodal_lora_cursor",
            "_decode_multimodal_lora_cursor",
        }:
            max_multimodal_lora_requests = effective_limit
            relaxed = max_multimodal_lora_requests > base_limit
            tightened = (
                max_multimodal_lora_requests > 0
                and base_limit > 0
                and max_multimodal_lora_requests < base_limit
            )
        if max_multimodal_lora_requests <= 0 or len(request_ids) <= 1:
            return (
                request_ids,
                max(0, int(max_multimodal_lora_requests)),
                False,
                relaxed,
                tightened,
                relaxed_by_fairness,
                tightened_by_locality,
                max_fairness_gap,
            )
        multimodal_lora_ids = [
            rid
            for rid in request_ids
            if self._is_multimodal_lora_request(scheduler, rid)
        ]
        if len(multimodal_lora_ids) <= max_multimodal_lora_requests:
            return (
                request_ids,
                max_multimodal_lora_requests,
                False,
                relaxed,
                tightened,
                relaxed_by_fairness,
                tightened_by_locality,
                max_fairness_gap,
            )
        cursor = getattr(self, cursor_attr) % max(1, len(multimodal_lora_ids))
        allowed_multimodal_lora = set(
            self._rotate_candidates(multimodal_lora_ids, cursor)[
                :max_multimodal_lora_requests
            ]
        )
        setattr(
            self,
            cursor_attr,
            (getattr(self, cursor_attr) + 1) % max(1, len(multimodal_lora_ids)),
        )
        shaped = [
            rid
            for rid in request_ids
            if (not self._is_multimodal_lora_request(scheduler, rid))
            or rid in allowed_multimodal_lora
        ]
        return (
            shaped or request_ids[:1],
            max_multimodal_lora_requests,
            True,
            relaxed,
            tightened,
            relaxed_by_fairness,
            tightened_by_locality,
            max_fairness_gap,
        )

    def _count_multimodal_lora_adapters(
        self,
        scheduler,
        request_ids: list[str],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rid in request_ids:
            if not self._is_multimodal_lora_request(scheduler, rid):
                continue
            adapter_key = self._lora_adapter_key(scheduler.get_request(rid))
            counts[adapter_key] = counts.get(adapter_key, 0) + 1
        return counts

    def _shape_lora_batch(
        self,
        *,
        scheduler,
        request_ids: list[str],
        baseline_counts: dict[str, int],
        max_lora_adapters: int,
        cursor_attr: str,
    ) -> tuple[list[str], int, dict[str, float], bool, bool]:
        if not request_ids:
            return [], max_lora_adapters, {}, False, False
        baseline_share = self._normalized_share_map(baseline_counts)
        pre_limit_gap = self._share_gap_map(
            self._normalized_share_map(self._count_lora_adapters(scheduler, request_ids)),
            baseline_share,
        )
        effective_limit = max_lora_adapters
        relaxed = False
        tightened = False
        if effective_limit > 0:
            adapter_count = len(self._lora_adapters_for_ids(scheduler, request_ids))
            max_gap = self._max_abs_share_gap(pre_limit_gap)
            if (
                self.lora_fairness_relax_threshold > 0
                and max_gap >= self.lora_fairness_relax_threshold
            ):
                effective_limit = min(
                    adapter_count,
                    effective_limit + self.lora_limit_relax_delta,
                )
                relaxed = effective_limit > max_lora_adapters
            elif (
                self.lora_locality_tighten_threshold > 0
                and max_gap <= self.lora_locality_tighten_threshold
                and effective_limit > 1
            ):
                effective_limit = max(
                    1,
                    effective_limit - self.lora_limit_tighten_delta,
                )
                tightened = effective_limit < max_lora_adapters
            request_ids = self._apply_lora_adapter_batch_limit(
                scheduler=scheduler,
                request_ids=request_ids,
                max_lora_adapters=effective_limit,
                cursor_attr=cursor_attr,
            )
        fairness_gap = self._share_gap_map(
            self._normalized_share_map(self._count_lora_adapters(scheduler, request_ids)),
            baseline_share,
        )
        return request_ids, effective_limit, fairness_gap, relaxed, tightened

    def _lora_adapters_for_ids(
        self,
        scheduler,
        request_ids: list[str],
    ) -> list[str]:
        adapters = {
            self._lora_adapter_key(scheduler.get_request(rid))
            for rid in request_ids
        }
        return sorted(adapters)

    def _select_weighted_requests(
        self,
        *,
        scheduler,
        request_ids: list[str],
        limit: int,
        cursor_attr: str,
        prefer_short_prompts: bool,
        intra_class_rotate: bool = False,
        quotas: dict[str, int] | None = None,
    ) -> list[str]:
        if limit <= 0 or not request_ids:
            return []
        groups: dict[str, list[str]] = {}
        for rid in request_ids:
            service_class = str(scheduler.get_request(rid).get("service_class") or "latency")
            groups.setdefault(service_class, []).append(rid)
        for service_class, grouped_ids in groups.items():
            groups[service_class] = sorted(
                grouped_ids,
                key=lambda rid: (
                    len(scheduler.get_request(rid).get("input_ids", []))
                    if prefer_short_prompts
                    else 0,
                    -float(scheduler.get_request(rid).get("queued_at") or 0.0),
                    rid,
                ),
            )
        service_classes = self._service_classes_for_ids(scheduler, request_ids)
        if not service_classes:
            return []
        selected: list[str] = []
        cursor = getattr(self, cursor_attr) % len(service_classes)
        rotated_classes = self._rotate_candidates(service_classes, cursor)
        requested_quotas = self._normalize_quotas(quotas or {})
        for service_class in rotated_classes:
            if len(selected) >= limit:
                break
            candidates = groups.get(service_class, [])
            if not candidates:
                continue
            quota = requested_quotas.get(service_class, 0)
            if quota <= 0:
                continue
            picked = self._take_candidates(
                candidates=candidates,
                take=min(quota, limit - len(selected), len(candidates)),
                intra_class_rotate=intra_class_rotate,
            )
            if not picked:
                continue
            selected.extend(picked)
            groups[service_class] = [rid for rid in groups[service_class] if rid not in set(picked)]
        while len(selected) < limit:
            progressed = False
            for service_class in rotated_classes:
                candidates = groups.get(service_class, [])
                if not candidates:
                    continue
                quota = self.service_class_weights.get(service_class, 1)
                if quota <= 0:
                    continue
                picked = self._take_candidates(
                    candidates=candidates,
                    take=min(quota, limit - len(selected), len(candidates)),
                    intra_class_rotate=intra_class_rotate,
                )
                if not picked:
                    continue
                selected.extend(picked)
                groups[service_class] = [rid for rid in groups[service_class] if rid not in set(picked)]
                progressed = True
                if len(selected) >= limit:
                    break
            if not progressed:
                break
            cursor = (cursor + 1) % len(service_classes)
            rotated_classes = self._rotate_candidates(service_classes, cursor)
        setattr(
            self,
            cursor_attr,
            (getattr(self, cursor_attr) + 1) % max(1, len(service_classes)),
        )
        return selected

    def _take_candidates(
        self,
        *,
        candidates: list[str],
        take: int,
        intra_class_rotate: bool,
    ) -> list[str]:
        if take <= 0 or not candidates:
            return []
        selected_candidates = candidates
        if intra_class_rotate and len(candidates) > 1:
            selected_candidates = self._rotate_candidates(
                candidates,
                self._decode_rr_cursor,
            )
        picked = selected_candidates[:take]
        if intra_class_rotate and picked:
            self._decode_rr_cursor = (
                self._decode_rr_cursor + len(picked)
            ) % max(1, len(candidates))
        return picked

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
            service_class = str(request.get("service_class") or "latency")
            lora_id = self._lora_adapter_key(request)
            queue_wait_s = max(0.0, now - float(request.get("queued_at") or now))
            counts[service_class] = counts.get(service_class, 0) + 1
            lora_counts[lora_id] = lora_counts.get(lora_id, 0) + 1
            if self._is_multimodal(request):
                multimodal_waits.append(queue_wait_s)
                if self._is_multimodal_lora(request):
                    multimodal_lora_requests += 1
            wait_sums[service_class] = wait_sums.get(service_class, 0.0) + queue_wait_s
            wait_max[service_class] = max(wait_max.get(service_class, 0.0), queue_wait_s)
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
                key: self._percentile(values, 0.95) for key, values in wait_values.items()
            },
        }

    @staticmethod
    def _normalize_quotas(quotas: dict[str, int]) -> dict[str, int]:
        return {
            key: max(0, int(value))
            for key, value in quotas.items()
        }

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
        per_class_p95 = (
            queued_metrics.get("service_class_p95_wait_s", {}) or {}
        )
        return any(
            float(per_class_p95.get(service_class, 0.0) or 0.0) >= threshold
            for service_class in self.fairness_guardrail_service_classes
        )

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        sorted_values = sorted(float(v) for v in values)
        idx = int(round((len(sorted_values) - 1) * max(0.0, min(1.0, q))))
        return sorted_values[idx]

    @staticmethod
    def _count_service_classes(
        scheduler,
        request_ids: list[str],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rid in request_ids:
            service_class = str(scheduler.get_request(rid).get("service_class") or "latency")
            counts[service_class] = counts.get(service_class, 0) + 1
        return counts

    @staticmethod
    def _count_multimodal_requests(
        scheduler,
        request_ids: list[str],
    ) -> int:
        return sum(
            1 for rid in request_ids if StepScheduler._is_multimodal(scheduler.get_request(rid))
        )

    @staticmethod
    def _count_multimodal_lora_requests(
        scheduler,
        request_ids: list[str],
    ) -> int:
        return sum(
            1
            for rid in request_ids
            if StepScheduler._is_multimodal_lora(scheduler.get_request(rid))
        )

    @staticmethod
    def _count_lora_adapters(
        scheduler,
        request_ids: list[str],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rid in request_ids:
            adapter = StepScheduler._lora_adapter_key(scheduler.get_request(rid))
            counts[adapter] = counts.get(adapter, 0) + 1
        return counts

    @staticmethod
    def _is_multimodal(request) -> bool:
        return bool(
            request.get("is_multimodal")
            or (request.get("multi_modal_data") or {}).get("image")
        )

    @classmethod
    def _is_multimodal_lora(cls, request) -> bool:
        return cls._is_multimodal(request) and bool(request.get("lora_id"))

    @classmethod
    def _is_multimodal_request(cls, scheduler, request_id: str) -> bool:
        return cls._is_multimodal(scheduler.get_request(request_id))

    @classmethod
    def _is_multimodal_lora_request(cls, scheduler, request_id: str) -> bool:
        return cls._is_multimodal_lora(scheduler.get_request(request_id))

    @staticmethod
    def _normalized_share_map(counts: dict[str, int]) -> dict[str, float]:
        if not counts:
            return {}
        normalized = {key: float(value) for key, value in counts.items()}
        total = sum(normalized.values())
        if total <= 0:
            return {key: 0.0 for key in normalized}
        return {key: value / total for key, value in normalized.items()}

    @staticmethod
    def _share_gap_map(
        target_share: dict[str, float],
        baseline_share: dict[str, float],
    ) -> dict[str, float]:
        keys = sorted(set(target_share) | set(baseline_share))
        return {
            key: float(target_share.get(key, 0.0) or 0.0)
            - float(baseline_share.get(key, 0.0) or 0.0)
            for key in keys
        }

    @staticmethod
    def _max_abs_share_gap(gaps: dict[str, float]) -> float:
        return max((abs(float(value or 0.0)) for value in gaps.values()), default=0.0)

    @classmethod
    def _lora_adapter_key(cls, request) -> str:
        lora_id = request.get("lora_id")
        if not lora_id:
            return cls.BASE_LORA_ADAPTER
        return str(lora_id)

    @staticmethod
    def _service_class_priority(service_class: object) -> int:
        priorities = {
            "latency": 0,
            "interactive": 0,
            "balanced": 1,
            "throughput": 2,
            "background": 3,
        }
        return priorities.get(str(service_class or "latency"), 1)

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
