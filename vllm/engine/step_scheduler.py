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
        self._decode_only_streak = 0
        self._decode_rr_cursor = 0
        self._prefill_rr_cursor = 0
        self._prefill_deferrals: dict[str, int] = {}
        self._admission_service_cursor = 0
        self._decode_service_cursor = 0

    def build_plan(self, scheduler) -> StepPlan:
        queued_metrics = self._compute_queued_metrics(scheduler)
        admissions, aged_admission_count, admitted_service_classes = self._build_admission_plan(
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

        prefill_plan = self._build_prefill_plan(scheduler, prefills, prefill_budget)
        prefill_service_classes = self._count_service_classes(scheduler, prefill_plan.request_ids if prefill_plan else [])
        decode_plan = self._build_decode_plan(
            scheduler,
            decodes,
            decode_limit,
            prefill_plan is None,
        )
        decode_service_classes = self._count_service_classes(scheduler, decode_plan.request_ids if decode_plan else [])
        plan = StepPlan(
            admissions=admissions,
            prefills=prefill_plan,
            decodes=decode_plan,
            step_token_budget=self.step_token_budget,
            queued_before=scheduler.queued_request_count,
            running_before=scheduler.running_request_count,
            prefill_starvation_protected=bool(starvation_protected and prefill_plan is not None),
            aged_admission_count=aged_admission_count,
            admitted_service_classes=admitted_service_classes,
            prefill_service_classes=prefill_service_classes,
            decode_service_classes=decode_service_classes,
            queued_service_classes=queued_metrics["service_classes"],
            queued_avg_wait_s=queued_metrics["avg_wait_s"],
            queued_max_wait_s=queued_metrics["max_wait_s"],
            queued_p95_wait_s=queued_metrics["p95_wait_s"],
            queued_service_class_avg_wait_s=queued_metrics["service_class_avg_wait_s"],
            queued_service_class_max_wait_s=queued_metrics["service_class_max_wait_s"],
            queued_service_class_p95_wait_s=queued_metrics["service_class_p95_wait_s"],
            fairness_guardrail_triggered=fairness_guardrail_triggered,
        )
        self._update_prefill_deferrals(prefills, decodes, prefill_plan)
        self._update_decode_streak(plan)
        return plan

    def _build_admission_plan(
        self,
        scheduler,
    ) -> tuple[AdmissionPlan | None, int, dict[str, int]]:
        if not scheduler.has_capacity() or scheduler.queued_request_count == 0:
            return None, 0, {}
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
        if not request_ids:
            return None, 0, {}
        aged_admission_count = sum(1 for rid in request_ids if rid in set(aged_ids))
        return (
            AdmissionPlan(request_ids=request_ids),
            aged_admission_count,
            self._count_service_classes(scheduler, request_ids),
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

    def _build_prefill_plan(self, scheduler, prefills: list[str], token_budget: int) -> PrefillPlan | None:
        if not prefills or token_budget <= 0:
            return None
        base_processed_len = scheduler.get_request(prefills[0])["seq_len"]
        candidate_prefills = [
            rid for rid in prefills if scheduler.get_request(rid)["seq_len"] == base_processed_len
        ]
        if not candidate_prefills:
            return None
        request_ids = self._rotate_candidates(
            candidate_prefills,
            self._prefill_rr_cursor,
        )[: min(self.prefill_microbatch_size, len(candidate_prefills))]
        self._prefill_rr_cursor = (
            self._prefill_rr_cursor + len(request_ids)
        ) % max(1, len(candidate_prefills))
        if not request_ids:
            return None
        min_remaining = min(
            len(scheduler.get_request(rid)["input_ids"]) - scheduler.get_request(rid)["seq_len"]
            for rid in request_ids
        )
        per_req_budget = max(1, token_budget // max(1, len(request_ids)))
        chunk_len = min(min_remaining, self.prefill_chunk_size, per_req_budget)
        if chunk_len <= 0:
            chunk_len = 1
        return PrefillPlan(
            request_ids=request_ids,
            chunk_len=chunk_len,
            token_budget=token_budget,
        )

    def _build_decode_plan(
        self,
        scheduler,
        decodes: list[str],
        decode_limit: int,
        no_prefills_selected: bool,
    ) -> DecodePlan | None:
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
        if not request_ids:
            return None
        use_fast_path = bool(no_prefills_selected and len(request_ids) == len(decodes))
        return DecodePlan(
            request_ids=request_ids,
            token_budget=decode_limit,
            use_fast_path=use_fast_path,
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
                "avg_wait_s": 0.0,
                "max_wait_s": 0.0,
                "p95_wait_s": 0.0,
                "service_class_avg_wait_s": {},
                "service_class_max_wait_s": {},
                "service_class_p95_wait_s": {},
            }
        counts: dict[str, int] = {}
        wait_sums: dict[str, float] = {}
        wait_max: dict[str, float] = {}
        wait_values: dict[str, list[float]] = {}
        all_waits: list[float] = []
        total_wait = 0.0
        max_wait = 0.0
        for rid in queued_ids:
            request = scheduler.get_request(rid)
            service_class = str(request.get("service_class") or "latency")
            queue_wait_s = max(0.0, now - float(request.get("queued_at") or now))
            counts[service_class] = counts.get(service_class, 0) + 1
            wait_sums[service_class] = wait_sums.get(service_class, 0.0) + queue_wait_s
            wait_max[service_class] = max(wait_max.get(service_class, 0.0), queue_wait_s)
            wait_values.setdefault(service_class, []).append(queue_wait_s)
            total_wait += queue_wait_s
            max_wait = max(max_wait, queue_wait_s)
            all_waits.append(queue_wait_s)
        return {
            "service_classes": counts,
            "avg_wait_s": total_wait / max(1, len(queued_ids)),
            "max_wait_s": max_wait,
            "p95_wait_s": self._percentile(all_waits, 0.95),
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
