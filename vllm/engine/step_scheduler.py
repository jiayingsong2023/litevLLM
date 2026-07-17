# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.engine.planners import AdmissionPlanner, BudgetComputer, DecodePrefillPlanner
from vllm.engine.step_plan import DecodePlan, PrefillPlan, StepPlan, StepPlanMetrics


class StepScheduler:
    """Single-GPU FIFO scheduler with decode priority and starvation protection."""

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
        max_prefill_deferrals: int = 2,
    ) -> None:
        self.step_token_budget = max(1, int(step_token_budget))
        self.prefill_chunk_size = max(1, int(prefill_chunk_size))
        self.max_decode_streak = max(1, int(max_decode_streak))
        self.max_prefill_deferrals = max(1, int(max_prefill_deferrals))
        self._decode_only_streak = 0
        self._prefill_deferrals: dict[str, int] = {}
        self._admission_planner = AdmissionPlanner(
            max_admit_per_step=max_admit_per_step
        )
        self._budget_computer = BudgetComputer(
            step_token_budget=self.step_token_budget,
            prefill_chunk_size=self.prefill_chunk_size,
            decode_priority_enabled=decode_priority_enabled,
            prefill_reserved_tokens=prefill_reserved_tokens,
            prefill_reserve_backlog=prefill_reserve_backlog,
            prefill_catchup_ratio=prefill_catchup_ratio,
        )
        self._decode_prefill_planner = DecodePrefillPlanner(
            prefill_chunk_size=self.prefill_chunk_size,
            prefill_microbatch_size=prefill_microbatch_size,
        )

    def set_verified_decode_batch_sizes(
        self, batch_sizes: tuple[int, ...] | None
    ) -> None:
        self._decode_prefill_planner.set_verified_decode_batch_sizes(batch_sizes)

    def build_plan(self, scheduler) -> StepPlan:
        fast_plan = self._build_single_request_fast_path(scheduler)
        if fast_plan is not None:
            self._update_prefill_deferrals(
                fast_plan.prefills.request_ids if fast_plan.prefills else [],
                fast_plan.decodes.request_ids if fast_plan.decodes else [],
                fast_plan.prefills,
            )
            self._update_decode_streak(fast_plan)
            return fast_plan

        admissions = self._admission_planner.plan(scheduler).plan
        prefills, decodes = scheduler.classify_requests()
        starvation_protected = self._should_protect_prefills(prefills, decodes)
        budget = self._budget_computer.compute(
            num_prefills=len(prefills),
            num_decodes=len(decodes),
            starvation_protected=starvation_protected,
        )
        prefill_plan = self._decode_prefill_planner.build_prefill_plan(
            scheduler, prefills, budget.prefill_budget
        ).plan
        decode_plan = self._decode_prefill_planner.build_decode_plan(
            scheduler, decodes, budget.decode_limit, prefill_plan is None
        ).plan
        plan = StepPlan(
            admissions=admissions,
            prefills=prefill_plan,
            decodes=decode_plan,
            step_token_budget=self.step_token_budget,
            metrics=StepPlanMetrics(
                queued_before=scheduler.queued_request_count,
                running_before=scheduler.running_request_count,
                prefill_starvation_protected=bool(
                    starvation_protected and prefill_plan is not None
                ),
            ),
        )
        self._update_prefill_deferrals(prefills, decodes, prefill_plan)
        self._update_decode_streak(plan)
        return plan

    def _build_single_request_fast_path(self, scheduler) -> StepPlan | None:
        if scheduler.queued_request_count != 0 or scheduler.running_request_count != 1:
            return None
        rid = scheduler.running_ids[0]
        req = scheduler.get_request(rid)
        if req.is_prefill:
            remaining = max(1, len(req.input_ids) - req.seq_len)
            return StepPlan(
                admissions=None,
                prefills=PrefillPlan(
                    request_ids=[rid],
                    chunk_len=min(
                        remaining, self.prefill_chunk_size, self.step_token_budget
                    ),
                    token_budget=self.step_token_budget,
                ),
                decodes=None,
                step_token_budget=self.step_token_budget,
                metrics=StepPlanMetrics(),
            )
        return StepPlan(
            admissions=None,
            prefills=None,
            decodes=DecodePlan(request_ids=[rid], token_budget=1, use_fast_path=True),
            step_token_budget=self.step_token_budget,
            metrics=StepPlanMetrics(),
        )

    def _should_protect_prefills(self, prefills: list[str], decodes: list[str]) -> bool:
        return bool(prefills and decodes) and (
            self._decode_only_streak >= self.max_decode_streak
            or any(
                self._prefill_deferrals.get(rid, 0) >= self.max_prefill_deferrals
                for rid in prefills
            )
        )

    def _update_decode_streak(self, plan: StepPlan) -> None:
        if plan.prefills is not None:
            self._decode_only_streak = 0
        elif plan.decodes is not None:
            self._decode_only_streak += 1
        else:
            self._decode_only_streak = 0

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
        selected = set(prefill_plan.request_ids) if prefill_plan else set()
        for rid in selected:
            self._prefill_deferrals[rid] = 0
        if decodes:
            for rid in current_prefills - selected:
                self._prefill_deferrals[rid] = self._prefill_deferrals.get(rid, 0) + 1
