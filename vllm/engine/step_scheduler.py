# SPDX-License-Identifier: Apache-2.0
from vllm.engine.step_plan import DecodePlan, PrefillPlan, StepPlan


class StepScheduler:
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
    ) -> None:
        self.step_token_budget = max(1, int(step_token_budget))
        self.decode_priority_enabled = decode_priority_enabled
        self.prefill_chunk_size = max(1, int(prefill_chunk_size))
        self.prefill_reserved_tokens = max(0, int(prefill_reserved_tokens))
        self.prefill_reserve_backlog = max(1, int(prefill_reserve_backlog))
        self.prefill_catchup_ratio = min(1.0, max(0.0, float(prefill_catchup_ratio)))
        self.prefill_microbatch_size = min(4, max(1, int(prefill_microbatch_size)))

    def build_plan(self, scheduler) -> StepPlan:
        prefills, decodes = scheduler.classify_requests()
        prefill_budget, decode_limit = self._compute_budgets(
            num_prefills=len(prefills),
            num_decodes=len(decodes),
        )

        prefill_plan = self._build_prefill_plan(scheduler, prefills, prefill_budget)
        decode_plan = self._build_decode_plan(decodes, decode_limit, prefill_plan is None)
        return StepPlan(
            prefills=prefill_plan,
            decodes=decode_plan,
            step_token_budget=self.step_token_budget,
        )

    def _compute_budgets(self, *, num_prefills: int, num_decodes: int) -> tuple[int, int]:
        if self.decode_priority_enabled:
            prefill_budget = 0
            if num_prefills and not num_decodes:
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

    def _build_prefill_plan(self, scheduler, prefills: list[str], token_budget: int) -> PrefillPlan | None:
        if not prefills or token_budget <= 0:
            return None
        base_processed_len = scheduler.get_request(prefills[0])["seq_len"]
        candidate_prefills = [
            rid for rid in prefills if scheduler.get_request(rid)["seq_len"] == base_processed_len
        ]
        request_ids = candidate_prefills[: min(self.prefill_microbatch_size, len(candidate_prefills))]
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
        decodes: list[str],
        decode_limit: int,
        no_prefills_selected: bool,
    ) -> DecodePlan | None:
        request_ids = decodes[:decode_limit]
        if not request_ids:
            return None
        use_fast_path = bool(no_prefills_selected and len(request_ids) == len(decodes))
        return DecodePlan(
            request_ids=request_ids,
            token_budget=decode_limit,
            use_fast_path=use_fast_path,
        )
