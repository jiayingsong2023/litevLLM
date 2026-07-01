# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.engine.planners.types import BudgetResult


class BudgetComputer:
    """Compute per-step token budgets for prefills and decodes."""

    def __init__(
        self,
        *,
        step_token_budget: int,
        prefill_chunk_size: int,
        decode_priority_enabled: bool,
        prefill_reserved_tokens: int,
        prefill_reserve_backlog: int,
        prefill_catchup_ratio: float,
    ) -> None:
        self.step_token_budget = max(1, int(step_token_budget))
        self.prefill_chunk_size = max(1, int(prefill_chunk_size))
        self.decode_priority_enabled = bool(decode_priority_enabled)
        self.prefill_reserved_tokens = max(0, int(prefill_reserved_tokens))
        self.prefill_reserve_backlog = max(1, int(prefill_reserve_backlog))
        self.prefill_catchup_ratio = min(1.0, max(0.0, float(prefill_catchup_ratio)))

    def compute(
        self,
        *,
        num_prefills: int,
        num_decodes: int,
        starvation_protected: bool,
    ) -> BudgetResult:
        if self.decode_priority_enabled:
            prefill_budget = 0
            if num_prefills:
                backlog_factor = (
                    min(2.0, float(num_prefills) / float(self.prefill_reserve_backlog))
                    if self.prefill_reserve_backlog > 0
                    else 1.0
                )
                adjusted_catchup_ratio = min(
                    0.9, self.prefill_catchup_ratio * backlog_factor
                )
                if num_prefills >= self.prefill_reserve_backlog:
                    adjusted_catchup_ratio = max(adjusted_catchup_ratio, 0.35)
                else:
                    adjusted_catchup_ratio = max(
                        adjusted_catchup_ratio, self.prefill_catchup_ratio
                    )
            else:
                adjusted_catchup_ratio = self.prefill_catchup_ratio

            if starvation_protected and num_prefills:
                reserve_tokens = max(
                    1,
                    self.prefill_reserved_tokens,
                    int(self.step_token_budget * max(adjusted_catchup_ratio, 0.25)),
                )
                prefill_budget = min(self.step_token_budget, reserve_tokens)
            elif num_prefills and not num_decodes:
                prefill_budget = min(self.prefill_chunk_size, self.step_token_budget)
            elif num_prefills and num_prefills >= self.prefill_reserve_backlog:
                reserve_tokens = max(
                    self.prefill_reserved_tokens,
                    int(self.step_token_budget * adjusted_catchup_ratio),
                )
                prefill_budget = min(self.step_token_budget, max(1, reserve_tokens))
            decode_limit = min(
                num_decodes, max(0, self.step_token_budget - prefill_budget)
            )
        else:
            if num_prefills:
                reserve_tokens = max(1, self.prefill_reserved_tokens or 1)
                decode_limit = max(
                    0, min(num_decodes, self.step_token_budget - reserve_tokens)
                )
            else:
                decode_limit = min(num_decodes, self.step_token_budget)
            prefill_budget = max(0, self.step_token_budget - decode_limit)
        return BudgetResult(prefill_budget=prefill_budget, decode_limit=decode_limit)
