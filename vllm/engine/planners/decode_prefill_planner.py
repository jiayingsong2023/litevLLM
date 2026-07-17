# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.engine.planners.types import DecodePlanResult, PrefillPlanResult
from vllm.engine.step_plan import DecodePlan, PrefillPlan


class DecodePrefillPlanner:
    """Build FIFO prefill and decode plans for one engine step."""

    def __init__(
        self, *, prefill_chunk_size: int, prefill_microbatch_size: int
    ) -> None:
        self.prefill_chunk_size = max(1, int(prefill_chunk_size))
        self.prefill_microbatch_size = max(1, int(prefill_microbatch_size))
        self._verified_decode_batch_sizes: tuple[int, ...] | None = None

    def set_verified_decode_batch_sizes(
        self, batch_sizes: tuple[int, ...] | None
    ) -> None:
        if batch_sizes is None:
            self._verified_decode_batch_sizes = None
            return
        normalized = tuple(sorted({int(size) for size in batch_sizes if int(size) > 0}))
        if not normalized or normalized[0] != 1:
            raise ValueError("verified decode batch sizes must include M=1")
        self._verified_decode_batch_sizes = normalized

    def build_prefill_plan(
        self, scheduler, prefills: list[str], token_budget: int
    ) -> PrefillPlanResult:
        if not prefills or token_budget <= 0:
            return PrefillPlanResult(plan=None)
        base_processed_len = scheduler.get_request(prefills[0]).seq_len
        request_ids = [
            rid
            for rid in prefills
            if scheduler.get_request(rid).seq_len == base_processed_len
        ][: self.prefill_microbatch_size]
        if not request_ids:
            return PrefillPlanResult(plan=None)
        min_remaining = min(
            len(scheduler.get_request(rid).input_ids)
            - scheduler.get_request(rid).seq_len
            for rid in request_ids
        )
        chunk_len = min(
            min_remaining,
            self.prefill_chunk_size,
            max(1, token_budget // len(request_ids)),
        )
        return PrefillPlanResult(
            plan=PrefillPlan(
                request_ids=request_ids,
                chunk_len=max(1, chunk_len),
                token_budget=token_budget,
            )
        )

    def build_decode_plan(
        self,
        scheduler,
        decodes: list[str],
        decode_limit: int,
        no_prefills_selected: bool,
    ) -> DecodePlanResult:
        del scheduler
        request_ids = decodes[: max(0, decode_limit)]
        if self._verified_decode_batch_sizes is not None and request_ids:
            request_ids = request_ids[
                : max(
                    size
                    for size in self._verified_decode_batch_sizes
                    if size <= len(request_ids)
                )
            ]
        if not request_ids:
            return DecodePlanResult(plan=None)
        return DecodePlanResult(
            plan=DecodePlan(
                request_ids=request_ids,
                token_budget=decode_limit,
                use_fast_path=bool(
                    no_prefills_selected and len(request_ids) == len(decodes)
                ),
            )
        )
