# SPDX-License-Identifier: Apache-2.0
from vllm.engine.request_scheduler import RequestScheduler
from vllm.engine.step_scheduler import StepScheduler


def _scheduler_with_requests(requests: list[dict]) -> RequestScheduler:
    scheduler = RequestScheduler(max_active_requests=max(1, len(requests)))
    for i, request in enumerate(requests):
        state = {
            "slot_idx": i,
            "is_prefill": request["is_prefill"],
            "seq_len": request.get("seq_len", 0),
            "input_ids": request.get("input_ids", [1, 2, 3, 4]),
            "generated_ids": request.get("generated_ids", [10]),
            "sampling_params": request.get("sampling_params"),
        }
        scheduler.add_request(f"r{i}", state)
    return scheduler


def test_step_scheduler_decode_fast_path_when_only_decodes() -> None:
    scheduler = _scheduler_with_requests(
        [
            {"is_prefill": False, "seq_len": 4},
            {"is_prefill": False, "seq_len": 5},
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=16,
        decode_priority_enabled=True,
        prefill_chunk_size=8,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
    )
    plan = step_scheduler.build_plan(scheduler)
    assert plan.prefills is None
    assert plan.decodes is not None
    assert plan.decodes.use_fast_path is True
    assert plan.decodes.request_ids == ["r0", "r1"]


def test_step_scheduler_reserves_prefill_budget_on_backlog() -> None:
    scheduler = _scheduler_with_requests(
        [
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3, 4, 5, 6]},
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3, 4, 5, 6]},
            {"is_prefill": False, "seq_len": 4},
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=8,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
    )
    plan = step_scheduler.build_plan(scheduler)
    assert plan.prefills is not None
    assert plan.prefills.request_ids == ["r0", "r1"]
    assert plan.prefills.token_budget == 2
    assert plan.prefills.chunk_len == 1
    assert plan.decodes is not None
    assert plan.decodes.request_ids == ["r2"]


def test_step_scheduler_prefill_only_uses_chunk_limit() -> None:
    scheduler = _scheduler_with_requests(
        [
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3, 4, 5, 6, 7]},
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=16,
        decode_priority_enabled=True,
        prefill_chunk_size=3,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
    )
    plan = step_scheduler.build_plan(scheduler)
    assert plan.prefills is not None
    assert plan.prefills.chunk_len == 3
    assert plan.decodes is None
