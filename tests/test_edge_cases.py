# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import pytest

from vllm.engine.request_scheduler import RequestScheduler
from vllm.engine.request_state import RequestState
from vllm.engine.step_scheduler import StepScheduler
from vllm.sampling_params import SamplingParams


def _scheduler_with_requests(requests: list[dict[str, Any]]) -> RequestScheduler:
    scheduler = RequestScheduler(max_active_requests=max(1, len(requests)))
    for i, request in enumerate(requests):
        state = RequestState(
            request_id=f"r{i}",
            prompt="",
            input_ids=request.get("input_ids", [1, 2, 3, 4]),
            sampling_params=request.get("sampling_params") or SamplingParams(),
            slot_idx=i,
            is_prefill=request["is_prefill"],
            seq_len=request.get("seq_len", 0),
            generated_ids=request.get("generated_ids", [10]),
            lora_id=request.get("lora_id"),
            is_multimodal=request.get("is_multimodal", False),
            multi_modal_data={"image": [{"image": "file:///tmp/demo.png"}]}
            if request.get("is_multimodal", False)
            else None,
        )
        scheduler.add_request(f"r{i}", state)
    return scheduler


def test_zero_token_prefill() -> None:
    """A request with 0 prompt tokens must not crash the scheduler."""
    scheduler = _scheduler_with_requests(
        [{"is_prefill": True, "seq_len": 0, "input_ids": []}]
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
    assert plan is not None


def test_zero_token_prefill_with_decode_present() -> None:
    """0-token prefill coexisting with active decodes must not crash."""
    scheduler = _scheduler_with_requests(
        [
            {"is_prefill": True, "seq_len": 0, "input_ids": []},
            {"is_prefill": False, "seq_len": 4},
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
    assert plan is not None


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_block_boundary_prefill(offset: int) -> None:
    """Token counts at block_size boundaries must not cause misalignment."""
    block_size = 16
    seq_len = block_size + offset
    if seq_len < 0:
        pytest.skip(f"seq_len={seq_len} is negative")
    scheduler = _scheduler_with_requests(
        [
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=max(16, seq_len * 2),
        decode_priority_enabled=True,
        prefill_chunk_size=seq_len if seq_len > 0 else 8,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=1,
    )
    plan = step_scheduler.build_plan(scheduler)
    assert plan is not None


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_block_boundary_decode(offset: int) -> None:
    """Decode seq_lens at block_size boundaries must not crash."""
    block_size = 16
    seq_len = block_size + offset
    if seq_len <= 0:
        pytest.skip(f"seq_len={seq_len} is non-positive")
    scheduler = _scheduler_with_requests([{"is_prefill": False, "seq_len": seq_len}])
    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=8,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=1,
    )
    plan = step_scheduler.build_plan(scheduler)
    assert plan is not None
    assert plan.decodes is not None


@pytest.mark.parametrize("num_prefills", [0, 1, 2, 4])
def test_all_decode_batch(num_prefills: int) -> None:
    """When only decodes are active, decode fast path must be selected."""
    requests = []
    for _ in range(num_prefills):
        requests.append({"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3]})
    for i in range(4):
        requests.append({"is_prefill": False, "seq_len": 4 + i})
    scheduler = _scheduler_with_requests(requests)
    step_scheduler = StepScheduler(
        step_token_budget=16 if num_prefills > 0 else 8,
        decode_priority_enabled=True,
        prefill_chunk_size=8,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=1,
    )
    plan = step_scheduler.build_plan(scheduler)
    assert plan is not None
