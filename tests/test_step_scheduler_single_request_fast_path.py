# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

from vllm.engine.step_scheduler import StepScheduler


@dataclass
class _MockScheduler:
    request: dict
    _queued: int = 0

    @property
    def queued_request_count(self) -> int:
        return self._queued

    @property
    def running_request_count(self) -> int:
        return 1

    @property
    def queued_ids(self) -> list[str]:
        return ["q0"] if self._queued else []

    @property
    def running_ids(self) -> list[str]:
        return ["r0"]

    @property
    def available_slots(self) -> int:
        return 1

    def has_capacity(self) -> bool:
        return True

    def classify_requests(self) -> tuple[list[str], list[str]]:
        if bool(self.request.get("is_prefill", False)):
            return ["r0"], []
        return [], ["r0"]

    def get_request(self, request_id: str) -> dict:
        assert request_id in {"r0", "q0"}
        return self.request


def _make_scheduler() -> StepScheduler:
    return StepScheduler(
        step_token_budget=128,
        decode_priority_enabled=True,
        prefill_chunk_size=32,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
    )


def test_single_request_fast_path_prefill_uses_chunk_cap() -> None:
    scheduler = _make_scheduler()
    mock = _MockScheduler(
        request={
            "is_prefill": True,
            "input_ids": list(range(80)),
            "seq_len": 10,
        }
    )
    plan = scheduler.build_plan(mock)
    assert plan.prefills is not None
    assert plan.decodes is None
    assert plan.prefills.request_ids == ["r0"]
    # remaining=70, chunk cap=32
    assert plan.prefills.chunk_len == 32


def test_single_request_fast_path_decode_uses_fast_decode() -> None:
    scheduler = _make_scheduler()
    mock = _MockScheduler(
        request={
            "is_prefill": False,
            "input_ids": list(range(32)),
            "seq_len": 32,
        }
    )
    plan = scheduler.build_plan(mock)
    assert plan.prefills is None
    assert plan.decodes is not None
    assert plan.decodes.request_ids == ["r0"]
    assert plan.decodes.token_budget == 1
    assert plan.decodes.use_fast_path is True


def test_single_request_fast_path_disabled_when_queue_not_empty() -> None:
    scheduler = _make_scheduler()
    mock = _MockScheduler(
        request={
            "is_prefill": False,
            "input_ids": list(range(32)),
            "seq_len": 32,
        },
        _queued=1,
    )
    plan = scheduler.build_plan(mock)
    assert plan.queued_before == 1
