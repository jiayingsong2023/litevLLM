from __future__ import annotations

from types import SimpleNamespace

from vllm.engine.step_scheduler import StepScheduler


def _request(*, prefill: bool, seq_len: int = 0, prompt_len: int = 8):
    return SimpleNamespace(
        is_prefill=prefill, seq_len=seq_len, input_ids=list(range(prompt_len))
    )


class _Scheduler:
    def __init__(self, requests, queued_ids=(), running_ids=(), slots=8):
        self.requests = requests
        self.queued_ids = list(queued_ids)
        self.running_ids = list(running_ids)
        self.available_slots = slots

    @property
    def queued_request_count(self):
        return len(self.queued_ids)

    @property
    def running_request_count(self):
        return len(self.running_ids)

    def has_capacity(self):
        return self.available_slots > 0

    def get_request(self, request_id):
        return self.requests[request_id]

    def classify_requests(self):
        return (
            [rid for rid in self.running_ids if self.requests[rid].is_prefill],
            [rid for rid in self.running_ids if not self.requests[rid].is_prefill],
        )


def _planner(**overrides):
    return StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=2,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        **overrides,
    )


def test_admission_is_fifo():
    scheduler = _Scheduler(
        {"old": _request(prefill=True), "new": _request(prefill=True)},
        queued_ids=("old", "new"),
        slots=1,
    )
    assert _planner(max_admit_per_step=2).build_plan(
        scheduler
    ).admissions.request_ids == ["old"]


def test_single_decode_uses_fast_path():
    scheduler = _Scheduler(
        {"r": _request(prefill=False, seq_len=8)}, running_ids=("r",)
    )
    plan = _planner().build_plan(scheduler)
    assert plan.decodes and plan.decodes.use_fast_path


def test_starvation_protects_prefill_after_decode_streak():
    requests = {"p": _request(prefill=True), "d": _request(prefill=False, seq_len=8)}
    scheduler = _Scheduler(requests, running_ids=("p", "d"))
    planner = _planner(max_decode_streak=1)
    planner._decode_only_streak = 1
    plan = planner.build_plan(scheduler)
    assert plan.metrics and plan.metrics.prefill_starvation_protected
