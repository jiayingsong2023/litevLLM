# SPDX-License-Identifier: Apache-2.0
import time

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
    assert plan.admissions is None
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
    assert plan.admissions is None
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
    assert plan.admissions is None
    assert plan.prefills is not None
    assert plan.prefills.chunk_len == 3
    assert plan.decodes is None


def test_step_scheduler_ignores_queued_requests_until_admitted() -> None:
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.add_request(
        "r0",
        {
            "slot_idx": 0,
            "is_prefill": False,
            "seq_len": 4,
            "generated_ids": [10],
            "input_ids": [1, 2, 3, 4],
        },
    )
    scheduler.enqueue_request(
        "r1",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2, 3, 4, 5, 6],
        },
    )

    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
    )
    plan = step_scheduler.build_plan(scheduler)

    assert plan.admissions is None
    assert plan.prefills is None
    assert plan.decodes is not None
    assert plan.decodes.request_ids == ["r0"]


def test_step_scheduler_limits_admissions_per_step() -> None:
    scheduler = RequestScheduler(max_active_requests=2)
    scheduler.enqueue_request("r0", {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3]})
    scheduler.enqueue_request("r1", {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3]})
    scheduler.enqueue_request("r2", {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3]})

    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        max_admit_per_step=1,
    )
    plan = step_scheduler.build_plan(scheduler)

    assert plan.admissions is not None
    assert plan.admissions.request_ids == ["r0"]
    assert plan.queued_before == 3
    assert plan.running_before == 0


def test_step_scheduler_prefers_shorter_queued_requests_for_admission() -> None:
    scheduler = RequestScheduler(max_active_requests=2)
    scheduler.enqueue_request(
        "long",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3, 4], "queued_at": 1.0},
    )
    scheduler.enqueue_request(
        "short",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2], "queued_at": 2.0},
    )

    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        max_admit_per_step=1,
    )
    plan = step_scheduler.build_plan(scheduler)

    assert plan.admissions is not None
    assert plan.admissions.request_ids == ["short"]


def test_step_scheduler_service_class_can_override_shorter_prompt() -> None:
    scheduler = RequestScheduler(max_active_requests=2)
    scheduler.enqueue_request(
        "throughput-short",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2],
            "queued_at": 1.0,
            "service_class": "throughput",
        },
    )
    scheduler.enqueue_request(
        "latency-long",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2, 3, 4],
            "queued_at": 1.1,
            "service_class": "latency",
        },
    )

    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        max_admit_per_step=1,
        queue_aging_threshold_s=10.0,
    )
    plan = step_scheduler.build_plan(scheduler)

    assert plan.admissions is not None
    assert plan.admissions.request_ids == ["latency-long"]


def test_step_scheduler_aging_can_override_short_prompt_preference(monkeypatch) -> None:
    scheduler = RequestScheduler(max_active_requests=2)
    scheduler.enqueue_request(
        "old-long",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3, 4, 5], "queued_at": 1.0},
    )
    scheduler.enqueue_request(
        "new-short",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2], "queued_at": 9.5},
    )
    monkeypatch.setattr(time, "perf_counter", lambda: 10.0)

    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        max_admit_per_step=1,
        queue_aging_threshold_s=2.0,
    )
    plan = step_scheduler.build_plan(scheduler)

    assert plan.admissions is not None
    assert plan.admissions.request_ids == ["old-long"]


def test_step_scheduler_starvation_protects_prefill_after_decode_streak() -> None:
    scheduler = _scheduler_with_requests(
        [
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3, 4, 5, 6]},
            {"is_prefill": False, "seq_len": 4},
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=8,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=99,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=1,
        max_decode_streak=2,
    )

    plan1 = step_scheduler.build_plan(scheduler)
    plan2 = step_scheduler.build_plan(scheduler)
    plan3 = step_scheduler.build_plan(scheduler)

    assert plan1.prefills is None
    assert plan2.prefills is None
    assert plan3.prefills is not None
    assert plan3.prefill_starvation_protected is True


def test_step_scheduler_decode_round_robin_when_budget_is_limited() -> None:
    scheduler = _scheduler_with_requests(
        [
            {"is_prefill": False, "seq_len": 4},
            {"is_prefill": False, "seq_len": 5},
            {"is_prefill": False, "seq_len": 6},
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=2,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=1,
    )

    plan1 = step_scheduler.build_plan(scheduler)
    plan2 = step_scheduler.build_plan(scheduler)

    assert plan1.decodes is not None
    assert plan2.decodes is not None
    assert plan1.decodes.request_ids == ["r0", "r1"]
    assert plan2.decodes.request_ids == ["r2", "r0"]


def test_step_scheduler_prefill_round_robin_and_deferral_protection() -> None:
    scheduler = _scheduler_with_requests(
        [
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3, 4, 5]},
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3, 4, 5]},
            {"is_prefill": False, "seq_len": 4},
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=99,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=1,
        max_decode_streak=99,
        max_prefill_deferrals=2,
    )

    plan1 = step_scheduler.build_plan(scheduler)
    plan2 = step_scheduler.build_plan(scheduler)
    plan3 = step_scheduler.build_plan(scheduler)

    assert plan1.prefills is None
    assert plan2.prefills is None
    assert plan3.prefills is not None
    assert plan3.prefill_starvation_protected is True


def test_step_scheduler_prefill_round_robin_selects_different_request_ids() -> None:
    scheduler = _scheduler_with_requests(
        [
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3, 4, 5, 6]},
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3, 4, 5, 6]},
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=1.0,
        prefill_microbatch_size=1,
    )

    plan1 = step_scheduler.build_plan(scheduler)
    plan2 = step_scheduler.build_plan(scheduler)

    assert plan1.prefills is not None
    assert plan2.prefills is not None
    assert plan1.prefills.request_ids == ["r0"]
    assert plan2.prefills.request_ids == ["r1"]


def test_step_scheduler_admission_uses_service_class_weights_across_step_limit() -> None:
    scheduler = RequestScheduler(max_active_requests=2)
    scheduler.enqueue_request(
        "latency-0",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2], "service_class": "latency"},
    )
    scheduler.enqueue_request(
        "latency-1",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3], "service_class": "latency"},
    )
    scheduler.enqueue_request(
        "background-0",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1], "service_class": "background"},
    )

    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        max_admit_per_step=2,
        service_class_weights={"latency": 1, "background": 1},
    )

    plan = step_scheduler.build_plan(scheduler)

    assert plan.admissions is not None
    assert set(plan.admissions.request_ids) == {"latency-0", "background-0"}
    assert plan.admitted_service_classes == {"latency": 1, "background": 1}


def test_step_scheduler_decode_uses_weighted_service_class_fairness() -> None:
    scheduler = RequestScheduler(max_active_requests=4)
    scheduler.add_request(
        "latency-0",
        {"slot_idx": 0, "is_prefill": False, "seq_len": 4, "generated_ids": [1], "input_ids": [1, 2], "service_class": "latency"},
    )
    scheduler.add_request(
        "background-0",
        {"slot_idx": 1, "is_prefill": False, "seq_len": 4, "generated_ids": [1], "input_ids": [1, 2], "service_class": "background"},
    )
    scheduler.add_request(
        "background-1",
        {"slot_idx": 2, "is_prefill": False, "seq_len": 4, "generated_ids": [1], "input_ids": [1, 2], "service_class": "background"},
    )

    step_scheduler = StepScheduler(
        step_token_budget=2,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=1,
        service_class_weights={"latency": 1, "background": 1},
    )

    plan1 = step_scheduler.build_plan(scheduler)
    plan2 = step_scheduler.build_plan(scheduler)

    assert plan1.decodes is not None
    assert plan2.decodes is not None
    assert set(plan1.decodes.request_ids) == {"latency-0", "background-0"}
    assert set(plan2.decodes.request_ids) == {"latency-0", "background-1"}
    assert plan1.decode_service_classes == {"latency": 1, "background": 1}


def test_step_scheduler_reports_aged_admissions() -> None:
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.enqueue_request(
        "old-latency",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3], "queued_at": 1.0, "service_class": "latency"},
    )

    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=1,
        queue_aging_threshold_s=1.0,
    )

    original_perf_counter = time.perf_counter
    try:
        time.perf_counter = lambda: 3.0  # type: ignore[assignment]
        plan = step_scheduler.build_plan(scheduler)
    finally:
        time.perf_counter = original_perf_counter  # type: ignore[assignment]

    assert plan.admissions is not None
    assert plan.aged_admission_count == 1


def test_step_scheduler_admission_service_class_quota_reserves_slots() -> None:
    scheduler = RequestScheduler(max_active_requests=3)
    scheduler.enqueue_request(
        "latency-0",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2],
            "service_class": "latency",
            "queued_at": 1.0,
        },
    )
    scheduler.enqueue_request(
        "latency-1",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2, 3],
            "service_class": "latency",
            "queued_at": 1.1,
        },
    )
    scheduler.enqueue_request(
        "background-0",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1],
            "service_class": "background",
            "queued_at": 1.2,
        },
    )

    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        max_admit_per_step=2,
        service_class_weights={"latency": 4, "background": 1},
        admission_service_class_quotas={"latency": 1, "background": 1},
        queue_aging_threshold_s=1_000_000.0,
    )

    plan = step_scheduler.build_plan(scheduler)

    assert plan.admissions is not None
    assert set(plan.admissions.request_ids) == {"latency-0", "background-0"}
    assert plan.admitted_service_classes == {"latency": 1, "background": 1}


def test_step_scheduler_decode_service_class_quota_reserves_budget() -> None:
    scheduler = RequestScheduler(max_active_requests=4)
    scheduler.add_request(
        "latency-0",
        {
            "slot_idx": 0,
            "is_prefill": False,
            "seq_len": 4,
            "generated_ids": [1],
            "input_ids": [1, 2],
            "service_class": "latency",
        },
    )
    scheduler.add_request(
        "latency-1",
        {
            "slot_idx": 1,
            "is_prefill": False,
            "seq_len": 4,
            "generated_ids": [1],
            "input_ids": [1, 2],
            "service_class": "latency",
        },
    )
    scheduler.add_request(
        "background-0",
        {
            "slot_idx": 2,
            "is_prefill": False,
            "seq_len": 4,
            "generated_ids": [1],
            "input_ids": [1, 2],
            "service_class": "background",
        },
    )

    step_scheduler = StepScheduler(
        step_token_budget=2,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=1,
        service_class_weights={"latency": 4, "background": 1},
        decode_service_class_quotas={"background": 1},
    )

    plan = step_scheduler.build_plan(scheduler)

    assert plan.decodes is not None
    assert "background-0" in plan.decodes.request_ids
    assert plan.decode_service_classes == {"latency": 1, "background": 1}


def test_step_scheduler_reports_queued_service_class_wait_metrics(monkeypatch) -> None:
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.enqueue_request(
        "latency-old",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2],
            "service_class": "latency",
            "queued_at": 2.0,
        },
    )
    scheduler.enqueue_request(
        "background-new",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1],
            "service_class": "background",
            "queued_at": 6.0,
        },
    )
    monkeypatch.setattr(time, "perf_counter", lambda: 10.0)

    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=1,
        max_admit_per_step=1,
    )

    plan = step_scheduler.build_plan(scheduler)

    assert plan.queued_service_classes == {"latency": 1, "background": 1}
    assert plan.queued_max_wait_s == 8.0
    assert plan.queued_avg_wait_s == 6.0
    assert plan.queued_p95_wait_s == 8.0
    assert plan.queued_service_class_avg_wait_s == {"latency": 8.0, "background": 4.0}
    assert plan.queued_service_class_max_wait_s == {"latency": 8.0, "background": 4.0}
    assert plan.queued_service_class_p95_wait_s == {"latency": 8.0, "background": 4.0}


def test_step_scheduler_fairness_guardrail_triggers_prefill_protection(monkeypatch) -> None:
    scheduler = RequestScheduler(max_active_requests=3)
    scheduler.add_request(
        "p0",
        {
            "slot_idx": 0,
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2, 3, 4, 5],
            "service_class": "background",
        },
    )
    scheduler.add_request(
        "d0",
        {
            "slot_idx": 1,
            "is_prefill": False,
            "seq_len": 4,
            "generated_ids": [1],
            "input_ids": [1, 2],
            "service_class": "latency",
        },
    )
    scheduler.enqueue_request(
        "q0",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2, 3],
            "service_class": "latency",
            "queued_at": 1.0,
        },
    )
    monkeypatch.setattr(time, "perf_counter", lambda: 10.0)

    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=99,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=1,
        fairness_guardrail_queue_wait_s=5.0,
        fairness_guardrail_service_classes={"latency"},
    )

    plan = step_scheduler.build_plan(scheduler)

    assert plan.fairness_guardrail_triggered is True
    assert plan.prefill_starvation_protected is True
    assert plan.prefills is not None
