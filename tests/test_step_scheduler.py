# SPDX-License-Identifier: Apache-2.0
import time

from vllm.engine.inference_config import LiteInferenceConfig
from vllm.engine.request_scheduler import RequestScheduler
from vllm.engine.step_scheduler import (
    LoraSchedulingParams,
    MultiModalSchedulingParams,
    StepScheduler,
)


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
            "service_class": request.get("service_class", "latency"),
            "lora_id": request.get("lora_id"),
            "is_multimodal": request.get("is_multimodal", False),
            "multi_modal_data": (
                {"image": [{"image": "file:///tmp/demo.png"}]}
                if request.get("is_multimodal", False)
                else None
            ),
        }
        scheduler.add_request(f"r{i}", state)
    return scheduler


def test_step_scheduler_constructor_does_not_read_lite_inference_env(
    monkeypatch,
) -> None:
    assert not hasattr(LiteInferenceConfig, "from_env")

    step_scheduler = StepScheduler(
        step_token_budget=2,
        decode_priority_enabled=True,
        prefill_chunk_size=8,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        min_prefill_chunk_size=4,
        prefill_sla_ttft_ms=1500.0,
    )

    assert step_scheduler.min_prefill_chunk_size == 4
    assert step_scheduler.prefill_sla_ttft_ms == 1500.0


def test_step_scheduler_none_max_prefill_uses_planner_chunk_size() -> None:
    scheduler = _scheduler_with_requests(
        [
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": list(range(1024)),
            },
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=2048,
        decode_priority_enabled=True,
        prefill_chunk_size=512,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        max_prefill_chunk_size=None,
    )

    plan = step_scheduler.build_plan(scheduler)

    assert step_scheduler.max_prefill_chunk_size == 512
    assert step_scheduler.prefill_chunk_size == 512
    assert plan.prefills is not None
    assert plan.prefills.chunk_len == 512


def test_step_scheduler_decode_fast_path_when_only_decodes() -> None:
    scheduler = _scheduler_with_requests(
        [
            {"is_prefill": False, "seq_len": 4},
            {"is_prefill": False, "seq_len": 5},
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=2,
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


def test_step_scheduler_limits_multimodal_admissions_per_step() -> None:
    scheduler = RequestScheduler(max_active_requests=3)
    scheduler.enqueue_request(
        "mm0",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2], "is_multimodal": True},
    )
    scheduler.enqueue_request(
        "mm1",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2], "is_multimodal": True},
    )
    scheduler.enqueue_request(
        "txt0",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2], "is_multimodal": False},
    )

    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=3,
        max_admit_per_step=3,
        multimodal_params=MultiModalSchedulingParams(
            max_admit_multimodal_per_step=1,
        ),
    )
    plan = step_scheduler.build_plan(scheduler)

    assert plan.admissions is not None
    admitted = set(plan.admissions.request_ids)
    assert "txt0" in admitted
    assert len([rid for rid in admitted if rid.startswith("mm")]) == 1
    assert plan.admitted_multimodal_requests == 1


def test_step_scheduler_limits_multimodal_prefill_batch() -> None:
    scheduler = _scheduler_with_requests(
        [
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3], "is_multimodal": True},
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3], "is_multimodal": True},
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3], "is_multimodal": False},
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=12,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=3,
        multimodal_params=MultiModalSchedulingParams(
            max_prefill_multimodal_requests_per_batch=1,
        ),
    )
    plan = step_scheduler.build_plan(scheduler)

    assert plan.prefills is not None
    selected = set(plan.prefills.request_ids)
    assert len([rid for rid in selected if rid in {"r0", "r1"}]) == 1
    assert plan.prefill_multimodal_requests == 1


def test_step_scheduler_relaxes_multimodal_prefill_limit_on_low_prefix_hit_rate() -> None:
    scheduler = _scheduler_with_requests(
        [
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3], "is_multimodal": True},
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3], "is_multimodal": True},
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3], "is_multimodal": True},
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=12,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=3,
        multimodal_params=MultiModalSchedulingParams(
            max_prefill_multimodal_requests_per_batch=1,
            multimodal_prefix_cache_relax_threshold=0.2,
            multimodal_prefix_cache_tighten_threshold=0.8,
            multimodal_prefill_limit_relax_delta=1,
        ),
    )
    scheduler = _scheduler_with_requests(
        [
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3], "is_multimodal": True},
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3], "is_multimodal": True},
            {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3], "is_multimodal": False},
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=12,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=3,
        multimodal_params=MultiModalSchedulingParams(
            max_prefill_multimodal_requests_per_batch=2,
            multimodal_prefix_cache_relax_threshold=0.2,
            multimodal_prefix_cache_tighten_threshold=0.8,
            multimodal_prefill_limit_tighten_delta=1,
        ),
    )
    step_scheduler.update_runtime_feedback(
        {"observer": {"multimodal": {"prefix_cache_hit_rate": 1.0}}}
    )

    plan = step_scheduler.build_plan(scheduler)

    assert plan.prefills is not None
    assert plan.prefill_multimodal_requests == 1
    assert plan.effective_prefill_multimodal_request_limit == 1
    assert plan.prefill_multimodal_limit_relaxed is False
    assert plan.prefill_multimodal_limit_tightened is True
    assert plan.prefill_multimodal_limit_triggered is True


def test_step_scheduler_limits_multimodal_lora_admissions_per_step() -> None:
    scheduler = RequestScheduler(max_active_requests=4)
    scheduler.enqueue_request(
        "mm_lora0",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2],
            "is_multimodal": True,
            "lora_id": "adapter-a",
        },
    )
    scheduler.enqueue_request(
        "mm_lora1",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2],
            "is_multimodal": True,
            "lora_id": "adapter-b",
        },
    )
    scheduler.enqueue_request(
        "mm_only",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2], "is_multimodal": True},
    )
    scheduler.enqueue_request(
        "txt_only",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2], "is_multimodal": False},
    )

    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=4,
        max_admit_per_step=4,
        multimodal_params=MultiModalSchedulingParams(
            max_admit_multimodal_per_step=2,
            max_admit_multimodal_lora_per_step=1,
        ),
    )
    plan = step_scheduler.build_plan(scheduler)

    assert plan.admissions is not None
    admitted = set(plan.admissions.request_ids)
    assert "txt_only" in admitted
    assert "mm_only" in admitted
    assert len([rid for rid in admitted if rid.startswith("mm_lora")]) == 1
    assert plan.admitted_multimodal_requests == 2
    assert plan.admitted_multimodal_lora_requests == 1


def test_step_scheduler_limits_multimodal_lora_prefill_batch() -> None:
    scheduler = _scheduler_with_requests(
        [
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": "adapter-a",
            },
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": "adapter-b",
            },
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": None,
            },
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=12,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=3,
        multimodal_params=MultiModalSchedulingParams(
            max_prefill_multimodal_requests_per_batch=3,
            max_prefill_multimodal_lora_requests_per_batch=1,
        ),
    )
    plan = step_scheduler.build_plan(scheduler)

    assert plan.prefills is not None
    selected = set(plan.prefills.request_ids)
    assert "r2" in selected
    assert len([rid for rid in selected if rid in {"r0", "r1"}]) == 1
    assert plan.prefill_multimodal_requests == 2
    assert plan.prefill_multimodal_lora_requests == 1


def test_step_scheduler_relaxes_multimodal_lora_prefill_limit_on_low_prefix_hit_rate() -> None:
    scheduler = _scheduler_with_requests(
        [
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": "adapter-a",
            },
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": "adapter-b",
            },
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": None,
            },
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=12,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=3,
        multimodal_params=MultiModalSchedulingParams(
            max_prefill_multimodal_requests_per_batch=3,
            max_prefill_multimodal_lora_requests_per_batch=1,
            multimodal_prefix_cache_relax_threshold=0.2,
            multimodal_prefix_cache_tighten_threshold=0.8,
            multimodal_lora_prefill_limit_relax_delta=1,
        ),
    )
    step_scheduler.update_runtime_feedback(
        {"observer": {"multimodal": {"prefix_cache_hit_rate": 0.0}}}
    )

    plan = step_scheduler.build_plan(scheduler)

    assert plan.prefills is not None
    assert plan.prefill_multimodal_lora_requests == 2
    assert plan.effective_prefill_multimodal_lora_request_limit == 2
    assert plan.prefill_multimodal_lora_limit_relaxed is True
    assert plan.prefill_multimodal_lora_limit_tightened is False
    assert plan.prefill_multimodal_lora_limit_triggered is False


def test_step_scheduler_tightens_multimodal_lora_prefill_limit_on_high_prefix_hit_rate() -> None:
    scheduler = _scheduler_with_requests(
        [
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": "adapter-a",
            },
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": "adapter-b",
            },
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": None,
            },
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=12,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=3,
        multimodal_params=MultiModalSchedulingParams(
            max_prefill_multimodal_requests_per_batch=3,
            max_prefill_multimodal_lora_requests_per_batch=2,
            multimodal_prefix_cache_relax_threshold=0.2,
            multimodal_prefix_cache_tighten_threshold=0.8,
            multimodal_lora_prefill_limit_tighten_delta=1,
        ),
    )
    step_scheduler.update_runtime_feedback(
        {"observer": {"multimodal": {"prefix_cache_hit_rate": 1.0}}}
    )

    plan = step_scheduler.build_plan(scheduler)

    assert plan.prefills is not None
    assert plan.prefill_multimodal_lora_requests == 1
    assert plan.effective_prefill_multimodal_lora_request_limit == 1
    assert plan.prefill_multimodal_lora_limit_relaxed is False
    assert plan.prefill_multimodal_lora_limit_tightened is True
    assert plan.prefill_multimodal_lora_limit_triggered is True


def test_step_scheduler_relaxes_multimodal_lora_prefill_limit_on_fairness_gap() -> None:
    scheduler = _scheduler_with_requests(
        [
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": "adapter-a",
            },
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": "adapter-a",
            },
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": "adapter-a",
            },
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": "adapter-b",
            },
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=2,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        multimodal_params=MultiModalSchedulingParams(
            max_prefill_multimodal_requests_per_batch=4,
            max_prefill_multimodal_lora_requests_per_batch=1,
            multimodal_prefix_cache_relax_threshold=0.0,
            multimodal_prefix_cache_tighten_threshold=0.8,
            multimodal_lora_fairness_relax_threshold=0.2,
            multimodal_lora_prefill_limit_relax_delta=1,
        ),
    )
    step_scheduler.update_runtime_feedback(
        {"observer": {"multimodal": {"prefix_cache_hit_rate": 0.5}}}
    )

    plan = step_scheduler.build_plan(scheduler)

    assert plan.prefills is not None
    assert plan.effective_prefill_multimodal_lora_request_limit == 2
    assert plan.prefill_multimodal_lora_limit_relaxed is True
    assert plan.prefill_multimodal_lora_limit_relaxed_by_fairness is True
    assert plan.prefill_multimodal_lora_max_fairness_gap >= 0.2


def test_step_scheduler_tightens_multimodal_lora_prefill_limit_on_locality_under_high_hit_rate() -> None:
    scheduler = _scheduler_with_requests(
        [
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": "adapter-a",
            },
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": "adapter-b",
            },
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3],
                "is_multimodal": True,
                "lora_id": None,
            },
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=12,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=3,
        multimodal_params=MultiModalSchedulingParams(
            max_prefill_multimodal_requests_per_batch=3,
            max_prefill_multimodal_lora_requests_per_batch=2,
            multimodal_prefix_cache_tighten_threshold=0.8,
            multimodal_lora_locality_tighten_threshold=0.2,
            multimodal_lora_prefill_limit_tighten_delta=1,
        ),
    )
    step_scheduler.update_runtime_feedback(
        {"observer": {"multimodal": {"prefix_cache_hit_rate": 1.0}}}
    )

    plan = step_scheduler.build_plan(scheduler)

    assert plan.prefills is not None
    assert plan.effective_prefill_multimodal_lora_request_limit == 1
    assert plan.prefill_multimodal_lora_limit_tightened is True
    assert plan.prefill_multimodal_lora_limit_tightened_by_locality is True
    assert plan.prefill_multimodal_lora_max_fairness_gap <= 0.2


def test_step_scheduler_relaxes_multimodal_lora_decode_limit_on_fairness_gap() -> None:
    scheduler = _scheduler_with_requests(
        [
            {"is_prefill": False, "seq_len": 8, "is_multimodal": True, "lora_id": "adapter-a"},
            {"is_prefill": False, "seq_len": 8, "is_multimodal": True, "lora_id": "adapter-a"},
            {"is_prefill": False, "seq_len": 8, "is_multimodal": True, "lora_id": "adapter-a"},
            {"is_prefill": False, "seq_len": 8, "is_multimodal": True, "lora_id": "adapter-b"},
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=2,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        multimodal_params=MultiModalSchedulingParams(
            max_decode_multimodal_requests_per_batch=4,
            max_decode_multimodal_lora_requests_per_batch=1,
            multimodal_lora_fairness_relax_threshold=0.2,
            multimodal_lora_prefill_limit_relax_delta=1,
        ),
    )
    step_scheduler.update_runtime_feedback(
        {"observer": {"multimodal": {"prefix_cache_hit_rate": 0.5}}}
    )

    plan = step_scheduler.build_plan(scheduler)

    assert plan.decodes is not None
    assert plan.effective_decode_multimodal_lora_request_limit == 2
    assert plan.decode_multimodal_lora_limit_relaxed is True
    assert plan.decode_multimodal_lora_limit_relaxed_by_fairness is True
    assert plan.decode_multimodal_lora_max_fairness_gap >= 0.2


def test_step_scheduler_tightens_multimodal_lora_decode_limit_on_locality_under_high_hit_rate() -> None:
    scheduler = _scheduler_with_requests(
        [
            {"is_prefill": False, "seq_len": 8, "is_multimodal": True, "lora_id": "adapter-a"},
            {"is_prefill": False, "seq_len": 8, "is_multimodal": True, "lora_id": "adapter-b"},
            {"is_prefill": False, "seq_len": 8, "is_multimodal": True, "lora_id": None},
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=4,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        multimodal_params=MultiModalSchedulingParams(
            max_decode_multimodal_requests_per_batch=3,
            max_decode_multimodal_lora_requests_per_batch=2,
            multimodal_prefix_cache_tighten_threshold=0.8,
            multimodal_lora_locality_tighten_threshold=0.2,
            multimodal_lora_prefill_limit_tighten_delta=1,
        ),
    )
    step_scheduler.update_runtime_feedback(
        {"observer": {"multimodal": {"prefix_cache_hit_rate": 1.0}}}
    )

    plan = step_scheduler.build_plan(scheduler)

    assert plan.decodes is not None
    assert plan.effective_decode_multimodal_lora_request_limit == 1
    assert plan.decode_multimodal_lora_limit_tightened is True
    assert plan.decode_multimodal_lora_limit_tightened_by_locality is True
    assert plan.decode_multimodal_lora_max_fairness_gap <= 0.2


def test_step_scheduler_tracks_multimodal_queue_wait_metrics(monkeypatch) -> None:
    scheduler = RequestScheduler(max_active_requests=2)
    scheduler.enqueue_request(
        "mm0",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2],
            "queued_at": 4.0,
            "is_multimodal": True,
            "multi_modal_data": {"image": [{"image": "file:///tmp/a.png"}]},
        },
    )
    scheduler.enqueue_request(
        "txt0",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2], "queued_at": 6.0},
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
    )
    plan = step_scheduler.build_plan(scheduler)

    assert plan.queued_multimodal_requests == 1
    assert plan.queued_multimodal_avg_wait_s == 6.0
    assert plan.queued_multimodal_max_wait_s == 6.0
    assert plan.queued_multimodal_p95_wait_s == 6.0


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


def test_step_scheduler_reports_lora_adapter_counts() -> None:
    scheduler = RequestScheduler(max_active_requests=3)
    scheduler.add_request(
        "prefill-a",
        {
            "slot_idx": 0,
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2, 3, 4],
            "generated_ids": [],
            "lora_id": "adapter-a",
        },
    )
    scheduler.add_request(
        "decode-b",
        {
            "slot_idx": 1,
            "is_prefill": False,
            "seq_len": 4,
            "generated_ids": [1],
            "input_ids": [1, 2],
            "lora_id": "adapter-b",
        },
    )
    scheduler.enqueue_request(
        "queued-a",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2],
            "service_class": "latency",
            "lora_id": "adapter-a",
        },
    )

    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=2,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=1,
        max_admit_per_step=1,
    )
    plan = step_scheduler.build_plan(scheduler)

    assert plan.admitted_lora_adapters == {"adapter-a": 1}
    assert plan.prefill_lora_adapters == {"adapter-a": 1}
    assert plan.decode_lora_adapters == {"adapter-b": 1}
    assert plan.queued_lora_adapters == {"adapter-a": 1}


def test_step_scheduler_limits_admit_lora_adapters_per_step_and_rotates() -> None:
    scheduler = RequestScheduler(max_active_requests=3)
    scheduler.enqueue_request(
        "adapter-a-0",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2],
            "lora_id": "adapter-a",
        },
    )
    scheduler.enqueue_request(
        "adapter-b-0",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2],
            "lora_id": "adapter-b",
        },
    )
    scheduler.enqueue_request(
        "adapter-c-0",
        {
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [1, 2],
            "lora_id": "adapter-c",
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
        lora_params=LoraSchedulingParams(
            max_admit_lora_adapters_per_step=1,
        ),
    )

    plan1 = step_scheduler.build_plan(scheduler)
    plan2 = step_scheduler.build_plan(scheduler)

    assert plan1.admissions is not None
    assert plan2.admissions is not None
    assert len(plan1.admitted_lora_adapters or {}) == 1
    assert len(plan2.admitted_lora_adapters or {}) == 1
    assert plan1.admissions.request_ids != plan2.admissions.request_ids


def test_step_scheduler_relaxes_admit_lora_limit_on_fairness_gap() -> None:
    scheduler = RequestScheduler(max_active_requests=4)
    scheduler.enqueue_request(
        "adapter-a-0",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1], "lora_id": "adapter-a"},
    )
    scheduler.enqueue_request(
        "adapter-b-0",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2], "lora_id": "adapter-b"},
    )
    scheduler.enqueue_request(
        "adapter-c-0",
        {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2, 3], "lora_id": "adapter-c"},
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
        lora_params=LoraSchedulingParams(
            max_admit_lora_adapters_per_step=1,
            lora_fairness_relax_threshold=0.3,
        ),
    )

    plan = step_scheduler.build_plan(scheduler)

    assert plan.admissions is not None
    assert plan.effective_admit_lora_adapter_limit == 2
    assert len(plan.admitted_lora_adapters or {}) == 2
    assert plan.admitted_max_lora_fairness_gap > 0.0


def test_step_scheduler_limits_prefill_lora_adapters_per_batch() -> None:
    scheduler = _scheduler_with_requests(
        [
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3, 4, 5, 6],
                "lora_id": "adapter-a",
            },
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3, 4, 5, 6],
                "lora_id": "adapter-b",
            },
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=1.0,
        prefill_microbatch_size=2,
        lora_params=LoraSchedulingParams(
            max_prefill_lora_adapters_per_batch=1,
        ),
    )

    plan1 = step_scheduler.build_plan(scheduler)
    plan2 = step_scheduler.build_plan(scheduler)

    assert plan1.prefills is not None
    assert plan2.prefills is not None
    assert len(plan1.prefill_lora_adapters or {}) == 1
    assert len(plan2.prefill_lora_adapters or {}) == 1
    assert plan1.prefills.request_ids != plan2.prefills.request_ids


def test_step_scheduler_tightens_prefill_lora_limit_when_locality_is_good() -> None:
    scheduler = _scheduler_with_requests(
        [
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3, 4, 5, 6],
                "lora_id": "adapter-a",
            },
            {
                "is_prefill": True,
                "seq_len": 0,
                "input_ids": [1, 2, 3, 4, 5, 6],
                "lora_id": "adapter-b",
            },
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=8,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=1.0,
        prefill_microbatch_size=2,
        lora_params=LoraSchedulingParams(
            max_prefill_lora_adapters_per_batch=2,
            lora_locality_tighten_threshold=0.01,
        ),
    )

    plan = step_scheduler.build_plan(scheduler)

    assert plan.prefills is not None
    assert plan.effective_prefill_lora_adapter_limit == 1
    assert len(plan.prefill_lora_adapters or {}) == 1
    assert len(plan.prefills.request_ids) == 1


def test_step_scheduler_limits_decode_lora_adapters_per_batch() -> None:
    scheduler = RequestScheduler(max_active_requests=4)
    scheduler.add_request(
        "decode-a",
        {
            "slot_idx": 0,
            "is_prefill": False,
            "seq_len": 4,
            "generated_ids": [1],
            "input_ids": [1, 2],
            "lora_id": "adapter-a",
        },
    )
    scheduler.add_request(
        "decode-b",
        {
            "slot_idx": 1,
            "is_prefill": False,
            "seq_len": 4,
            "generated_ids": [1],
            "input_ids": [1, 2],
            "lora_id": "adapter-b",
        },
    )
    scheduler.add_request(
        "decode-c",
        {
            "slot_idx": 2,
            "is_prefill": False,
            "seq_len": 4,
            "generated_ids": [1],
            "input_ids": [1, 2],
            "lora_id": "adapter-c",
        },
    )

    step_scheduler = StepScheduler(
        step_token_budget=3,
        decode_priority_enabled=True,
        prefill_chunk_size=4,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=1,
        lora_params=LoraSchedulingParams(
            max_decode_lora_adapters_per_batch=2,
        ),
    )

    plan1 = step_scheduler.build_plan(scheduler)
    plan2 = step_scheduler.build_plan(scheduler)

    assert plan1.decodes is not None
    assert plan2.decodes is not None
    assert len(plan1.decode_lora_adapters or {}) == 2
    assert len(plan2.decode_lora_adapters or {}) == 2
    assert set(plan1.decodes.request_ids) != set(plan2.decodes.request_ids)


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
