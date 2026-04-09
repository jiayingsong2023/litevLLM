# SPDX-License-Identifier: Apache-2.0
from vllm.engine.runtime_observer import InMemoryRuntimeObserver
from vllm.engine.step_plan import StepPlan


def test_inmemory_runtime_observer_records_lifecycle_events() -> None:
    observer = InMemoryRuntimeObserver()
    observer.on_request_added("r1", {"input_ids": [1, 2], "is_prefill": True})
    observer.on_request_admitted("r1", 0.125, "latency")
    observer.on_prefix_cache_event(
        "r1",
        hit=True,
        exact=False,
        prefix_len=3,
        saved_prefill_tokens=3,
    )
    observer.on_prefix_cache_event(
        "r2",
        hit=False,
        exact=False,
        prefix_len=0,
        saved_prefill_tokens=0,
    )
    observer.on_prefix_cache_event(
        "r3",
        hit=True,
        exact=True,
        prefix_len=5,
        saved_prefill_tokens=5,
    )
    observer.on_preemption_event(
        preempted_prefill_requests=2,
        queued_backlog=4,
    )
    observer.on_request_rejected("r2", "capacity")
    observer.on_step_started(
        StepPlan(
            admissions=None,
            prefills=None,
            decodes=None,
            step_token_budget=4,
            aged_admission_count=1,
            admitted_service_classes={"latency": 1},
            prefill_service_classes={"balanced": 1},
            decode_service_classes={"background": 2},
            queued_service_classes={"latency": 1},
            queued_avg_wait_s=0.125,
            queued_max_wait_s=0.125,
            queued_p95_wait_s=0.125,
            queued_service_class_avg_wait_s={"latency": 0.125},
            queued_service_class_max_wait_s={"latency": 0.125},
            queued_service_class_p95_wait_s={"latency": 0.125},
            fairness_guardrail_triggered=True,
            prefill_starvation_protected=True,
        )
    )
    observer.on_first_token("r1", 0.25)
    observer.on_request_finished("r1", "eos")
    observer.on_request_aborted("r3")
    observer.on_background_error(RuntimeError("boom"), ["r4"])

    assert observer.added == ["r1"]
    assert observer.admitted == [("r1", 0.125, "latency")]
    assert observer.prefix_cache_events == [
        ("r1", True, False, 3, 3),
        ("r2", False, False, 0, 0),
        ("r3", True, True, 5, 5),
    ]
    assert observer.prefix_cache_hits == 2
    assert observer.prefix_cache_misses == 1
    assert observer.prefix_cache_exact_hits == 1
    assert observer.prefix_cache_partial_hits == 1
    assert observer.prefix_cache_saved_prefill_tokens == 8
    assert observer.preemption_events == [(2, 4)]
    assert observer.preempted_steps == 1
    assert observer.preempted_prefill_requests == 2
    assert observer.rejected == [("r2", "capacity")]
    assert observer.step_count == 1
    assert observer.aged_admissions == 1
    assert observer.starvation_protected_steps == 1
    assert observer.fairness_guardrail_triggered_steps == 1
    assert observer.fairness_backlog_service_classes == {"latency": 1}
    assert observer.fairness_admitted_service_classes == {"latency": 1}
    assert observer.fairness_prefill_service_classes == {"balanced": 1}
    assert observer.fairness_decode_service_classes == {"background": 2}
    assert observer.max_queue_wait_s == 0.125
    assert observer.per_class_max_queue_wait_s == {"latency": 0.125}
    assert observer.first_tokens == [("r1", 0.25)]
    assert observer.finished == [("r1", "eos")]
    assert observer.aborted == ["r3"]
    assert observer.background_errors


def test_inmemory_runtime_observer_stats_snapshot() -> None:
    observer = InMemoryRuntimeObserver()

    observer.on_request_added("r1", {"input_ids": [1], "is_prefill": True})
    observer.on_request_rejected("r2", "capacity")
    observer.on_request_admitted("r1", 0.2, "latency")
    observer.on_prefix_cache_event(
        "r1",
        hit=True,
        exact=True,
        prefix_len=4,
        saved_prefill_tokens=4,
    )
    observer.on_preemption_event(preempted_prefill_requests=1, queued_backlog=2)
    observer.on_request_finished("r1", "eos")
    observer.on_request_aborted("r3")
    observer.on_background_error(RuntimeError("boom"), ["r4"])
    observer.on_step_started(
        StepPlan(
            admissions=None,
            prefills=None,
            decodes=None,
            step_token_budget=4,
            aged_admission_count=2,
            admitted_service_classes={"latency": 1, "background": 1},
            prefill_service_classes={"latency": 1},
            decode_service_classes={"background": 1},
            queued_service_classes={"latency": 2, "background": 1},
            queued_avg_wait_s=2.0,
            queued_max_wait_s=5.0,
            queued_p95_wait_s=5.0,
            queued_service_class_avg_wait_s={"latency": 2.5, "background": 1.0},
            queued_service_class_max_wait_s={"latency": 5.0, "background": 1.0},
            queued_service_class_p95_wait_s={"latency": 5.0, "background": 1.0},
            fairness_guardrail_triggered=True,
            prefill_starvation_protected=True,
        )
    )

    stats = observer.stats()

    assert stats["added_requests"] == 1
    assert stats["rejected_requests"] == 1
    assert stats["admitted_requests"] == 1
    assert stats["finished_requests"] == 1
    assert stats["aborted_requests"] == 1
    assert stats["background_error_count"] == 1
    assert stats["prefix_cache"] == {
        "events": 1,
        "hits": 1,
        "misses": 0,
        "exact_hits": 1,
        "partial_hits": 0,
        "saved_prefill_tokens": 4,
        "hit_rate": 1.0,
        "exact_hit_rate": 1.0,
        "partial_hit_rate": 0.0,
        "avg_saved_prefill_tokens_per_hit": 4.0,
        "avg_saved_prefill_tokens_per_request": 4.0,
    }
    assert stats["preemption"] == {
        "events": 1,
        "preempted_steps": 1,
        "preempted_prefill_requests": 1,
    }
    assert stats["fairness"] == {
        "aged_admissions": 2,
        "starvation_protected_steps": 1,
        "fairness_guardrail_triggered_steps": 1,
        "backlog_service_classes": {"latency": 2, "background": 1},
        "admitted_service_classes": {"latency": 1, "background": 1},
        "prefill_service_classes": {"latency": 1},
        "decode_service_classes": {"background": 1},
        "max_queue_wait_s": 5.0,
        "p95_queue_wait_s": 5.0,
        "avg_admitted_queue_wait_s": 0.2,
        "per_class_avg_queue_wait_s": {"latency": 1.7333333333333334, "background": 1.0},
        "per_class_max_queue_wait_s": {"latency": 5.0, "background": 1.0},
        "per_class_p95_queue_wait_s": {"latency": 5.0, "background": 1.0},
    }


def test_inmemory_runtime_observer_reset_stats() -> None:
    observer = InMemoryRuntimeObserver()
    observer.on_request_added("r1", {"input_ids": [1], "is_prefill": True})
    observer.on_prefix_cache_event(
        "r1",
        hit=True,
        exact=False,
        prefix_len=2,
        saved_prefill_tokens=2,
    )
    observer.on_preemption_event(preempted_prefill_requests=1, queued_backlog=1)
    observer.on_request_finished("r1", "eos")
    observer.reset_stats()

    assert observer.stats() == {
        "added_requests": 0,
        "rejected_requests": 0,
        "admitted_requests": 0,
        "finished_requests": 0,
        "aborted_requests": 0,
        "background_error_count": 0,
        "step_count": 0,
        "first_token_count": 0,
        "prefix_cache": {
            "events": 0,
            "hits": 0,
            "misses": 0,
            "exact_hits": 0,
            "partial_hits": 0,
            "saved_prefill_tokens": 0,
            "hit_rate": 0.0,
            "exact_hit_rate": 0.0,
            "partial_hit_rate": 0.0,
            "avg_saved_prefill_tokens_per_hit": 0.0,
            "avg_saved_prefill_tokens_per_request": 0.0,
        },
        "preemption": {
            "events": 0,
            "preempted_steps": 0,
            "preempted_prefill_requests": 0,
        },
        "fairness": {
            "aged_admissions": 0,
            "starvation_protected_steps": 0,
            "fairness_guardrail_triggered_steps": 0,
            "backlog_service_classes": {},
            "admitted_service_classes": {},
            "prefill_service_classes": {},
            "decode_service_classes": {},
            "max_queue_wait_s": 0.0,
            "p95_queue_wait_s": 0.0,
            "avg_admitted_queue_wait_s": 0.0,
            "per_class_avg_queue_wait_s": {},
            "per_class_max_queue_wait_s": {},
            "per_class_p95_queue_wait_s": {},
        },
    }
