# SPDX-License-Identifier: Apache-2.0
from vllm.engine.request_state import RequestState
from vllm.engine.runtime_observer import InMemoryRuntimeObserver
from vllm.engine.step_plan import StepPlan, StepPlanMetrics


def _request_state(**overrides) -> RequestState:
    data = {
        "request_id": "r1",
        "prompt": "",
        "input_ids": [1],
        "sampling_params": object(),
    }
    data.update(overrides)
    return RequestState(**data)


def test_inmemory_runtime_observer_records_lifecycle_events() -> None:
    observer = InMemoryRuntimeObserver()
    observer.on_request_added(
        "r1",
        _request_state(
            input_ids=[1, 2],
            multi_modal_data={"image": [{"image": "file:///tmp/cat.png"}]},
            lora_id="adapter-a",
        ),
    )
    observer.on_request_admitted("r1", 0.125, "latency")
    observer.on_prefix_cache_event(
        "r1",
        hit=True,
        exact=False,
        prefix_len=3,
        saved_prefill_tokens=3,
        is_multimodal=True,
    )
    observer.on_prefix_cache_event(
        "r2",
        hit=False,
        exact=False,
        prefix_len=0,
        saved_prefill_tokens=0,
        is_multimodal=True,
    )
    observer.on_prefix_cache_event(
        "r3",
        hit=True,
        exact=True,
        prefix_len=5,
        saved_prefill_tokens=5,
        is_multimodal=False,
    )
    observer.on_preemption_event(
        preempted_prefill_requests=2,
        queued_backlog=4,
        multimodal_prefill_requests=1,
    )
    observer.on_request_rejected("r2", "capacity")
    observer.on_step_started(
        StepPlan(
            admissions=None,
            prefills=None,
            decodes=None,
            step_token_budget=4,
            metrics=StepPlanMetrics(
                aged_admission_count=1,
                admitted_service_classes={"latency": 1},
                admitted_multimodal_requests=1,
                admitted_multimodal_lora_requests=1,
                effective_admit_multimodal_lora_request_limit=1,
                admit_multimodal_lora_limit_triggered=True,
                admitted_lora_adapters={"adapter-a": 1},
                admit_lora_limit_relaxed=True,
                prefill_service_classes={"balanced": 1},
                prefill_multimodal_requests=1,
                prefill_multimodal_lora_requests=1,
                effective_prefill_multimodal_lora_request_limit=1,
                prefill_multimodal_lora_limit_relaxed=True,
                prefill_multimodal_lora_limit_relaxed_by_fairness=True,
                prefill_multimodal_lora_max_fairness_gap=0.25,
                prefill_multimodal_lora_limit_triggered=False,
                prefill_lora_adapters={"adapter-a": 1, "adapter-b": 1},
                prefill_lora_limit_tightened=True,
                decode_service_classes={"background": 2},
                decode_multimodal_requests=0,
                decode_multimodal_lora_requests=0,
                effective_decode_multimodal_lora_request_limit=0,
                decode_lora_adapters={"adapter-b": 2},
                decode_lora_limit_relaxed=True,
                queued_service_classes={"latency": 1},
                queued_multimodal_requests=1,
                queued_multimodal_lora_requests=1,
                queued_lora_adapters={"adapter-a": 1},
                queued_avg_wait_s=0.125,
                queued_max_wait_s=0.125,
                queued_p95_wait_s=0.125,
                queued_multimodal_avg_wait_s=0.2,
                queued_multimodal_max_wait_s=0.2,
                queued_multimodal_p95_wait_s=0.2,
                queued_service_class_avg_wait_s={"latency": 0.125},
                queued_service_class_max_wait_s={"latency": 0.125},
                queued_service_class_p95_wait_s={"latency": 0.125},
                fairness_guardrail_triggered=True,
                prefill_starvation_protected=True,
            ),
        )
    )
    observer.on_prefill_executed(object(), 1)
    observer.on_first_token("r1", 0.25)
    observer.on_decode_executed(object(), 1)
    observer.on_request_finished("r1", "eos")
    observer.on_request_aborted("r3")
    observer.on_background_error(RuntimeError("boom"), ["r4"])
    observer.on_model_surface_resolved(
        event_name="experimental_model_surface",
        model_name="models/custom-llama-8b",
        model_type="llama",
        status="experimental",
        reason="model_not_in_regression_surface",
    )

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
    assert observer.multimodal_prefix_cache_hits == 1
    assert observer.multimodal_prefix_cache_misses == 1
    assert observer.multimodal_prefix_cache_saved_prefill_tokens == 3
    assert observer.preemption_events == [(2, 4)]
    assert observer.preempted_steps == 1
    assert observer.preempted_prefill_requests == 2
    assert observer.preempted_multimodal_prefill_requests == 1
    assert observer.rejected == [("r2", "capacity")]
    assert observer.rejection_reason_counts == {"capacity": 1}
    assert observer.step_count == 1
    assert observer.step_multimodal_admitted_requests == 1
    assert observer.step_multimodal_lora_admitted_requests == 1
    assert observer.step_multimodal_prefill_requests == 1
    assert observer.step_multimodal_lora_prefill_requests == 1
    assert observer.step_multimodal_decode_requests == 0
    assert observer.step_multimodal_lora_decode_requests == 0
    assert observer.step_multimodal_queued_requests == 1
    assert observer.step_multimodal_lora_queued_requests == 1
    assert observer.mixed_multimodal_lora_prefill_steps == 0
    assert observer.multimodal_prefill_limit_sum == 0
    assert observer.multimodal_prefill_limit_relaxed_steps == 0
    assert observer.multimodal_prefill_limit_tightened_steps == 0
    assert observer.multimodal_prefill_limit_triggered_steps == 0
    assert observer.multimodal_lora_admit_limit_triggered_steps == 1
    assert observer.multimodal_lora_prefill_limit_relaxed_steps == 1
    assert observer.multimodal_lora_prefill_limit_relaxed_by_fairness_steps == 1
    assert observer.multimodal_lora_prefill_limit_tightened_steps == 0
    assert observer.multimodal_lora_prefill_limit_tightened_by_locality_steps == 0
    assert observer.multimodal_lora_prefill_limit_triggered_steps == 0
    assert observer.multimodal_lora_decode_limit_relaxed_steps == 0
    assert observer.multimodal_lora_decode_limit_tightened_steps == 0
    assert observer.multimodal_lora_decode_limit_triggered_steps == 0
    assert observer.multimodal_max_queue_wait_s == 0.2
    assert observer.aged_admissions == 1
    assert observer.starvation_protected_steps == 1
    assert observer.fairness_guardrail_triggered_steps == 1
    assert observer.fairness_backlog_service_classes == {"latency": 1}
    assert observer.fairness_admitted_service_classes == {"latency": 1}
    assert observer.fairness_prefill_service_classes == {"balanced": 1}
    assert observer.fairness_decode_service_classes == {"background": 2}
    assert observer.lora_admitted_adapters == {"adapter-a": 1}
    assert observer.lora_prefill_adapters == {"adapter-a": 1, "adapter-b": 1}
    assert observer.lora_decode_adapters == {"adapter-b": 2}
    assert observer.lora_backlog_adapters == {"adapter-a": 1}
    assert observer.lora_admit_relaxed_steps == 1
    assert observer.lora_admit_tightened_steps == 0
    assert observer.lora_prefill_relaxed_steps == 0
    assert observer.lora_prefill_tightened_steps == 1
    assert observer.lora_decode_relaxed_steps == 1
    assert observer.lora_decode_tightened_steps == 0
    assert observer.prefill_step_count == 1
    assert observer.decode_step_count == 1
    assert observer.mixed_lora_prefill_steps == 1
    assert observer.mixed_lora_decode_steps == 0
    assert observer.max_queue_wait_s == 0.125
    assert observer.per_class_max_queue_wait_s == {"latency": 0.125}
    assert observer.first_tokens == [("r1", 0.25)]
    assert observer.finished == [("r1", "eos")]
    assert observer.aborted == ["r3"]
    assert observer.background_errors
    assert observer.model_surface_events == [
        {
            "event_name": "experimental_model_surface",
            "model_name": "models/custom-llama-8b",
            "model_type": "llama",
            "status": "experimental",
            "reason": "model_not_in_regression_surface",
        }
    ]



def test_inmemory_runtime_observer_records_deepseek_events() -> None:
    observer = InMemoryRuntimeObserver()

    observer.on_deepseek_event("decode_batch", batch_size=2, latency_ms=1.5)
    observer.on_deepseek_event("decode_batch", batch_size=1, latency_ms=0.5)
    observer.on_deepseek_event("stager_cache_hit", cache="grouped")

    assert observer.stats()["deepseek"] == {
        "events": {
            "decode_batch": 2,
            "stager_cache_hit": 1,
        },
        "decode_batch_tokens": 3,
        "decode_batch_max_size": 2,
        "decode_batch_latency_ms_sum": 2.0,
        "stager_cache_hits": 1,
        "stager_cache_misses": 0,
        "kv_family_allocations": 0,
    }

    observer.reset_stats()

    assert observer.stats()["deepseek"]["events"] == {}

def test_inmemory_runtime_observer_stats_snapshot() -> None:
    observer = InMemoryRuntimeObserver()

    observer.on_request_added("r1", _request_state())
    observer.on_request_rejected("r2", "capacity")
    observer.on_request_admitted("r1", 0.2, "latency")
    observer.on_prefix_cache_event(
        "r1",
        hit=True,
        exact=True,
        prefix_len=4,
        saved_prefill_tokens=4,
        is_multimodal=True,
    )
    observer.on_preemption_event(preempted_prefill_requests=1, queued_backlog=2)
    observer.on_preemption_event(
        preempted_prefill_requests=0,
        queued_backlog=1,
        multimodal_prefill_requests=1,
    )
    observer.on_multimodal_preemption_guard(
        protected_prefill_requests=2,
        prefix_cache_hit_rate=0.9,
    )
    observer.on_prefill_executed(object(), 1)
    observer.on_decode_executed(object(), 1)
    observer.on_request_finished("r1", "eos")
    observer.on_request_aborted("r3")
    observer.on_background_error(RuntimeError("boom"), ["r4"])
    observer.on_model_surface_resolved(
        event_name="experimental_model_surface",
        model_name="models/custom-llama-8b",
        model_type="llama",
        status="experimental",
        reason="model_not_in_regression_surface",
    )
    observer.on_step_started(
        StepPlan(
            admissions=None,
            prefills=None,
            decodes=None,
            step_token_budget=4,
            metrics=StepPlanMetrics(
                aged_admission_count=2,
                admitted_service_classes={"latency": 1, "background": 1},
                admitted_multimodal_requests=1,
                admitted_multimodal_lora_requests=1,
                effective_admit_multimodal_lora_request_limit=1,
                admit_multimodal_lora_limit_triggered=True,
                admitted_lora_adapters={"adapter-a": 1},
                admit_lora_limit_relaxed=True,
                prefill_service_classes={"latency": 1},
                prefill_multimodal_requests=2,
                prefill_multimodal_lora_requests=1,
                effective_prefill_multimodal_request_limit=2,
                prefill_multimodal_limit_relaxed=True,
                prefill_multimodal_limit_triggered=True,
                effective_prefill_multimodal_lora_request_limit=1,
                prefill_multimodal_lora_limit_tightened=True,
                prefill_multimodal_lora_limit_tightened_by_locality=True,
                prefill_multimodal_lora_max_fairness_gap=0.1,
                prefill_multimodal_lora_limit_triggered=True,
                prefill_lora_adapters={"adapter-a": 1},
                prefill_lora_limit_tightened=True,
                decode_service_classes={"background": 1},
                decode_multimodal_requests=1,
                decode_multimodal_lora_requests=1,
                effective_decode_multimodal_lora_request_limit=1,
                decode_multimodal_lora_limit_tightened=True,
                decode_multimodal_lora_limit_tightened_by_locality=True,
                decode_multimodal_lora_limit_triggered=True,
                decode_multimodal_lora_max_fairness_gap=0.1,
                decode_lora_adapters={"adapter-b": 1},
                decode_lora_limit_relaxed=True,
                queued_service_classes={"latency": 2, "background": 1},
                queued_multimodal_requests=2,
                queued_multimodal_lora_requests=1,
                queued_lora_adapters={"adapter-a": 2, "adapter-b": 1},
                queued_avg_wait_s=2.0,
                queued_max_wait_s=5.0,
                queued_p95_wait_s=5.0,
                queued_multimodal_avg_wait_s=3.0,
                queued_multimodal_max_wait_s=4.0,
                queued_multimodal_p95_wait_s=4.0,
                queued_service_class_avg_wait_s={"latency": 2.5, "background": 1.0},
                queued_service_class_max_wait_s={"latency": 5.0, "background": 1.0},
                queued_service_class_p95_wait_s={"latency": 5.0, "background": 1.0},
                fairness_guardrail_triggered=True,
                prefill_starvation_protected=True,
            ),
        )
    )

    stats = observer.stats()

    assert stats["added_requests"] == 1
    assert stats["rejected_requests"] == 1
    assert stats["admitted_requests"] == 1
    assert stats["finished_requests"] == 1
    assert stats["aborted_requests"] == 1
    assert stats["background_error_count"] == 1
    assert stats["rejections"] == {
        "reasons": {"capacity": 1},
        "queue_timeout": 0,
    }
    assert stats["model_surface"] == {
        "events": 1,
        "experimental_events": 1,
        "supported_events": 0,
        "last_event_name": "experimental_model_surface",
        "last_model_name": "models/custom-llama-8b",
        "last_model_type": "llama",
        "last_status": "experimental",
        "last_reason": "model_not_in_regression_surface",
    }
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
        "events": 2,
        "preempted_steps": 2,
        "preempted_prefill_requests": 1,
        "preempted_multimodal_prefill_requests": 1,
        "protected_multimodal_prefix_steps": 1,
        "protected_multimodal_prefix_prefill_requests": 2,
    }
    assert stats["multimodal"] == {
        "requests": 0,
        "images": 0,
        "multimodal_lora_requests": 0,
        "admitted_requests": 1,
        "prefill_requests": 2,
        "decode_requests": 1,
        "queued_requests": 2,
        "admitted_multimodal_lora_requests": 1,
        "prefill_multimodal_lora_requests": 1,
        "decode_multimodal_lora_requests": 1,
        "queued_multimodal_lora_requests": 1,
        "mixed_multimodal_lora_prefill_steps": 1,
        "avg_effective_prefill_multimodal_limit": 2.0,
        "prefill_multimodal_limit_relaxed_steps": 1,
        "prefill_multimodal_limit_tightened_steps": 0,
        "prefill_multimodal_limit_triggered_steps": 1,
        "prefix_cache_hits": 1,
        "prefix_cache_misses": 0,
        "prefix_cache_saved_prefill_tokens": 4,
        "prefix_cache_hit_rate": 1.0,
        "mixed_multimodal_lora_prefill_ratio": 1.0,
        "avg_effective_admit_multimodal_lora_limit": 1.0,
        "avg_effective_prefill_multimodal_lora_limit": 1.0,
        "avg_effective_decode_multimodal_lora_limit": 1.0,
        "admit_multimodal_lora_limit_triggered_steps": 1,
        "prefill_multimodal_lora_limit_relaxed_steps": 0,
        "prefill_multimodal_lora_limit_tightened_steps": 1,
        "prefill_multimodal_lora_limit_relaxed_by_fairness_steps": 0,
        "prefill_multimodal_lora_limit_tightened_by_locality_steps": 1,
        "prefill_multimodal_lora_limit_triggered_steps": 1,
        "avg_prefill_multimodal_lora_max_fairness_gap": 0.1,
        "decode_multimodal_lora_limit_relaxed_steps": 0,
        "decode_multimodal_lora_limit_tightened_steps": 1,
        "decode_multimodal_lora_limit_relaxed_by_fairness_steps": 0,
        "decode_multimodal_lora_limit_tightened_by_locality_steps": 1,
        "decode_multimodal_lora_limit_triggered_steps": 1,
        "avg_decode_multimodal_lora_max_fairness_gap": 0.1,
        "max_queue_wait_s": 4.0,
        "p95_queue_wait_s": 4.0,
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
        "per_class_avg_queue_wait_s": {
            "latency": 1.7333333333333334,
            "background": 1.0,
        },
        "per_class_max_queue_wait_s": {"latency": 5.0, "background": 1.0},
        "per_class_p95_queue_wait_s": {"latency": 5.0, "background": 1.0},
    }
    assert stats["lora"] == {
        "admitted_adapters": {"adapter-a": 1},
        "prefill_adapters": {"adapter-a": 1},
        "decode_adapters": {"adapter-b": 1},
        "backlog_adapters": {"adapter-a": 2, "adapter-b": 1},
        "admit_relaxed_steps": 1,
        "admit_tightened_steps": 0,
        "prefill_relaxed_steps": 0,
        "prefill_tightened_steps": 1,
        "decode_relaxed_steps": 1,
        "decode_tightened_steps": 0,
        "prefill_step_count": 1,
        "decode_step_count": 1,
        "mixed_lora_prefill_steps": 0,
        "mixed_lora_decode_steps": 0,
        "prefill_locality_rate": 1.0,
        "decode_locality_rate": 1.0,
    }


def test_inmemory_runtime_observer_reset_stats() -> None:
    observer = InMemoryRuntimeObserver()
    observer.on_request_added("r1", _request_state())
    observer.on_prefix_cache_event(
        "r1",
        hit=True,
        exact=False,
        prefix_len=2,
        saved_prefill_tokens=2,
        is_multimodal=True,
    )
    observer.on_preemption_event(
        preempted_prefill_requests=1,
        queued_backlog=1,
        multimodal_prefill_requests=1,
    )
    observer.on_multimodal_preemption_guard(
        protected_prefill_requests=1,
        prefix_cache_hit_rate=0.8,
    )
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
        "rejections": {
            "reasons": {},
            "queue_timeout": 0,
        },
        "deepseek": {
            "events": {},
            "decode_batch_tokens": 0,
            "decode_batch_max_size": 0,
            "decode_batch_latency_ms_sum": 0.0,
            "stager_cache_hits": 0,
            "stager_cache_misses": 0,
            "kv_family_allocations": 0,
        },
        "model_surface": {
            "events": 0,
            "experimental_events": 0,
            "supported_events": 0,
            "last_event_name": "",
            "last_model_name": "",
            "last_model_type": "",
            "last_status": "",
            "last_reason": "",
        },
        "multimodal": {
            "requests": 0,
            "images": 0,
            "multimodal_lora_requests": 0,
            "admitted_requests": 0,
            "prefill_requests": 0,
            "decode_requests": 0,
            "queued_requests": 0,
            "admitted_multimodal_lora_requests": 0,
            "prefill_multimodal_lora_requests": 0,
            "decode_multimodal_lora_requests": 0,
            "queued_multimodal_lora_requests": 0,
            "mixed_multimodal_lora_prefill_steps": 0,
            "avg_effective_prefill_multimodal_limit": 0.0,
            "prefill_multimodal_limit_relaxed_steps": 0,
            "prefill_multimodal_limit_tightened_steps": 0,
            "prefill_multimodal_limit_triggered_steps": 0,
            "prefix_cache_hits": 0,
            "prefix_cache_misses": 0,
            "prefix_cache_saved_prefill_tokens": 0,
            "prefix_cache_hit_rate": 0.0,
            "mixed_multimodal_lora_prefill_ratio": 0.0,
            "avg_effective_admit_multimodal_lora_limit": 0.0,
            "avg_effective_prefill_multimodal_lora_limit": 0.0,
            "avg_effective_decode_multimodal_lora_limit": 0.0,
            "admit_multimodal_lora_limit_triggered_steps": 0,
            "prefill_multimodal_lora_limit_relaxed_steps": 0,
            "prefill_multimodal_lora_limit_tightened_steps": 0,
            "prefill_multimodal_lora_limit_relaxed_by_fairness_steps": 0,
            "prefill_multimodal_lora_limit_tightened_by_locality_steps": 0,
            "prefill_multimodal_lora_limit_triggered_steps": 0,
            "avg_prefill_multimodal_lora_max_fairness_gap": 0.0,
            "decode_multimodal_lora_limit_relaxed_steps": 0,
            "decode_multimodal_lora_limit_tightened_steps": 0,
            "decode_multimodal_lora_limit_relaxed_by_fairness_steps": 0,
            "decode_multimodal_lora_limit_tightened_by_locality_steps": 0,
            "decode_multimodal_lora_limit_triggered_steps": 0,
            "avg_decode_multimodal_lora_max_fairness_gap": 0.0,
            "max_queue_wait_s": 0.0,
            "p95_queue_wait_s": 0.0,
        },
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
            "preempted_multimodal_prefill_requests": 0,
            "protected_multimodal_prefix_steps": 0,
            "protected_multimodal_prefix_prefill_requests": 0,
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
        "lora": {
            "admitted_adapters": {},
            "prefill_adapters": {},
            "decode_adapters": {},
            "backlog_adapters": {},
            "admit_relaxed_steps": 0,
            "admit_tightened_steps": 0,
            "prefill_relaxed_steps": 0,
            "prefill_tightened_steps": 0,
            "decode_relaxed_steps": 0,
            "decode_tightened_steps": 0,
            "prefill_step_count": 0,
            "decode_step_count": 0,
            "mixed_lora_prefill_steps": 0,
            "mixed_lora_decode_steps": 0,
            "prefill_locality_rate": 0.0,
            "decode_locality_rate": 0.0,
        },
    }
def test_in_memory_observer_bounds_event_history() -> None:
    observer = InMemoryRuntimeObserver(history_limit=2)

    observer.on_request_aborted("first")
    observer.on_request_aborted("second")
    observer.on_request_aborted("third")

    assert observer.aborted == ["second", "third"]
