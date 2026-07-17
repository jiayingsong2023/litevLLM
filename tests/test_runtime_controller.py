# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.engine.errors import RequestRejectedError
from vllm.engine.request_state import RequestState
from vllm.engine.runtime_controller import RuntimeController
from vllm.engine.runtime_observer import InMemoryRuntimeObserver
from vllm.engine.step_plan import (
    AdmissionPlan,
    DecodePlan,
    StepPlan,
    StepPlanMetrics,
)


class _FakeScheduler:
    def __init__(self) -> None:
        self.active_request_count = 2
        self.running_request_count = 0
        self._queued = ["q1"]
        self.admitted = []
        self.req = RequestState(
            request_id="q1",
            prompt="",
            input_ids=[],
            sampling_params=type("SP", (), {"temperature": 0, "max_tokens": 1})(),
            queued_at=10.0,
            admitted_at=None,
            first_token_at=None,
            generated_ids=[],
            seq_len=0,
            finished=False,
        )

    def reject_expired_queued_requests(self, *, now: float, max_queue_wait_s: float):
        del now, max_queue_wait_s
        return []

    def admit_specific_requests(self, request_ids, *, admitted_at):
        self.admitted.append((list(request_ids), admitted_at))
        self.running_request_count = len(request_ids)
        self.req.admitted_at = admitted_at
        return list(request_ids)

    def get_request(self, _request_id: str):
        return self.req

    def publish_exception(self, request_id, exc):
        raise AssertionError(f"unexpected exception publish {request_id}: {exc}")

    def publish_output(self, request_id, output):
        self.published = (request_id, output)

    def free_request(self, request_id):
        self.freed = request_id


class _FakeStepScheduler:
    def build_plan(self, scheduler):
        return StepPlan(
            admissions=AdmissionPlan(request_ids=["q1"]),
            prefills=None,
            decodes=DecodePlan(request_ids=["q1"], token_budget=1, use_fast_path=True),
            step_token_budget=4,
            metrics=StepPlanMetrics(
                queued_before=1,
                running_before=scheduler.running_request_count,
            ),
        )


class _FakeObserver:
    def __init__(self) -> None:
        self.admitted = []
        self.steps = []
        self.first_tokens = []
        self.finished = []

    def on_request_admitted(self, request_id: str, queue_wait_s: float) -> None:
        self.admitted.append((request_id, queue_wait_s))

    def on_step_started(self, plan) -> None:
        self.steps.append(plan)

    def on_first_token(self, request_id: str, ttft_s: float) -> None:
        self.first_tokens.append((request_id, ttft_s))

    def on_request_finished(self, request_id: str, reason: str) -> None:
        self.finished.append((request_id, reason))

    def on_request_rejected(self, request_id: str, reason: str) -> None:
        self.rejected = getattr(self, "rejected", [])
        self.rejected.append((request_id, reason))

    def stats(self) -> dict[str, object]:
        return {
            "step_count": len(self.steps),
            "admitted_requests": len(self.admitted),
        }

    def reset_stats(self) -> None:
        self.steps.clear()
        self.admitted.clear()


class _FakeBackend:
    def __init__(self) -> None:
        self.fast_path_calls = []

    def maybe_apply_prefix_cache(self, request_state):
        self.prefix_cache_seen = request_state

    def ensure_kv_blocks(self, _step_plan) -> None:
        return None

    def decode_step_sync(self, request_ids):
        self.fast_path_calls.append(list(request_ids))
        return [{"request_id": request_ids[0], "finished": True}]

    def run_prefills(self, step_plan, results):
        del step_plan, results
        raise AssertionError("prefill path should not be used in this test")

    def run_decodes(self, step_plan, results):
        del step_plan, results
        raise AssertionError("decode batch path should not be used in this test")

    def release_request(self, _request_id: str):
        self.released = getattr(self, "released", [])
        self.released.append(_request_id)
        return None

    def stats(self) -> dict[str, object]:
        return {
            "backend_type": "fake",
            "prefix_cache": {"entries": 0, "capacity": 0},
        }

    def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
        self.reset_call = clear_prefix_cache


class _FakeLoRARegistry:
    def stats(self) -> dict[str, object]:
        return {
            "registered_adapters": 1,
            "active_adapters": 1,
            "active_requests": 1,
            "total_routed_requests": 2,
            "adapters": {
                "demo": {
                    "lora_int_id": 7,
                    "lora_path": "/tmp/demo",
                    "active_requests": 1,
                    "total_requests": 2,
                }
            },
        }

    def on_request_removed(self, _lora_name: str | None) -> None:
        return None


def test_runtime_controller_admits_then_uses_decode_fast_path(monkeypatch) -> None:
    scheduler = _FakeScheduler()
    observer = _FakeObserver()
    backend = _FakeBackend()
    controller = RuntimeController(
        scheduler=scheduler,
        step_scheduler=_FakeStepScheduler(),
        observer=observer,
        backend=backend,
        queue_timeout_s=30.0,
    )
    monkeypatch.setattr(
        "vllm.engine.runtime_controller.time.perf_counter", lambda: 12.0
    )

    outputs = controller.step()

    assert outputs == [{"request_id": "q1", "finished": True}]
    assert scheduler.admitted == [(["q1"], 12.0)]
    assert observer.admitted == [("q1", 2.0)]
    assert len(observer.steps) == 1
    assert backend.prefix_cache_seen is scheduler.req
    assert backend.fast_path_calls == [["q1"]]


def test_runtime_controller_stats_snapshot() -> None:
    scheduler = _FakeScheduler()
    scheduler.running_request_count = 1
    observer = _FakeObserver()
    backend = _FakeBackend()
    controller = RuntimeController(
        scheduler=scheduler,
        step_scheduler=_FakeStepScheduler(),
        observer=observer,
        backend=backend,
        queue_timeout_s=15.0,
        lora_registry=_FakeLoRARegistry(),
    )

    stats = controller.stats()

    assert stats["queue_timeout_s"] == 15.0
    assert isinstance(stats["observed_at_unix_s"], float)
    assert isinstance(stats["stats_window_started_at_unix_s"], float)
    assert isinstance(stats["stats_window_elapsed_s"], float)
    assert stats["scheduler"] == {
        "active_request_count": 2,
        "running_request_count": 1,
        "queued_request_count": 0,
        "available_slots": 0,
    }
    assert stats["observer"] == {
        "step_count": 0,
        "admitted_requests": 0,
    }
    assert stats["backend"] == {
        "backend_type": "fake",
        "prefix_cache": {"entries": 0, "capacity": 0},
    }
    assert stats["lora"] == {
        "registered_adapters": 1,
        "active_adapters": 1,
        "active_requests": 1,
        "total_routed_requests": 2,
        "adapters": {
            "demo": {
                "lora_int_id": 7,
                "lora_path": "/tmp/demo",
                "active_requests": 1,
                "total_requests": 2,
            }
        },
    }


def test_runtime_controller_timeout_publishes_before_releasing() -> None:
    scheduler = _FakeScheduler()
    scheduler.active_request_count = 1
    scheduler.running_request_count = 0
    events: list[tuple[str, str]] = []

    def reject_expired(*, now: float, max_queue_wait_s: float):
        del now, max_queue_wait_s
        return [("q1", "queue timeout", scheduler.req)]

    scheduler.reject_expired_queued_requests = reject_expired
    scheduler.publish_exception = lambda request_id, exc: events.append(
        (request_id, str(exc))
    )
    observer = _FakeObserver()
    backend = _FakeBackend()
    controller = RuntimeController(
        scheduler=scheduler,
        step_scheduler=_FakeStepScheduler(),
        observer=observer,
        backend=backend,
        queue_timeout_s=1.0,
    )

    controller._reject_expired_queued_requests(now=12.0)

    assert events == [("q1", "queue timeout")]
    assert backend.released == ["q1"]
    assert observer.rejected == [("q1", "queue timeout")]


def test_runtime_controller_reset_stats_resets_window_and_dependencies(
    monkeypatch,
) -> None:
    scheduler = _FakeScheduler()
    observer = _FakeObserver()
    backend = _FakeBackend()
    controller = RuntimeController(
        scheduler=scheduler,
        step_scheduler=_FakeStepScheduler(),
        observer=observer,
        backend=backend,
        queue_timeout_s=15.0,
    )
    times = iter([100.0, 100.0, 101.0, 101.0])
    monkeypatch.setattr("vllm.engine.runtime_controller.time.time", lambda: next(times))
    monkeypatch.setattr(
        "vllm.engine.runtime_controller.time.perf_counter", lambda: next(times)
    )

    controller.reset_stats(clear_prefix_cache=True)
    stats = controller.stats()

    assert backend.reset_call is True
    assert stats["observed_at_unix_s"] == 101.0
    assert stats["stats_window_started_at_unix_s"] == 100.0
    assert stats["stats_window_elapsed_s"] == 1.0


class _TimeoutScheduler:
    def __init__(self) -> None:
        self.active_request_count = 1
        self.running_request_count = 0
        self.queued_request_count = 1
        self.available_slots = 0
        self.published = []
        self.expired_request = RequestState(
            request_id="q-timeout",
            prompt="",
            input_ids=[],
            sampling_params=type("SP", (), {"temperature": 0, "max_tokens": 1})(),
            queued_at=1.0,
            lora_id="adapter-a",
        )

    def reject_expired_queued_requests(self, *, now: float, max_queue_wait_s: float):
        assert now == 10.0
        assert max_queue_wait_s == 5.0
        self.active_request_count = 0
        self.queued_request_count = 0
        return [
            (
                "q-timeout",
                "queue timeout after 9.000s (limit=5.000s)",
                self.expired_request,
            )
        ]

    def publish_exception(self, request_id, exc):
        self.published.append((request_id, exc))


class _UnexpectedStepScheduler:
    def build_plan(self, scheduler):
        del scheduler
        raise AssertionError("expired queued request should stop the step")


def test_runtime_controller_queue_timeout_is_observable(monkeypatch) -> None:
    scheduler = _TimeoutScheduler()
    observer = InMemoryRuntimeObserver()
    lora_registry = _FakeLoRARegistry()
    controller = RuntimeController(
        scheduler=scheduler,
        step_scheduler=_UnexpectedStepScheduler(),
        observer=observer,
        backend=_FakeBackend(),
        queue_timeout_s=5.0,
        lora_registry=lora_registry,
    )
    monkeypatch.setattr(
        "vllm.engine.runtime_controller.time.perf_counter", lambda: 10.0
    )

    outputs = controller.step()

    assert outputs == []
    assert observer.rejected == [
        ("q-timeout", "queue timeout after 9.000s (limit=5.000s)")
    ]
    assert observer.stats()["rejections"] == {
        "reasons": {"queue_timeout": 1},
        "queue_timeout": 1,
    }
    assert scheduler.published
    assert scheduler.published[0][0] == "q-timeout"
    assert isinstance(scheduler.published[0][1], RequestRejectedError)
