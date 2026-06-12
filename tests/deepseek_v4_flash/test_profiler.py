from __future__ import annotations

from collections.abc import Callable, Iterator

from pytest import MonkeyPatch

import vllm.model_executor.models.deepseek_v4_flash.profiler as profiler_module
from vllm.model_executor.models.deepseek_v4_flash.profiler import (
    DeepSeekV4FlashProfileEvent,
    DeepSeekV4FlashProfiler,
)


def test_profiler_records_section_elapsed_ms_and_metadata(
    monkeypatch: MonkeyPatch,
) -> None:
    times = iter([10.0, 10.125])
    monkeypatch.setattr(profiler_module, "perf_counter", _next_time(times))
    profiler = DeepSeekV4FlashProfiler(enabled=True)

    with profiler.section("stage_matrix", tensor="blk.0.attn_q.weight"):
        pass

    data = profiler.to_dict()

    assert data["enabled"] is True
    assert data["events"][0]["name"] == "stage_matrix"
    assert data["events"][0]["elapsed_ms"] == 125.0
    assert data["events"][0]["metadata"] == {"tensor": "blk.0.attn_q.weight"}


def test_profiler_counter_accumulates_values() -> None:
    profiler = DeepSeekV4FlashProfiler(enabled=True)

    profiler.add_counter("staging_cache_hit", 1)
    profiler.add_counter("staging_cache_hit", 2)

    assert profiler.to_dict()["counters"]["staging_cache_hit"] == 3


def test_disabled_profiler_has_no_events_but_accepts_calls() -> None:
    sync_calls = 0

    def sync_fn() -> None:
        nonlocal sync_calls
        sync_calls += 1

    profiler = DeepSeekV4FlashProfiler(enabled=False, sync_fn=sync_fn)

    with profiler.section("ignored"):
        pass
    profiler.add_counter("ignored", 1)

    assert profiler.to_dict() == {
        "enabled": False,
        "events": [],
        "counters": {},
    }
    assert sync_calls == 0


def test_enabled_profiler_syncs_before_and_after_section() -> None:
    sync_calls = 0

    def sync_fn() -> None:
        nonlocal sync_calls
        sync_calls += 1

    profiler = DeepSeekV4FlashProfiler(enabled=True, sync_fn=sync_fn)

    with profiler.section("synced"):
        pass

    assert sync_calls == 2


def test_snapshot_can_drain_events_and_counters(monkeypatch: MonkeyPatch) -> None:
    times = iter([1.0, 1.25])
    monkeypatch.setattr(profiler_module, "perf_counter", _next_time(times))
    profiler = DeepSeekV4FlashProfiler(enabled=True)

    with profiler.section("forward"):
        pass
    profiler.add_counter("layers", 2)

    assert profiler.snapshot(reset=True) == {
        "enabled": True,
        "events": [
            {
                "name": "forward",
                "elapsed_ms": 250.0,
                "metadata": {},
            }
        ],
        "counters": {"layers": 2},
    }
    assert profiler.to_dict() == {
        "enabled": True,
        "events": [],
        "counters": {},
    }


def test_profile_event_serializes_without_mutating_metadata() -> None:
    event = DeepSeekV4FlashProfileEvent(
        name="forward_layer",
        elapsed_ms=1.25,
        metadata={"layer": 2},
    )

    serialized = event.to_dict()

    assert serialized == {
        "name": "forward_layer",
        "elapsed_ms": 1.25,
        "metadata": {"layer": 2},
    }
    serialized["metadata"]["layer"] = 3
    assert event.metadata == {"layer": 2}


def _next_time(times: Iterator[float]) -> Callable[[], float]:
    return lambda: next(times)
