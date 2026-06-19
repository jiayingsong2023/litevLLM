# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any

_COMPACT_PHASE_NAMES = (
    "layer_attention",
    "layer_moe",
    "router_selected_experts_kernel",
    "router_expert_stage",
    "compressed_kv_update",
    "compressed_indexer_update",
    "output_projection",
)


@dataclass(frozen=True)
class DeepSeekV4FlashProfileEvent:
    name: str
    elapsed_ms: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elapsed_ms": self.elapsed_ms,
            "metadata": dict(self.metadata),
        }


class DeepSeekV4FlashProfiler:
    def __init__(
        self,
        *,
        enabled: bool = False,
        sync_fn: Callable[[], None] | None = None,
    ) -> None:
        self.enabled = enabled
        self._sync_fn = sync_fn
        self._events: list[DeepSeekV4FlashProfileEvent] = []
        self._counters: dict[str, int] = {}

    @contextmanager
    def section(self, name: str, **metadata: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if self._sync_fn is not None:
            self._sync_fn()
        start = perf_counter()
        try:
            yield
        finally:
            if self._sync_fn is not None:
                self._sync_fn()
            elapsed_ms = (perf_counter() - start) * 1000.0
            self._events.append(
                DeepSeekV4FlashProfileEvent(
                    name=name,
                    elapsed_ms=elapsed_ms,
                    metadata=dict(metadata),
                )
            )

    def record(
        self,
        name: str,
        elapsed_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._events.append(
            DeepSeekV4FlashProfileEvent(
                name=name,
                elapsed_ms=float(elapsed_ms),
                metadata=dict(metadata or {}),
            )
        )

    def add_counter(self, name: str, value: int = 1) -> None:
        if not self.enabled:
            return
        self._counters[name] = self._counters.get(name, 0) + int(value)

    def reset(self) -> None:
        self._events.clear()
        self._counters.clear()

    def _aggregate_by_name(self) -> dict[str, dict[str, float | int]]:
        aggregates: dict[str, dict[str, float | int]] = {}
        for event in self._events:
            aggregate = aggregates.setdefault(
                event.name,
                {
                    "count": 0,
                    "total_ms": 0.0,
                    "avg_ms": 0.0,
                    "max_ms": 0.0,
                },
            )
            count = int(aggregate["count"]) + 1
            total_ms = float(aggregate["total_ms"]) + event.elapsed_ms
            aggregate["count"] = count
            aggregate["total_ms"] = total_ms
            aggregate["avg_ms"] = total_ms / count
            aggregate["max_ms"] = max(float(aggregate["max_ms"]), event.elapsed_ms)
        return aggregates

    def compact_summary(self, top_k: int = 10) -> dict[str, Any]:
        aggregates = self._aggregate_by_name()
        sorted_events = sorted(
            aggregates.items(),
            key=lambda item: (-float(item[1]["total_ms"]), item[0]),
        )
        top_events = [
            {
                "name": name,
                "total_ms": float(aggregate["total_ms"]),
                "avg_ms": float(aggregate["avg_ms"]),
                "count": int(aggregate["count"]),
            }
            for name, aggregate in sorted_events[: max(top_k, 0)]
        ]
        phase_totals_ms = {
            phase_name: float(aggregates.get(phase_name, {}).get("total_ms", 0.0))
            for phase_name in _COMPACT_PHASE_NAMES
        }
        return {
            "top_events": top_events,
            "phase_totals_ms": phase_totals_ms,
        }

    def snapshot(self, reset: bool = False) -> dict[str, Any]:
        data = {
            "enabled": self.enabled,
            "events": [event.to_dict() for event in self._events],
            "counters": dict(self._counters),
        }
        if self.enabled:
            data["aggregate_by_name"] = self._aggregate_by_name()
        if reset:
            self.reset()
        return data

    def to_dict(self) -> dict[str, Any]:
        return self.snapshot(reset=False)
