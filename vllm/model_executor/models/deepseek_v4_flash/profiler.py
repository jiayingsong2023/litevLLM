# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any


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
