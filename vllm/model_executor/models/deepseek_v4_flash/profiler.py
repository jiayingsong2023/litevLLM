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

    def snapshot(self, reset: bool = False) -> dict[str, Any]:
        data = {
            "enabled": self.enabled,
            "events": [event.to_dict() for event in self._events],
            "counters": dict(self._counters),
        }
        if reset:
            self.reset()
        return data

    def to_dict(self) -> dict[str, Any]:
        return self.snapshot(reset=False)
