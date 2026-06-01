# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from vllm.engine.errors import BackgroundLoopError
from vllm.logger import init_logger

logger = init_logger(__name__)


class AsyncDriver:
    """Event-driven wrapper around LiteEngine.step()."""

    def __init__(
        self,
        engine: object,
        *,
        min_step_interval_s: float = 0.001,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.engine = engine
        self.min_step_interval_s = max(0.0, float(min_step_interval_s))
        self._sleep = sleep_fn
        self._loop_task: asyncio.Task | None = None
        self._running = False
        self._work_event = asyncio.Event()
        self._steps = 0
        self._backpressure_sleeps = 0
        self._idle_waits = 0
        self._background_errors = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())

    def notify_new_work(self) -> None:
        self.start()
        self._work_event.set()

    async def _run_loop(self) -> None:
        while self._running:
            if self.engine.active_request_count == 0:
                self._idle_waits += 1
                self._work_event.clear()
                await self._work_event.wait()
                continue

            try:
                outputs = self.engine.step()
                del outputs
                self._steps += 1
                # Yield with optional positive backpressure between active steps so
                # streaming consumers can run without letting the driver spin in a
                # pure event-loop busy loop under sustained decode load.
                if self.engine.active_request_count > 0:
                    self._backpressure_sleeps += 1
                    await self._sleep(self.min_step_interval_s)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = BackgroundLoopError(str(exc))
                self._background_errors += 1
                logger.exception("AsyncDriver loop error: %s", error)
                if hasattr(self.engine, "handle_background_error"):
                    self.engine.handle_background_error(error)
                await self._sleep(self.min_step_interval_s)

    def stats(self) -> dict[str, int | float]:
        return {
            "steps": self._steps,
            "backpressure_sleeps": self._backpressure_sleeps,
            "idle_waits": self._idle_waits,
            "background_errors": self._background_errors,
            "min_step_interval_s": self.min_step_interval_s,
        }

    def shutdown(self) -> None:
        self._running = False
        self._work_event.set()
        if self._loop_task is not None:
            self._loop_task.cancel()
