# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from vllm.engine.errors import BackgroundLoopError
from vllm.logger import init_logger

logger = init_logger(__name__)


class AsyncDriver:
    """Background-thread driver for LiteEngine.step().

    The synchronous ``engine.step()`` call (including its CUDA synchronization
    and sampling CPU work) runs on a dedicated worker thread so that the asyncio
    event loop remains responsive to HTTP/SSE traffic. Outputs are still produced
    by ``engine.step()`` internally via ``scheduler.publish_output``; the
    scheduler is thread-safe and dispatches queue writes back to the event loop
    thread when called from the worker.
    """

    def __init__(
        self,
        engine: Any,
        *,
        min_step_interval_s: float = 0.001,
        sleep_fn: Callable[[float], None] = time.sleep,
        engine_lock: Any | None = None,
    ):
        self.engine = engine
        self.min_step_interval_s = max(0.0, float(min_step_interval_s))
        self._sleep = sleep_fn
        self._engine_lock = engine_lock or getattr(engine, "_async_engine_lock", None)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._work_queue: queue.Queue[bool] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._running = False
        self._steps = 0
        self._backpressure_sleeps = 0
        self._idle_waits = 0
        self._background_errors = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        scheduler = getattr(self.engine, "scheduler", None)
        if scheduler is not None and hasattr(scheduler, "_set_event_loop"):
            scheduler._set_event_loop(self._loop)
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._worker_thread.start()

    def notify_new_work(self) -> None:
        self.start()
        self._work_queue.put(True)

    def _run_loop(self) -> None:
        while self._running:
            with self._locked_engine():
                active_request_count = self.engine.active_request_count
            if active_request_count == 0:
                self._idle_waits += 1
                try:
                    self._work_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if not self._running:
                    break
                # A notification arrived; loop around and process if there is
                # still active work.
                continue

            try:
                with self._locked_engine():
                    self.engine.step()
                    active_request_count = self.engine.active_request_count
                self._steps += 1
            except Exception as exc:
                error = BackgroundLoopError(str(exc))
                self._background_errors += 1
                logger.exception("AsyncDriver worker thread error: %s", error)
                loop = self._loop
                if loop is not None and hasattr(self.engine, "handle_background_error"):
                    loop.call_soon_threadsafe(
                        self.engine.handle_background_error, error
                    )
                self._sleep(self.min_step_interval_s)
                continue

            if active_request_count > 0:
                self._backpressure_sleeps += 1
                self._sleep(self.min_step_interval_s)

        self._shutdown_event.set()

    def _locked_engine(self):
        if self._engine_lock is None:
            return contextlib.nullcontext()
        return self._engine_lock

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
        with contextlib.suppress(Exception):
            self._work_queue.put_nowait(False)
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        self._shutdown_event.set()
