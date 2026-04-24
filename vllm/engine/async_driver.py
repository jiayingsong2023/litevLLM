# SPDX-License-Identifier: Apache-2.0
import asyncio
from typing import Optional

from vllm.engine.errors import BackgroundLoopError
from vllm.logger import init_logger

logger = init_logger(__name__)


class AsyncDriver:
    """Event-driven wrapper around LiteEngine.step()."""

    def __init__(self, engine: object):
        self.engine = engine
        self._loop_task: Optional[asyncio.Task] = None
        self._running = False
        self._work_event = asyncio.Event()

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
                self._work_event.clear()
                await self._work_event.wait()
                continue

            try:
                outputs = self.engine.step()
                # Always yield to event loop between engine steps so streaming
                # consumers can observe incremental tokens instead of tail-batched
                # queue draining when the driver keeps stepping in a tight loop.
                if self.engine.active_request_count > 0:
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = BackgroundLoopError(str(exc))
                logger.exception("AsyncDriver loop error: %s", error)
                if hasattr(self.engine, "handle_background_error"):
                    self.engine.handle_background_error(error)
                await asyncio.sleep(0)

    def shutdown(self) -> None:
        self._running = False
        self._work_event.set()
        if self._loop_task is not None:
            self._loop_task.cancel()
