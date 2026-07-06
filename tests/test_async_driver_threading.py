# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from vllm.engine.async_driver import AsyncDriver


@pytest.mark.asyncio
async def test_async_driver_runs_step_in_background_thread() -> None:
    step_started = threading.Event()
    step_finished = threading.Event()

    class FakeEngine:
        def __init__(self) -> None:
            self.active_request_count = 1
            self.calls = 0
            self.worker_thread_name: str | None = None

        def step(self) -> list[object]:
            self.calls += 1
            self.worker_thread_name = threading.current_thread().name
            step_started.set()
            time.sleep(0.05)
            self.active_request_count = 0
            step_finished.set()
            return []

    engine = FakeEngine()
    driver = AsyncDriver(engine)
    driver.notify_new_work()

    step_started.wait(timeout=2.0)
    step_finished.wait(timeout=2.0)
    driver.shutdown()

    assert engine.calls == 1
    assert engine.worker_thread_name != threading.current_thread().name


@pytest.mark.asyncio
async def test_async_driver_serializes_step_with_shared_engine_lock() -> None:
    step_started = threading.Event()

    class FakeEngine:
        def __init__(self) -> None:
            self.active_request_count = 1
            self.calls = 0

        def step(self) -> list[object]:
            self.calls += 1
            step_started.set()
            self.active_request_count = 0
            return []

    engine = FakeEngine()
    engine_lock = threading.RLock()
    driver = AsyncDriver(engine, engine_lock=engine_lock)

    with engine_lock:
        driver.notify_new_work()
        await asyncio.sleep(0.05)
        assert step_started.is_set() is False

    assert step_started.wait(timeout=2.0)
    driver.shutdown()

    assert engine.calls == 1


@pytest.mark.asyncio
async def test_async_driver_does_not_block_event_loop_during_step() -> None:
    step_started = threading.Event()
    step_finished = threading.Event()

    class FakeEngine:
        def __init__(self) -> None:
            self.active_request_count = 1
            self.calls = 0

        def step(self) -> list[object]:
            self.calls += 1
            step_started.set()
            time.sleep(0.05)
            self.active_request_count = 0
            step_finished.set()
            return []

    engine = FakeEngine()
    driver = AsyncDriver(engine)
    driver.notify_new_work()

    step_started.wait(timeout=2.0)

    # While the worker thread is inside the blocking step(), the event loop
    # must still be able to execute a tiny coroutine promptly.
    loop_responsive = False

    async def probe() -> None:
        nonlocal loop_responsive
        loop_responsive = True

    await asyncio.wait_for(probe(), timeout=0.01)

    step_finished.wait(timeout=2.0)
    driver.shutdown()

    assert engine.calls == 1
    assert loop_responsive is True


@pytest.mark.asyncio
async def test_async_driver_notifies_worker_after_idle() -> None:
    step_finished = asyncio.Event()

    class FakeEngine:
        def __init__(self) -> None:
            self.active_request_count = 0
            self.calls = 0

        def step(self) -> list[object]:
            self.calls += 1
            self.active_request_count = 0
            step_finished.set()
            return []

    engine = FakeEngine()
    driver = AsyncDriver(engine)
    driver.notify_new_work()

    # Notify arrives while idle; the worker should wake and execute step once
    # active_request_count is positive.
    engine.active_request_count = 1
    driver.notify_new_work()

    await asyncio.wait_for(step_finished.wait(), timeout=2.0)
    driver.shutdown()

    assert engine.calls >= 1
