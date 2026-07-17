# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.engine import async_llm as async_llm_module
from vllm.engine.async_driver import AsyncDriver
from vllm.engine.errors import BackgroundLoopError
from vllm.engine.lite_engine import LiteEngine
from vllm.engine.request_scheduler import RequestScheduler
from vllm.engine.request_state import RequestState
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams


def _rs(request_id: str, **kwargs: Any) -> RequestState:
    return RequestState(
        request_id=request_id,
        prompt=kwargs.pop("prompt", ""),
        input_ids=kwargs.pop("input_ids", []),
        sampling_params=kwargs.pop("sampling_params", SamplingParams()),
        **kwargs,
    )


def _request_output(request_id: str, finished: bool) -> RequestOutput:
    return RequestOutput(
        request_id=request_id,
        prompt="p",
        prompt_token_ids=[1, 2],
        outputs=[
            CompletionOutput(
                index=0,
                text="x",
                token_ids=[3],
                cumulative_logprob=0.0,
            )
        ],
        finished=finished,
    )


@pytest.mark.asyncio
async def test_request_scheduler_stream_finishes_with_terminal_output() -> None:
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.add_request("r1", _rs("r1", slot_idx=0, is_prefill=True))
    scheduler.publish_output("r1", _request_output("r1", finished=False))
    scheduler.publish_output("r1", _request_output("r1", finished=True))
    scheduler.free_request("r1")

    outputs = []
    async for output in scheduler.get_request_stream("r1"):
        outputs.append(output)

    assert len(outputs) == 2
    assert outputs[-1].finished is True
    assert "r1" not in scheduler._request_streams


@pytest.mark.asyncio
async def test_request_scheduler_stream_raises_published_exception() -> None:
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.add_request("r1", _rs("r1", slot_idx=0, is_prefill=True))
    scheduler.publish_exception("r1", RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in scheduler.get_request_stream("r1"):
            pass
    assert "r1" not in scheduler._request_streams


def test_lite_engine_background_error_publishes_then_releases_every_request() -> None:
    published: list[tuple[str, BaseException]] = []
    released: list[str] = []
    observed: list[tuple[BaseException, list[str]]] = []
    engine = SimpleNamespace(
        _fatal_error=None,
        scheduler=SimpleNamespace(
            request_ids=lambda: ["r1", "r2"],
            publish_exception=lambda request_id, exc: published.append((request_id, exc)),
        ),
        execution_backend=SimpleNamespace(
            release_request=lambda request_id: released.append(request_id),
        ),
        observer=SimpleNamespace(
            on_background_error=lambda exc, request_ids: observed.append(
                (exc, request_ids)
            ),
        ),
    )

    LiteEngine.handle_background_error(engine, RuntimeError("step failed"))
    LiteEngine.handle_background_error(engine, RuntimeError("later failure"))

    assert isinstance(engine._fatal_error, BackgroundLoopError)
    assert "step failed" in str(engine._fatal_error)
    assert [request_id for request_id, _ in published] == ["r1", "r2"]
    assert all(exc is engine._fatal_error for _, exc in published)
    assert released == ["r1", "r2"]
    assert observed == [(engine._fatal_error, ["r1", "r2"])]


def test_request_scheduler_rejects_duplicate_id_until_stream_is_detached() -> None:
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.add_request("r1", _rs("r1", slot_idx=0, is_prefill=True))
    scheduler.free_request("r1")

    with pytest.raises(ValueError, match="already active"):
        scheduler.enqueue_request("r1", _rs("r1", is_prefill=True))


def test_request_scheduler_enqueue_and_admit_releases_queue() -> None:
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.add_request("r1", _rs("r1", slot_idx=0, is_prefill=True))
    scheduler.enqueue_request(
        "r2", _rs("r2", is_prefill=True, seq_len=0, input_ids=[1, 2])
    )

    assert scheduler.active_request_count == 2
    assert scheduler.running_request_count == 1
    assert scheduler.queued_request_count == 1
    assert scheduler.queued_ids == ["r2"]

    scheduler.free_request("r1")
    admitted = scheduler.admit_queued_requests()

    assert admitted == ["r2"]
    assert scheduler.running_ids == ["r2"]
    assert scheduler.queued_request_count == 0
    assert scheduler.get_request("r2").slot_idx == 0


def test_request_scheduler_abort_removes_queued_request() -> None:
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.enqueue_request("r1", _rs("r1", is_prefill=True))

    assert scheduler.queued_ids == ["r1"]
    scheduler.abort_request("r1")

    assert scheduler.active_request_count == 0
    assert scheduler.queued_request_count == 0


def test_request_scheduler_rejects_expired_queued_requests() -> None:
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.enqueue_request("r1", _rs("r1", is_prefill=True, queued_at=1.0))
    scheduler.enqueue_request("r2", _rs("r2", is_prefill=True, queued_at=9.5))

    expired = scheduler.reject_expired_queued_requests(now=10.0, max_queue_wait_s=5.0)

    assert len(expired) == 1
    assert expired[0][0] == "r1"
    assert expired[0][2].queued_at == 1.0
    assert scheduler.queued_ids == ["r1", "r2"]


def test_async_driver_default_backpressure_interval_is_positive() -> None:
    class FakeEngine:
        active_request_count = 0

        def step(self):
            return []

    driver = AsyncDriver(FakeEngine())

    assert driver.stats()["min_step_interval_s"] > 0.0


@pytest.mark.asyncio
async def test_async_driver_applies_positive_backpressure_interval() -> None:
    sleep_delays: list[float] = []
    done_event = asyncio.Event()

    def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)
        if len(sleep_delays) >= 2:
            engine.active_request_count = 0
            done_event.set()

    class FakeEngine:
        def __init__(self) -> None:
            self.active_request_count = 1
            self.calls = 0

        def step(self):
            self.calls += 1
            return []

    engine = FakeEngine()
    driver = AsyncDriver(
        engine,
        min_step_interval_s=0.002,
        sleep_fn=fake_sleep,
    )
    driver.notify_new_work()
    await asyncio.wait_for(done_event.wait(), timeout=5.0)
    driver.shutdown()

    assert engine.calls >= 2
    assert sleep_delays[:2] == [0.002, 0.002]
    assert driver.stats()["steps"] >= 2
    assert driver.stats()["backpressure_sleeps"] >= 2


@pytest.mark.asyncio
async def test_async_driver_propagates_background_error_to_engine() -> None:
    error_event = threading.Event()

    class FakeEngine:
        def __init__(self) -> None:
            self.active_request_count = 1
            self.calls = 0
            self.error = None

        def step(self):
            self.calls += 1
            raise RuntimeError("driver-failure")

        def handle_background_error(self, exc: BaseException) -> None:
            self.error = exc
            self.active_request_count = 0
            error_event.set()

    engine = FakeEngine()
    driver = AsyncDriver(engine)
    driver.notify_new_work()
    assert await asyncio.to_thread(error_event.wait, 5.0)
    driver.shutdown()

    assert engine.calls >= 1
    assert isinstance(engine.error, RuntimeError)
    assert "driver-failure" in str(engine.error)


def test_async_llm_passes_runtime_backpressure_interval_to_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEngine:
        def __init__(self, _vllm_config) -> None:
            self.tokenizer = None

        def set_tokenizer(self, tokenizer) -> None:
            self.tokenizer = tokenizer

    class FakeDriver:
        def __init__(self, engine, *, min_step_interval_s: float) -> None:
            self.engine = engine
            self.min_step_interval_s = min_step_interval_s

        def shutdown(self) -> None:
            return None

    class DummyConfig:
        class ModelConfig:
            model = "dummy"

        model_config = ModelConfig()
        runtime_config = type(
            "RuntimeConfig", (), {"async_driver_min_step_interval_s": 0.007}
        )()

    monkeypatch.setattr(async_llm_module, "LiteEngine", FakeEngine)
    monkeypatch.setattr(async_llm_module, "AsyncDriver", FakeDriver)
    monkeypatch.setattr(
        async_llm_module, "get_tokenizer", lambda *_args, **_kwargs: object()
    )

    llm = async_llm_module.AsyncLLM(DummyConfig())

    assert llm.driver.min_step_interval_s == 0.007


def test_async_llm_stats_include_async_driver_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEngine:
        def __init__(self, _vllm_config) -> None:
            self.tokenizer = None

        def set_tokenizer(self, tokenizer) -> None:
            self.tokenizer = tokenizer

        def stats(self) -> dict[str, object]:
            return {"scheduler": {"active_request_count": 1}}

    class FakeDriver:
        def __init__(self, engine, *, min_step_interval_s: float = 0.001) -> None:
            del min_step_interval_s
            self.engine = engine

        def stats(self) -> dict[str, object]:
            return {"steps": 3, "backpressure_sleeps": 2}

        def shutdown(self) -> None:
            return None

    class DummyConfig:
        class ModelConfig:
            model = "dummy"

        model_config = ModelConfig()

    monkeypatch.setattr(async_llm_module, "LiteEngine", FakeEngine)
    monkeypatch.setattr(async_llm_module, "AsyncDriver", FakeDriver)
    monkeypatch.setattr(
        async_llm_module, "get_tokenizer", lambda *_args, **_kwargs: object()
    )

    llm = async_llm_module.AsyncLLM(DummyConfig())

    assert llm.stats() == {
        "scheduler": {"active_request_count": 1},
        "async_driver": {"steps": 3, "backpressure_sleeps": 2},
    }


def test_async_llm_serializes_all_engine_state_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TrackingLock:
        def __init__(self) -> None:
            self.depth = 0

        def __enter__(self):
            self.depth += 1
            return self

        def __exit__(self, *_args) -> None:
            self.depth -= 1

    class FakeEngine:
        def __init__(self, _vllm_config) -> None:
            self._async_engine_lock = None

        def set_tokenizer(self, _tokenizer) -> None:
            return None

        def _assert_locked(self) -> None:
            assert self._async_engine_lock.depth > 0

        def abort_request(self, _request_id) -> None:
            self._assert_locked()

        def stats(self) -> dict[str, object]:
            self._assert_locked()
            return {}

        def register_lora_adapter(self, **_kwargs) -> dict[str, object]:
            self._assert_locked()
            return {}

        def unregister_lora_adapter(self, _lora_name: str) -> bool:
            self._assert_locked()
            return True

        def reset_stats(self, **_kwargs) -> None:
            self._assert_locked()

    class FakeDriver:
        def __init__(self, _engine, *, min_step_interval_s: float = 0.001) -> None:
            del min_step_interval_s

        def stats(self) -> dict[str, object]:
            return {}

        def shutdown(self) -> None:
            return None

    class DummyConfig:
        class ModelConfig:
            model = "dummy"

        model_config = ModelConfig()

    monkeypatch.setattr(async_llm_module, "LiteEngine", FakeEngine)
    monkeypatch.setattr(async_llm_module, "AsyncDriver", FakeDriver)
    monkeypatch.setattr(
        async_llm_module, "get_tokenizer", lambda *_args, **_kwargs: object()
    )
    llm = async_llm_module.AsyncLLM(DummyConfig())
    lock = TrackingLock()
    llm._engine_lock = lock
    llm.engine._async_engine_lock = lock

    asyncio.run(llm.abort("r1"))
    llm.stats()
    llm.register_lora_adapter(lora_name="adapter")
    assert llm.unregister_lora_adapter("adapter") is True
    llm.reset_stats()


@pytest.mark.asyncio
async def test_async_llm_generate_abort_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEngine:
        def __init__(self, _vllm_config) -> None:
            self.tokenizer = None

        def set_tokenizer(self, tokenizer) -> None:
            self.tokenizer = tokenizer
            self.streams: dict[str, asyncio.Queue] = {}

        def add_request(
            self,
            request_id: str,
            prompt: str,
            sampling_params: SamplingParams,
            lora_id=None,
            lora_request=None,
        ) -> None:
            del prompt, sampling_params, lora_id, lora_request
            self.streams[request_id] = asyncio.Queue()

        async def get_request_stream(self, request_id: str):
            queue = self.streams[request_id]
            while True:
                output = await queue.get()
                yield output
                if output.finished:
                    break

        def abort_request(self, request_id: str) -> None:
            self.streams[request_id].put_nowait(
                _request_output(request_id, finished=True)
            )

    class FakeDriver:
        def __init__(self, engine, *, min_step_interval_s: float = 0.001) -> None:
            del min_step_interval_s
            self.engine = engine
            self.notified = 0

        def notify_new_work(self) -> None:
            asyncio.get_running_loop()
            self.notified += 1

        def shutdown(self) -> None:
            return None

    class DummyConfig:
        class ModelConfig:
            model = "dummy"

        model_config = ModelConfig()

    monkeypatch.setattr(async_llm_module, "LiteEngine", FakeEngine)
    monkeypatch.setattr(async_llm_module, "AsyncDriver", FakeDriver)
    monkeypatch.setattr(
        async_llm_module, "get_tokenizer", lambda *_args, **_kwargs: object()
    )

    llm = async_llm_module.AsyncLLM(DummyConfig())
    await llm.submit("hi", SamplingParams(max_tokens=4), "req-1")
    agen = llm.stream("req-1")
    abort_task = asyncio.create_task(llm.abort("req-1"))
    output = await agen.__anext__()
    await abort_task

    assert output.request_id == "req-1"
    assert output.finished is True
    assert llm.driver.notified == 1


@pytest.mark.asyncio
async def test_async_llm_stream_cancellation_aborts_admitted_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEngine:
        def __init__(self, _vllm_config) -> None:
            self.tokenizer = None
            self.aborted: list[str] = []

        def set_tokenizer(self, tokenizer) -> None:
            self.tokenizer = tokenizer

        def add_request(self, *_args, **_kwargs) -> None:
            return None

        async def get_request_stream(self, _request_id: str):
            await asyncio.Event().wait()
            yield _request_output("unreachable", finished=True)

        def abort_request(self, request_id: str) -> None:
            self.aborted.append(request_id)

    class FakeDriver:
        def __init__(self, *_args, **_kwargs) -> None:
            return None

        def notify_new_work(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

    class DummyConfig:
        class ModelConfig:
            model = "dummy"

        model_config = ModelConfig()

    monkeypatch.setattr(async_llm_module, "LiteEngine", FakeEngine)
    monkeypatch.setattr(async_llm_module, "AsyncDriver", FakeDriver)
    monkeypatch.setattr(
        async_llm_module, "get_tokenizer", lambda *_args, **_kwargs: object()
    )

    llm = async_llm_module.AsyncLLM(DummyConfig())
    await llm.submit("hi", SamplingParams(max_tokens=1), "req-cancel")
    stream = llm.stream("req-cancel")
    pending = asyncio.create_task(stream.__anext__())
    await asyncio.sleep(0)
    pending.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await pending

    assert llm.engine.aborted == ["req-cancel"]


@pytest.mark.asyncio
async def test_async_llm_generate_passes_lora_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEngine:
        def __init__(self, _vllm_config) -> None:
            self.tokenizer = None

        def set_tokenizer(self, tokenizer) -> None:
            self.tokenizer = tokenizer
            self.calls = []
            self.streams: dict[str, asyncio.Queue] = {}

        def add_request(
            self,
            request_id: str,
            prompt: str,
            sampling_params: SamplingParams,
            lora_id=None,
            lora_request=None,
        ) -> None:
            self.calls.append(
                (request_id, prompt, sampling_params, lora_id, lora_request)
            )
            queue = asyncio.Queue()
            queue.put_nowait(_request_output(request_id, finished=True))
            self.streams[request_id] = queue

        async def get_request_stream(self, request_id: str):
            queue = self.streams[request_id]
            while True:
                output = await queue.get()
                yield output
                if output.finished:
                    break

    class FakeDriver:
        def __init__(self, engine, *, min_step_interval_s: float = 0.001) -> None:
            del min_step_interval_s
            self.engine = engine

        def notify_new_work(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

    class DummyConfig:
        class ModelConfig:
            model = "dummy"

        model_config = ModelConfig()

    monkeypatch.setattr(async_llm_module, "LiteEngine", FakeEngine)
    monkeypatch.setattr(async_llm_module, "AsyncDriver", FakeDriver)
    monkeypatch.setattr(
        async_llm_module, "get_tokenizer", lambda *_args, **_kwargs: object()
    )

    llm = async_llm_module.AsyncLLM(DummyConfig())
    agen = llm.generate(
        "hi",
        SamplingParams(max_tokens=4),
        "req-1",
        lora_request=LoRARequest(
            lora_name="adapter-a", lora_int_id=3, lora_path="/tmp/a"
        ),
    )
    output = await agen.__anext__()

    assert output.request_id == "req-1"
    assert llm.engine.calls[0][3] == "adapter-a"
    assert isinstance(llm.engine.calls[0][4], LoRARequest)
