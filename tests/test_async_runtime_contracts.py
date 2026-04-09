# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest

from vllm.engine import async_llm as async_llm_module
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.engine.async_driver import AsyncDriver
from vllm.engine.request_scheduler import RequestScheduler


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
    scheduler.add_request("r1", {"slot_idx": 0, "is_prefill": True})
    scheduler.publish_output("r1", _request_output("r1", finished=False))
    scheduler.publish_output("r1", _request_output("r1", finished=True))

    outputs = []
    async for output in scheduler.get_request_stream("r1"):
        outputs.append(output)

    assert len(outputs) == 2
    assert outputs[-1].finished is True


@pytest.mark.asyncio
async def test_request_scheduler_stream_raises_published_exception() -> None:
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.add_request("r1", {"slot_idx": 0, "is_prefill": True})
    scheduler.publish_exception("r1", RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in scheduler.get_request_stream("r1"):
            pass


def test_request_scheduler_enqueue_and_admit_releases_queue() -> None:
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.add_request("r1", {"slot_idx": 0, "is_prefill": True})
    scheduler.enqueue_request("r2", {"is_prefill": True, "seq_len": 0, "input_ids": [1, 2]})

    assert scheduler.active_request_count == 2
    assert scheduler.running_request_count == 1
    assert scheduler.queued_request_count == 1
    assert scheduler.queued_ids == ["r2"]

    scheduler.free_request("r1")
    admitted = scheduler.admit_queued_requests()

    assert admitted == ["r2"]
    assert scheduler.running_ids == ["r2"]
    assert scheduler.queued_request_count == 0
    assert scheduler.get_request("r2")["slot_idx"] == 0


def test_request_scheduler_abort_removes_queued_request() -> None:
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.enqueue_request("r1", {"is_prefill": True})

    assert scheduler.queued_ids == ["r1"]
    scheduler.abort_request("r1")

    assert scheduler.active_request_count == 0
    assert scheduler.queued_request_count == 0


def test_request_scheduler_rejects_expired_queued_requests() -> None:
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.enqueue_request("r1", {"is_prefill": True, "queued_at": 1.0})
    scheduler.enqueue_request("r2", {"is_prefill": True, "queued_at": 9.5})

    expired = scheduler.reject_expired_queued_requests(now=10.0, max_queue_wait_s=5.0)

    assert len(expired) == 1
    assert expired[0][0] == "r1"
    assert scheduler.queued_ids == ["r2"]


@pytest.mark.asyncio
async def test_async_driver_propagates_background_error_to_engine() -> None:
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

    engine = FakeEngine()
    driver = AsyncDriver(engine)
    driver.notify_new_work()
    await asyncio.sleep(0.05)
    driver.shutdown()

    assert engine.calls >= 1
    assert isinstance(engine.error, RuntimeError)
    assert "driver-failure" in str(engine.error)


@pytest.mark.asyncio
async def test_async_llm_generate_abort_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEngine:
        def __init__(self, _vllm_config) -> None:
            self.tokenizer = None
            self.streams: dict[str, asyncio.Queue] = {}

        def add_request(self, request_id: str, prompt: str, sampling_params: SamplingParams, lora_id=None) -> None:
            del prompt, sampling_params, lora_id
            self.streams[request_id] = asyncio.Queue()

        async def get_request_stream(self, request_id: str):
            queue = self.streams[request_id]
            while True:
                output = await queue.get()
                yield output
                if output.finished:
                    break

        def abort_request(self, request_id: str) -> None:
            self.streams[request_id].put_nowait(_request_output(request_id, finished=True))

    class FakeDriver:
        def __init__(self, engine) -> None:
            self.engine = engine
            self.notified = 0

        def notify_new_work(self) -> None:
            self.notified += 1

        def shutdown(self) -> None:
            return None

    class DummyConfig:
        class ModelConfig:
            model = "dummy"

        model_config = ModelConfig()

    monkeypatch.setattr(async_llm_module, "LiteEngine", FakeEngine)
    monkeypatch.setattr(async_llm_module, "AsyncDriver", FakeDriver)
    monkeypatch.setattr(async_llm_module, "get_tokenizer", lambda *_args, **_kwargs: object())

    llm = async_llm_module.AsyncLLM(DummyConfig())
    agen = llm.generate("hi", SamplingParams(max_tokens=4), "req-1")
    abort_task = asyncio.create_task(llm.abort("req-1"))
    output = await agen.__anext__()
    await abort_task

    assert output.request_id == "req-1"
    assert output.finished is True
    assert llm.driver.notified == 1
