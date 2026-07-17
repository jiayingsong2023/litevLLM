from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator

import torch

from vllm.engine.async_llm import AsyncLLM
from vllm.sampling_params import SamplingParams


class _FakeModel:
    def generate_greedy_reference(
        self,
        input_ids: torch.Tensor,
        *,
        max_tokens: int,
    ) -> torch.Tensor:
        assert input_ids.tolist() == [7]
        assert max_tokens == 1
        return torch.tensor([7, 42], dtype=torch.long)


class _FakeTokenizer:
    eos_token_id = 0

    def encode(self, prompt: str) -> list[int]:
        assert prompt == "hello"
        return [3, 7]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        assert ids == [42]
        assert skip_special_tokens is True
        return "world"


class _FakeEngine:
    model = _FakeModel()
    tokenizer = _FakeTokenizer()


class _FakeRequestOutput:
    def __init__(self, text: str, finished: bool) -> None:
        self.outputs = [type("_FakeToken", (), {"text": text})()]
        self.finished = finished


class _BridgeEngine:
    def __init__(self) -> None:
        self.added_requests: list[tuple[str, str]] = []
        self.aborted_requests: list[str] = []

    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params,
        *,
        lora_id=None,
        lora_request=None,
        multi_modal_data=None,
    ) -> None:
        self.added_requests.append((request_id, prompt))
        assert getattr(sampling_params, "max_tokens", None) == 1
        assert lora_id is None
        assert lora_request is None
        self.multi_modal_data = multi_modal_data

    async def get_request_stream(self, request_id: str) -> AsyncGenerator[object, None]:
        yield _FakeRequestOutput("bridge", True)

    def abort_request(self, request_id: str) -> None:
        self.aborted_requests.append(request_id)

    def stats(self) -> dict[str, object]:
        return {"bridge": True}


class _BridgeDriver:
    def __init__(self) -> None:
        self.notified = False

    def notify_new_work(self) -> None:
        self.notified = True

    def stats(self) -> dict[str, object]:
        return {"driver": True}


def _bridge_llm() -> AsyncLLM:
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.engine = _BridgeEngine()
    llm.driver = _BridgeDriver()
    llm._engine_lock = threading.RLock()
    return llm


def _generate_greedy_reference_chat(
    engine: _FakeEngine,
    prompt: str,
    *,
    max_tokens: int,
) -> str:
    if max_tokens != 1:
        raise ValueError(
            "direct greedy reference chat currently supports max_tokens=1; "
            f"got {max_tokens}"
        )
    token_ids = engine.tokenizer.encode(prompt)
    if not token_ids:
        eos = getattr(engine.tokenizer, "eos_token_id", None)
        token_ids = [0 if eos is None else int(eos)]
    tokens = engine.model.generate_greedy_reference(
        torch.tensor([int(token_ids[-1])], dtype=torch.long),
        max_tokens=1,
    )
    generated = int(tokens[-1].item())
    try:
        return engine.tokenizer.decode([generated], skip_special_tokens=True)
    except TypeError:
        return engine.tokenizer.decode([generated])


def test_async_llm_does_not_expose_direct_reference_chat_helper() -> None:
    assert not hasattr(AsyncLLM, "generate_greedy_reference_chat")


def test_direct_reference_chat_helper_is_local_to_the_test_module() -> None:
    text = _generate_greedy_reference_chat(_FakeEngine(), "hello", max_tokens=1)
    assert text == "world"


def test_async_llm_generate_bridges_request_stream() -> None:
    llm = _bridge_llm()

    async def run() -> list[str]:
        from vllm.sampling_params import SamplingParams

        outputs: list[str] = []
        async for output in llm.generate(
            "bridge prompt",
            SamplingParams(max_tokens=1, temperature=0.0),
            "req-1",
        ):
            outputs.append(output.outputs[0].text)
        return outputs

    assert asyncio.run(run()) == ["bridge"]
    assert llm.engine.added_requests == [("req-1", "bridge prompt")]
    assert llm.driver.notified is True
    assert llm.stats() == {"bridge": True, "async_driver": {"driver": True}}


def test_async_llm_generate_uses_the_engine_request_path() -> None:
    llm = _bridge_llm()

    async def run() -> list[str]:
        outputs: list[str] = []
        async for output in llm.generate(
            "direct prompt",
            SamplingParams(max_tokens=1, temperature=0.0),
            "req-direct",
        ):
            outputs.append(output.outputs[0].text)
        return outputs

    assert asyncio.run(run()) == ["bridge"]
    assert llm.engine.added_requests == [("req-direct", "direct prompt")]
    assert llm.driver.notified is True


def test_async_llm_generate_forwards_multimodal_to_engine() -> None:
    llm = _bridge_llm()
    image = object()

    async def run() -> None:
        async for _ in llm.generate(
            "direct prompt",
            SamplingParams(max_tokens=1, temperature=0.0),
            "req-direct",
            multi_modal_data={"image": image},
        ):
            pass

    asyncio.run(run())

    assert llm.engine.multi_modal_data == {"image": image}


def test_async_llm_abort_forwards_request_ids() -> None:
    llm = _bridge_llm()

    asyncio.run(llm.abort(["req-a", "req-b"]))

    assert llm.engine.aborted_requests == ["req-a", "req-b"]


def test_async_llm_close_stream_aborts_uniterated_request() -> None:
    llm = _bridge_llm()

    asyncio.run(llm.close_stream("req-never-iterated"))

    assert llm.engine.aborted_requests == ["req-never-iterated"]
