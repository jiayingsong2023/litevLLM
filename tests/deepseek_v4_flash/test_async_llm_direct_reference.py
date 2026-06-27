from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest
import torch

from vllm.engine.async_llm import AsyncLLM
from vllm.outputs import CompletionOutput, RequestOutput
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
        assert multi_modal_data is None

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


class _DirectDeepSeekEngine:
    _deepseek_v4_flash_direct = True

    def __init__(self) -> None:
        self.generated: list[tuple[str, str, int]] = []

    def generate_deepseek_v4_flash_greedy(
        self,
        *,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> RequestOutput:
        self.generated.append((request_id, prompt, int(sampling_params.max_tokens)))
        return RequestOutput(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=[7],
            outputs=[
                CompletionOutput(
                    index=0,
                    text="direct",
                    token_ids=[42],
                    cumulative_logprob=0.0,
                )
            ],
            finished=True,
        )


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
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.engine = _BridgeEngine()
    llm.driver = _BridgeDriver()

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


def test_async_llm_generate_uses_deepseek_direct_runtime_without_driver() -> None:
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.engine = _DirectDeepSeekEngine()
    llm.driver = _BridgeDriver()

    async def run() -> list[str]:
        outputs: list[str] = []
        async for output in llm.generate(
            "direct prompt",
            SamplingParams(max_tokens=1, temperature=0.0),
            "req-direct",
        ):
            outputs.append(output.outputs[0].text)
        return outputs

    assert asyncio.run(run()) == ["direct"]
    assert llm.engine.generated == [("req-direct", "direct prompt", 1)]
    assert llm.driver.notified is False


def test_async_llm_deepseek_direct_rejects_multimodal_input() -> None:
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.engine = _DirectDeepSeekEngine()
    llm.driver = _BridgeDriver()

    async def run() -> None:
        async for _ in llm.generate(
            "direct prompt",
            SamplingParams(max_tokens=1, temperature=0.0),
            "req-direct",
            multi_modal_data={"image": object()},
        ):
            pass

    with pytest.raises(ValueError, match="multimodal"):
        asyncio.run(run())


def test_async_llm_abort_forwards_request_ids() -> None:
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.engine = _BridgeEngine()
    llm.driver = _BridgeDriver()

    asyncio.run(llm.abort(["req-a", "req-b"]))

    assert llm.engine.aborted_requests == ["req-a", "req-b"]
