# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from vllm.entrypoints.llm import LLM
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams


class _FakeEngine:
    def __init__(self, steps: list[list[RequestOutput]], active_count: int = 1) -> None:
        self.max_active_requests = 1
        self.active_request_count = active_count
        self._steps = iter(steps)
        self.request_ids: list[str] = []

    def add_request(
        self, request_id: str, _prompt: str, _params: SamplingParams
    ) -> None:
        self.request_ids.append(request_id)

    def step(self) -> list[RequestOutput]:
        outputs = next(self._steps)
        if outputs:
            for output in outputs:
                output.request_id = self.request_ids[0]
            self.active_request_count = 0
        return outputs


def _output() -> RequestOutput:
    return RequestOutput(
        request_id="offline_0_0",
        prompt="prompt",
        prompt_token_ids=[1],
        outputs=[
            CompletionOutput(index=0, text="x", token_ids=[2], cumulative_logprob=0.0)
        ],
        finished=True,
    )


def _llm(engine: _FakeEngine) -> LLM:
    llm = object.__new__(LLM)
    llm.engine = engine
    llm.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
    llm.output_processor = type(
        "OutputProcessor",
        (),
        {"process_outputs": staticmethod(lambda outputs, **_: outputs)},
    )()
    return llm


def test_generate_allows_empty_chunked_prefill_step() -> None:
    llm = _llm(_FakeEngine([[], [_output()]]))

    outputs = llm.generate(["prompt"], SamplingParams(max_tokens=1))

    assert len(outputs) == 1
    assert outputs[0].outputs[0].token_ids == [2]


def test_generate_rejects_empty_step_after_runtime_stops() -> None:
    llm = _llm(_FakeEngine([[]], active_count=0))

    with pytest.raises(RuntimeError, match="stopped before all requests completed"):
        llm.generate(["prompt"], SamplingParams(max_tokens=1))
