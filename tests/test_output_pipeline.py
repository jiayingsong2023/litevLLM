# SPDX-License-Identifier: Apache-2.0
"""Tests for OutputPipeline deferred text decoding."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.engine.output_pipeline import OutputPipeline
from vllm.engine.output_processor import OutputProcessor
from vllm.engine.request_state import RequestState
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.policies.base import GenerationPolicies
from vllm.sampling_params import SamplingParams


class FakeTokenizer:
    def __init__(self) -> None:
        self.decode_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        self.decode_calls.append((tuple(token_ids), kwargs))
        return "".join(chr(t + 96) for t in token_ids)


class CountingOutputProcessor(OutputProcessor):
    """Policy that counts should_early_stop calls and can force early stop."""

    def __init__(self, stop_text: str | None = None) -> None:
        super().__init__(MagicMock(), "")
        self.stop_text = stop_text
        self.early_stop_calls = 0

    def needs_partial_text_for_early_stop(self) -> bool:
        return self.stop_text is not None

    def should_early_stop(self, generated_ids: list[int], partial_text: str) -> bool:
        self.early_stop_calls += 1
        return self.stop_text is not None and self.stop_text in partial_text


class CleanupOutputProcessor(OutputProcessor):
    def cleanup_output_text(self, text: str) -> str:
        return text.upper()


def _make_request(
    generated_ids: list[int], max_tokens: int = 10, ignore_eos: bool = False
) -> RequestState:
    return RequestState(
        request_id="r1",
        prompt="hello",
        input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos),
        generated_ids=generated_ids,
    )


def _make_sampling_driver() -> MagicMock:
    driver = MagicMock()
    driver.completion_eos_ids.return_value = {3}
    return driver


def test_completion_output_lazy_decodes_on_text_access() -> None:
    tokenizer = FakeTokenizer()
    comp = CompletionOutput(
        index=0,
        text=None,
        token_ids=[1, 2],
        cumulative_logprob=0.0,
        tokenizer=tokenizer,
        sampling_params=SamplingParams(),
        finished=False,
    )
    assert not tokenizer.decode_calls
    assert comp.text == "ab"
    assert len(tokenizer.decode_calls) == 1
    # Accessing again should decode again because the request is not finished
    # and token_ids may have grown.
    assert comp.text == "ab"
    assert len(tokenizer.decode_calls) == 2


def test_completion_output_caches_text_when_finished() -> None:
    tokenizer = FakeTokenizer()
    comp = CompletionOutput(
        index=0,
        text=None,
        token_ids=[1, 2],
        cumulative_logprob=0.0,
        tokenizer=tokenizer,
        sampling_params=SamplingParams(),
        finished=True,
    )
    assert comp.text == "ab"
    assert len(tokenizer.decode_calls) == 1
    assert comp.text == "ab"
    assert len(tokenizer.decode_calls) == 1


def test_completion_output_text_processor_applied_lazily() -> None:
    tokenizer = FakeTokenizer()
    comp = CompletionOutput(
        index=0,
        text=None,
        token_ids=[1, 2],
        cumulative_logprob=0.0,
        tokenizer=tokenizer,
        sampling_params=SamplingParams(),
        text_processor=lambda s: s.upper(),
        finished=False,
    )
    assert comp.text == "AB"


def test_output_pipeline_skips_decode_for_intermediate_step() -> None:
    tokenizer = FakeTokenizer()
    policies = CountingOutputProcessor()
    pipeline = OutputPipeline(tokenizer, policies, _make_sampling_driver())
    req = _make_request(generated_ids=[1, 2])

    out = pipeline.finalize_step("r1", req, next_token=4)

    assert isinstance(out, RequestOutput)
    assert not out.finished
    assert not tokenizer.decode_calls
    assert policies.early_stop_calls == 0


def test_output_pipeline_decodes_when_finished() -> None:
    tokenizer = FakeTokenizer()
    policies = CountingOutputProcessor()
    pipeline = OutputPipeline(tokenizer, policies, _make_sampling_driver())
    req = _make_request(generated_ids=[1, 2], max_tokens=2)

    out = pipeline.finalize_step("r1", req, next_token=4)

    assert out.finished
    assert len(tokenizer.decode_calls) == 1
    assert out.outputs[0].text == "ab"


def test_output_pipeline_decodes_for_early_stop_policy() -> None:
    tokenizer = FakeTokenizer()
    policies = CountingOutputProcessor(stop_text="ab")
    pipeline = OutputPipeline(tokenizer, policies, _make_sampling_driver())
    req = _make_request(generated_ids=[1, 2], max_tokens=10)

    out = pipeline.finalize_step("r1", req, next_token=4)

    assert not out.finished
    assert policies.early_stop_calls == 1
    assert len(tokenizer.decode_calls) == 1


def test_output_pipeline_applies_cleanup_at_finish() -> None:
    tokenizer = FakeTokenizer()
    policies = CleanupOutputProcessor(MagicMock(), "")
    pipeline = OutputPipeline(tokenizer, policies, _make_sampling_driver())
    req = _make_request(generated_ids=[1, 2], max_tokens=2)

    out = pipeline.finalize_step("r1", req, next_token=4)

    assert out.outputs[0].text == "AB"


def test_build_abort_output_has_empty_text() -> None:
    pipeline = OutputPipeline(
        FakeTokenizer(), CountingOutputProcessor(), _make_sampling_driver()
    )
    req = _make_request(generated_ids=[1, 2])
    out = pipeline.build_abort_output("r1", req)
    assert out.finished
    assert out.outputs[0].text == ""


def test_output_pipeline_accepts_generation_policies_wrapper() -> None:
    """GenerationPolicies is the real object passed to OutputPipeline in-engine."""
    tokenizer = FakeTokenizer()
    backend = CountingOutputProcessor()
    policies = GenerationPolicies(backend=backend)
    pipeline = OutputPipeline(tokenizer, policies, _make_sampling_driver())
    req = _make_request(generated_ids=[1, 2], max_tokens=2)

    out = pipeline.finalize_step("r1", req, next_token=4)

    assert out.finished
    assert out.outputs[0].text == "ab"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
