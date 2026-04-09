# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from vllm.engine.request_builder import LiteRequestBuilder
from vllm.sampling_params import SamplingParams, StructuredOutputsParams


class _Tokenizer:
    eos_token_id = 99

    def encode(self, text: str):
        mapping = {
            "A": [1],
            "B": [2],
            "AB": [1, 2],
            "AX": [1, 3],
            "prompt": [10, 11],
            "guarded": [10, 11],
        }
        return list(mapping.get(text, [ord(c) for c in text]))


class _Policies:
    def normalize_prompt(self, prompt: str) -> str:
        del prompt
        return "guarded"

    def is_chinese_capital_question(self, prompt: str) -> bool:
        del prompt
        return False

    def capital_question_bias_token_ids(self, prompt: str) -> list[int]:
        del prompt
        return []

    def anti_template_token_ids(self) -> list[int]:
        return []


def _hf_tokenizer():
    vocab = {
        "[UNK]": 0,
        "A": 1,
        "B": 2,
        "C": 3,
        "AB": 4,
        "AC": 5,
        "{": 6,
        "}": 7,
        "\"": 8,
        ":": 9,
        ",": 10,
        "1": 11,
        "prompt": 12,
        "guarded": 13,
        "x": 14,
        "y": 15,
        '{"A":1}': 16,
    }
    base = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    base.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=base,
        unk_token="[UNK]",
        eos_token="[UNK]",
    )


def test_request_builder_attaches_choice_structured_output_constraint() -> None:
    builder = LiteRequestBuilder(
        tokenizer=_Tokenizer(),
        policies=_Policies(),
        device=torch.device("cpu"),
        num_layers=2,
        max_model_len=16,
        max_tokens_cap=8,
    )

    request = builder.build(
        request_id="r1",
        prompt="prompt",
        sampling_params=SamplingParams(
            max_tokens=4,
            structured_outputs=StructuredOutputsParams(choice=["AB", "AX"]),
        ),
    )

    assert request["input_ids"] == [10, 11]
    assert request["structured_output_constraint"] is not None
    assert request["queued_at"] is not None
    assert request["service_class"] == "latency"


def test_request_builder_preserves_explicit_service_class() -> None:
    builder = LiteRequestBuilder(
        tokenizer=_Tokenizer(),
        policies=_Policies(),
        device=torch.device("cpu"),
        num_layers=2,
        max_model_len=16,
        max_tokens_cap=8,
    )

    request = builder.build(
        request_id="r1",
        prompt="prompt",
        sampling_params=SamplingParams(
            max_tokens=4,
            service_class="background",
        ),
    )

    assert request["service_class"] == "background"


def test_request_builder_attaches_json_object_structured_output_constraint() -> None:
    builder = LiteRequestBuilder(
        tokenizer=_Tokenizer(),
        policies=_Policies(),
        device=torch.device("cpu"),
        num_layers=2,
        max_model_len=16,
        max_tokens_cap=8,
    )

    request = builder.build(
        request_id="r1",
        prompt="prompt",
        sampling_params=SamplingParams(
            max_tokens=4,
            structured_outputs=StructuredOutputsParams(json_object=True),
        ),
    )

    assert request["structured_output_constraint"] is not None


def test_request_builder_rejects_unsupported_structured_output_type() -> None:
    builder = LiteRequestBuilder(
        tokenizer=_Tokenizer(),
        policies=_Policies(),
        device=torch.device("cpu"),
        num_layers=2,
        max_model_len=16,
        max_tokens_cap=8,
    )

    try:
        builder.build(
            request_id="r1",
            prompt="prompt",
            sampling_params=SamplingParams(
                max_tokens=4,
                structured_outputs=StructuredOutputsParams(grammar="root ::= \"x\""),
            ),
        )
    except ValueError as exc:
        assert "xgrammar-compatible tokenizer" in str(exc)
    else:
        raise AssertionError("expected unsupported structured outputs type to fail")


def test_request_builder_attaches_grammar_structured_output_constraint_for_hf_tokenizer() -> None:
    builder = LiteRequestBuilder(
        tokenizer=_hf_tokenizer(),
        policies=_Policies(),
        device=torch.device("cpu"),
        num_layers=2,
        max_model_len=16,
        max_tokens_cap=8,
    )

    request = builder.build(
        request_id="r1",
        prompt="prompt",
        sampling_params=SamplingParams(
            max_tokens=4,
            structured_outputs=StructuredOutputsParams(grammar='root ::= "x"'),
        ),
    )

    assert request["structured_output_constraint"] is not None


def test_request_builder_attaches_choice_structured_output_constraint_for_hf_tokenizer() -> None:
    builder = LiteRequestBuilder(
        tokenizer=_hf_tokenizer(),
        policies=_Policies(),
        device=torch.device("cpu"),
        num_layers=2,
        max_model_len=16,
        max_tokens_cap=8,
    )

    request = builder.build(
        request_id="r1",
        prompt="prompt",
        sampling_params=SamplingParams(
            max_tokens=4,
            structured_outputs=StructuredOutputsParams(choice=["AB", "AC"]),
        ),
    )

    assert request["structured_output_constraint"] is not None


def test_request_builder_attaches_json_structured_output_constraint_for_hf_tokenizer() -> None:
    builder = LiteRequestBuilder(
        tokenizer=_hf_tokenizer(),
        policies=_Policies(),
        device=torch.device("cpu"),
        num_layers=2,
        max_model_len=16,
        max_tokens_cap=8,
    )

    request = builder.build(
        request_id="r1",
        prompt="prompt",
        sampling_params=SamplingParams(
            max_tokens=4,
            structured_outputs=StructuredOutputsParams(json_object=True),
        ),
    )

    assert request["structured_output_constraint"] is not None
