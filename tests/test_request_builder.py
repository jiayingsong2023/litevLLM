# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from vllm.engine.request_builder import LiteRequestBuilder
from vllm.engine.request_state import RequestState
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
        '"': 8,
        ":": 9,
        ",": 10,
        "1": 11,
        "prompt": 12,
        "guarded": 13,
        "x": 14,
        "y": 15,
        '{"A":1}': 16,
        "<image>": 17,
    }
    base = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    base.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=base,
        unk_token="[UNK]",
        eos_token="[UNK]",
    )


def test_request_state_declares_expected_fields() -> None:
    request = RequestState(
        request_id="r-state",
        prompt="prompt",
        guarded_prompt="guarded",
        input_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=4),
    )

    assert request.request_id == "r-state"
    assert request.prompt == "prompt"
    assert request.guarded_prompt == "guarded"
    assert request.input_ids == [1, 2]
    assert request.is_prefill
    assert not request.finished


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

    assert request.input_ids == [10, 11]
    assert request.structured_output_constraint is not None
    assert request.queued_at is not None
    assert request.service_class == "latency"


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

    assert request.service_class == "background"


def test_request_builder_uses_explicit_default_min_new_tokens(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_LITE_DEFAULT_MIN_NEW_TOKENS", "9")
    builder = LiteRequestBuilder(
        tokenizer=_Tokenizer(),
        policies=_Policies(),
        device=torch.device("cpu"),
        num_layers=2,
        max_model_len=16,
        max_tokens_cap=8,
        default_min_new_tokens=3,
    )

    request = builder.build(
        request_id="r1",
        prompt="prompt",
        sampling_params=SamplingParams(max_tokens=4),
    )

    assert request.sampling_params.min_tokens == 3


def test_request_builder_preserves_explicit_min_tokens() -> None:
    builder = LiteRequestBuilder(
        tokenizer=_Tokenizer(),
        policies=_Policies(),
        device=torch.device("cpu"),
        num_layers=2,
        max_model_len=16,
        max_tokens_cap=8,
        default_min_new_tokens=3,
    )

    request = builder.build(
        request_id="r1",
        prompt="prompt",
        sampling_params=SamplingParams(max_tokens=4, min_tokens=2),
    )

    assert request.sampling_params.min_tokens == 2


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

    assert request.structured_output_constraint is not None


def test_request_builder_marks_multimodal_lora_request() -> None:
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
        sampling_params=SamplingParams(max_tokens=4),
        lora_id="adapter-a",
        lora_int_id=7,
        lora_path="/tmp/adapter-a",
        multi_modal_data={"image": [{"image": "file:///tmp/cat.png"}]},
    )

    assert request.lora_id == "adapter-a"
    assert request.lora_int_id == 7
    assert request.lora_path == "/tmp/adapter-a"
    assert request.is_multimodal is True
    assert request.is_multimodal_lora is True


def test_request_builder_adds_structured_constraint_for_multimodal() -> None:
    builder = LiteRequestBuilder(
        tokenizer=_hf_tokenizer(),
        policies=_Policies(),
        device=torch.device("cpu"),
        num_layers=1,
        max_model_len=64,
        max_tokens_cap=16,
    )

    request = builder.build(
        request_id="req-mm-structured",
        prompt="describe image",
        sampling_params=SamplingParams(
            max_tokens=4,
            structured_outputs=StructuredOutputsParams(choice=["AB", "AC"]),
        ),
        multi_modal_data={"image": [{"image": "file:///tmp/cat.png"}]},
    )

    assert request.is_multimodal is True
    assert request.is_multimodal_lora is False
    assert request.structured_output_constraint is not None


class _PreTokenizeMultiModalProcessor:
    def __init__(self) -> None:
        self.calls = []

    def prepare_before_tokenize(self, prompt, multi_modal_data):
        self.calls.append((prompt, multi_modal_data))
        prepared = {
            "image": [{"prepared_image": "prepared"}],
            "image_token": "<image>",
            "image_token_count": 3,
        }
        return "A B C", prepared


class _MultiImagePreTokenizeMultiModalProcessor:
    def prepare_before_tokenize(self, prompt, multi_modal_data):
        del prompt, multi_modal_data
        return (
            "A B C A B C",
            {
                "image": [{"prepared_image": "a"}, {"prepared_image": "b"}],
                "image_token": "<image>",
                "image_token_count": 6,
                "image_token_counts": [3, 3],
            },
        )


def test_request_builder_expands_multimodal_prompt_before_tokenize() -> None:
    processor = _PreTokenizeMultiModalProcessor()
    builder = LiteRequestBuilder(
        tokenizer=_hf_tokenizer(),
        policies=_Policies(),
        device=torch.device("cpu"),
        num_layers=1,
        max_model_len=64,
        max_tokens_cap=16,
        multimodal_processor=processor,
    )

    request = builder.build(
        request_id="req-mm-expanded",
        prompt="describe <image>",
        sampling_params=SamplingParams(max_tokens=4),
        multi_modal_data={"image": [{"image": "file:///tmp/cat.png"}]},
    )

    assert processor.calls == [("guarded", {"image": [{"image": "file:///tmp/cat.png"}]})]
    assert request.guarded_prompt == "A B C"
    assert request.input_ids == [1, 2, 3]
    assert request.multi_modal_data == {
        "image": [{"prepared_image": "prepared"}],
        "image_token": "<image>",
        "image_token_id": 17,
        "image_token_count": 3,
    }
    assert request.is_multimodal is True


def test_request_builder_preserves_multi_image_token_counts() -> None:
    builder = LiteRequestBuilder(
        tokenizer=_hf_tokenizer(),
        policies=_Policies(),
        device=torch.device("cpu"),
        num_layers=1,
        max_model_len=64,
        max_tokens_cap=16,
        multimodal_processor=_MultiImagePreTokenizeMultiModalProcessor(),
    )

    request = builder.build(
        request_id="req-mm-expanded",
        prompt="compare <image> and <image>",
        sampling_params=SamplingParams(max_tokens=4),
        multi_modal_data={
            "image": [
                {"image": "file:///tmp/a.png"},
                {"image": "file:///tmp/b.png"},
            ]
        },
    )

    assert request.input_ids == [1, 2, 3, 1, 2, 3]
    assert request.multi_modal_data["image_token_count"] == 6
    assert request.multi_modal_data["image_token_counts"] == [3, 3]
    assert request.multi_modal_data["image_token_id"] == 17


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
                structured_outputs=StructuredOutputsParams(grammar='root ::= "x"'),
            ),
        )
    except ValueError as exc:
        assert "xgrammar-compatible tokenizer" in str(exc)
    else:
        raise AssertionError("expected unsupported structured outputs type to fail")


def test_request_builder_adds_grammar_constraint_for_hf_tokenizer() -> None:
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

    assert request.structured_output_constraint is not None


def test_request_builder_adds_choice_constraint_for_hf_tokenizer() -> None:
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

    assert request.structured_output_constraint is not None


def test_request_builder_adds_json_constraint_for_hf_tokenizer() -> None:
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

    assert request.structured_output_constraint is not None
