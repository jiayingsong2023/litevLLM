# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from fastapi.testclient import TestClient
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import json as pyjson

from vllm.entrypoints.openai import api_server
from vllm.engine.output_pipeline import OutputPipeline
from vllm.engine.sampling_driver import SamplingDriver
from vllm.engine.structured_output_lite import build_structured_output_constraint
from vllm.outputs import CompletionOutput, RequestOutput


class _FakeOutput:
    def __init__(self, text: str = "ok", finished: bool = True) -> None:
        self.finished = finished
        self.outputs = [type("_Choice", (), {"text": text})()]


class _FakeEngine:
    def __init__(self) -> None:
        self.calls = []

    async def get_model_config(self):
        return type("ModelConfig", (), {"model": "demo-model"})()

    async def generate(self, prompt, sampling_params, request_id):
        self.calls.append(
            {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "request_id": request_id,
            }
        )
        yield _FakeOutput()


class _Policies:
    def apply_context_bias(
        self,
        logits: torch.Tensor,
        generated_ids,
        sampling_params,
        capital_question_bias_token_ids,
        is_chinese_capital_question,
    ) -> torch.Tensor:
        del generated_ids, sampling_params, capital_question_bias_token_ids, is_chinese_capital_question
        return logits

    def should_early_stop(self, generated_ids, current_text: str) -> bool:
        del generated_ids, current_text
        return False

    def cleanup_output_text(self, text: str) -> str:
        return text


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
        "x": 12,
        "y": 13,
        '{"A":1}': 14,
        "hello": 15,
    }
    base = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    base.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=base,
        unk_token="[UNK]",
        eos_token="[UNK]",
    )


class _BehaviorEngine:
    def __init__(self) -> None:
        self.calls = []
        self.tokenizer = _hf_tokenizer()
        self.sampling_driver = SamplingDriver(self.tokenizer, None, _Policies())
        self.output_pipeline = OutputPipeline(self.tokenizer, _Policies(), self.sampling_driver)

    async def get_model_config(self):
        return type("ModelConfig", (), {"model": "demo-model"})()

    async def generate(self, prompt, sampling_params, request_id):
        self.calls.append(
            {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "request_id": request_id,
            }
        )
        request = {
            "request_id": request_id,
            "prompt": prompt,
            "input_ids": [15],
            "generated_ids": [],
            "sampling_params": sampling_params,
            "finished": False,
            "low_info_hits": 0,
            "structured_output_constraint": build_structured_output_constraint(
                self.tokenizer,
                sampling_params,
            ),
        }
        for logits in self._logits_sequence_for(sampling_params):
            token = self.sampling_driver.sample_next_token(logits, request)
            request["generated_ids"].append(token)
            out = self.output_pipeline.finalize_step(request_id, request, token)
            yield out
            if out.finished:
                return

    def _logits_sequence_for(self, sampling_params):
        params = sampling_params.structured_outputs
        size = self.tokenizer.vocab_size
        if params is None:
            logits = torch.full((size,), 0.0)
            logits[12] = 5.0
            return [logits]
        if params.choice is not None:
            logits = torch.full((size,), 0.0)
            logits[4] = 9.0
            logits[5] = 8.0
            return [logits]
        if params.regex is not None:
            if "AB|AC" in params.regex:
                logits = torch.full((size,), 0.0)
                logits[4] = 9.0
                logits[5] = 8.0
                return [logits]
            if '"x"' in params.regex or "(?:x)" in params.regex:
                logits1 = torch.full((size,), 0.0)
                logits1[12] = 9.0
                logits2 = torch.full((size,), 0.0)
                logits2[0] = 9.0
                return [logits1, logits2]
        if params.grammar is not None:
            logits1 = torch.full((size,), 0.0)
            logits1[12] = 9.0
            logits2 = torch.full((size,), 0.0)
            logits2[0] = 9.0
            return [logits1, logits2]
        if params.json_object or params.json is not None:
            logits = torch.full((size,), 0.0)
            logits[14] = 9.0
            return [logits]
        raise AssertionError("unsupported structured outputs test case")


class _ToyTokenizer:
    eos_token_id = 99

    def encode(self, text: str):
        mapping = {
            "A": [1],
            "B": [2],
            "AB": [1, 2],
            "AC": [1, 3],
            '{"A":1}': [14],
            "hello": [15],
            "{": [4],
        }
        return list(mapping[text])

    def decode(self, token_ids, **kwargs):
        del kwargs
        inv = {
            1: "A",
            2: "B",
            3: "C",
            4: "{",
            14: '{"A":1}',
            15: "hello",
            99: "",
        }
        return "".join(inv.get(int(t), "?") for t in token_ids)


class _FallbackBehaviorEngine:
    def __init__(self) -> None:
        self.calls = []
        self.tokenizer = _ToyTokenizer()
        self.sampling_driver = SamplingDriver(self.tokenizer, None, _Policies())
        self.output_pipeline = OutputPipeline(self.tokenizer, _Policies(), self.sampling_driver)

    async def get_model_config(self):
        return type("ModelConfig", (), {"model": "demo-model"})()

    async def generate(self, prompt, sampling_params, request_id):
        self.calls.append(
            {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "request_id": request_id,
            }
        )
        request = {
            "request_id": request_id,
            "prompt": prompt,
            "input_ids": [15],
            "generated_ids": [],
            "sampling_params": sampling_params,
            "finished": False,
            "low_info_hits": 0,
            "structured_output_constraint": build_structured_output_constraint(
                self.tokenizer,
                sampling_params,
            ),
        }
        for logits in self._logits_sequence_for(sampling_params):
            token = self.sampling_driver.sample_next_token(logits, request)
            request["generated_ids"].append(token)
            out = self.output_pipeline.finalize_step(request_id, request, token)
            yield out
            if out.finished:
                return

    def _logits_sequence_for(self, sampling_params):
        params = sampling_params.structured_outputs
        size = 32
        if params is None:
            logits = torch.full((size,), 0.0)
            logits[1] = 5.0
            return [logits]
        if params.choice is not None:
            logits1 = torch.full((size,), 0.0)
            logits1[1] = 9.0
            logits2 = torch.full((size,), 0.0)
            logits2[2] = 9.0
            return [logits1, logits2]
        if params.json_object or params.json is not None:
            logits = torch.full((size,), 0.0)
            logits[14] = 9.0
            return [logits]
        raise AssertionError("unsupported fallback structured outputs test case")


def _complex_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "A": {"type": "number"},
        },
        "required": ["A"],
        "additionalProperties": False,
        "allOf": [
            {"properties": {"A": {"minimum": 1}}},
            {"properties": {"A": {"maximum": 1}}},
        ],
    }


def _post_chat(body: dict) -> tuple[dict, _FakeEngine]:
    fake_engine = _FakeEngine()
    prev_engine = api_server.engine
    api_server.engine = fake_engine
    try:
        client = TestClient(api_server.app)
        response = client.post("/v1/chat/completions", json=body)
    finally:
        api_server.engine = prev_engine
    assert response.status_code == 200, response.text
    return response.json(), fake_engine


def _post_chat_with_engine(body: dict, fake_engine) -> dict:
    prev_engine = api_server.engine
    api_server.engine = fake_engine
    try:
        client = TestClient(api_server.app)
        response = client.post("/v1/chat/completions", json=body)
    finally:
        api_server.engine = prev_engine
    assert response.status_code == 200, response.text
    return response.json()


def _post_chat_stream_with_engine(body: dict, fake_engine) -> tuple[list[dict], str]:
    prev_engine = api_server.engine
    api_server.engine = fake_engine
    try:
        client = TestClient(api_server.app)
        response = client.post("/v1/chat/completions", json=body)
    finally:
        api_server.engine = prev_engine
    assert response.status_code == 200, response.text
    events: list[dict] = []
    done = False
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            done = True
            continue
        events.append(pyjson.loads(payload))
    assert done is True
    content = "".join(
        event["choices"][0]["delta"].get("content", "")
        for event in events
    )
    return events, content


def _post_chat_expect_error(body: dict, fake_engine) -> tuple[int, str]:
    prev_engine = api_server.engine
    api_server.engine = fake_engine
    try:
        client = TestClient(api_server.app, raise_server_exceptions=False)
        response = client.post("/v1/chat/completions", json=body)
    finally:
        api_server.engine = prev_engine
    return response.status_code, response.text


def test_chat_completion_maps_structured_outputs_choice() -> None:
    _, fake_engine = _post_chat(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "structured_outputs": {"choice": ["A", "B"]},
        }
    )

    params = fake_engine.calls[0]["sampling_params"]
    assert params.structured_outputs is not None
    assert params.structured_outputs.choice == ["A", "B"]


def test_chat_completion_maps_structured_outputs_regex() -> None:
    _, fake_engine = _post_chat(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "structured_outputs": {"regex": "^(?:A|B)$"},
        }
    )

    params = fake_engine.calls[0]["sampling_params"]
    assert params.structured_outputs is not None
    assert params.structured_outputs.regex == "^(?:A|B)$"


def test_chat_completion_maps_structured_outputs_grammar() -> None:
    _, fake_engine = _post_chat(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "structured_outputs": {"grammar": 'root ::= "x"'},
        }
    )

    params = fake_engine.calls[0]["sampling_params"]
    assert params.structured_outputs is not None
    assert params.structured_outputs.grammar == 'root ::= "x"'


def test_chat_completion_maps_response_format_json_object() -> None:
    _, fake_engine = _post_chat(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "json_object"},
        }
    )

    params = fake_engine.calls[0]["sampling_params"]
    assert params.structured_outputs is not None
    assert params.structured_outputs.json_object is True


def test_chat_completion_maps_response_format_json_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    _, fake_engine = _post_chat(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_schema",
                    "schema": schema,
                },
            },
        }
    )

    params = fake_engine.calls[0]["sampling_params"]
    assert params.structured_outputs is not None
    assert params.structured_outputs.json == schema


def test_chat_completion_prefers_structured_outputs_over_response_format() -> None:
    _, fake_engine = _post_chat(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "structured_outputs": {"choice": ["A"]},
            "response_format": {"type": "json_object"},
        }
    )

    params = fake_engine.calls[0]["sampling_params"]
    assert params.structured_outputs is not None
    assert params.structured_outputs.choice == ["A"]
    assert params.structured_outputs.json_object is False


def test_chat_completion_rejects_invalid_json_schema_response_format() -> None:
    fake_engine = _FakeEngine()
    prev_engine = api_server.engine
    api_server.engine = fake_engine
    try:
        client = TestClient(api_server.app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "demo-model",
                "messages": [{"role": "user", "content": "hello"}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "bad"}},
            },
        )
    finally:
        api_server.engine = prev_engine

    assert response.status_code == 400
    assert response.json()["detail"] == "response_format.json_schema.schema must be an object"


def test_chat_completion_streaming_preserves_structured_outputs_mapping() -> None:
    fake_engine = _FakeEngine()
    prev_engine = api_server.engine
    api_server.engine = fake_engine
    try:
        client = TestClient(api_server.app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "demo-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "structured_outputs": {"regex": "^(?:A|B)$"},
            },
        )
    finally:
        api_server.engine = prev_engine

    assert response.status_code == 200
    params = fake_engine.calls[0]["sampling_params"]
    assert params.structured_outputs is not None
    assert params.structured_outputs.regex == "^(?:A|B)$"


def test_chat_completion_choice_response_content_satisfies_constraint() -> None:
    response = _post_chat_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "structured_outputs": {"choice": ["AB", "AC"]},
        },
        _BehaviorEngine(),
    )

    assert response["choices"][0]["message"]["content"] in {"AB", "AC"}


def test_chat_completion_json_object_response_content_is_json_object() -> None:
    response = _post_chat_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "json_object"},
        },
        _BehaviorEngine(),
    )

    assert response["choices"][0]["message"]["content"] == '{"A":1}'


def test_chat_completion_json_schema_response_content_matches_schema() -> None:
    response = _post_chat_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"A": {"type": "number"}},
                        "required": ["A"],
                    },
                },
            },
        },
        _BehaviorEngine(),
    )

    assert response["choices"][0]["message"]["content"] == '{"A":1}'


def test_chat_completion_structured_outputs_json_complex_schema_matches_content() -> None:
    schema = _complex_json_schema()
    response = _post_chat_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "structured_outputs": {"json": schema},
        },
        _BehaviorEngine(),
    )

    assert pyjson.loads(response["choices"][0]["message"]["content"]) == {"A": 1}


def test_chat_completion_response_format_complex_json_schema_matches_content() -> None:
    schema = _complex_json_schema()
    response = _post_chat_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "nested_answer",
                    "schema": schema,
                },
            },
        },
        _BehaviorEngine(),
    )

    assert pyjson.loads(response["choices"][0]["message"]["content"]) == {"A": 1}


def test_chat_completion_regex_response_content_matches_pattern() -> None:
    response = _post_chat_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "structured_outputs": {"regex": "^(?:AB|AC)$"},
        },
        _BehaviorEngine(),
    )

    assert response["choices"][0]["message"]["content"] in {"AB", "AC"}


def test_chat_completion_grammar_response_content_matches_grammar() -> None:
    response = _post_chat_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "structured_outputs": {"grammar": 'root ::= "x"'},
        },
        _BehaviorEngine(),
    )

    assert response["choices"][0]["message"]["content"] == "x"


def test_chat_completion_streaming_choice_content_satisfies_constraint() -> None:
    events, content = _post_chat_stream_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "structured_outputs": {"choice": ["AB", "AC"]},
        },
        _BehaviorEngine(),
    )

    assert content in {"AB", "AC"}
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_streaming_json_object_content_is_json_object() -> None:
    events, content = _post_chat_stream_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "response_format": {"type": "json_object"},
        },
        _BehaviorEngine(),
    )

    assert content == '{"A":1}'
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_streaming_json_schema_content_matches_schema() -> None:
    events, content = _post_chat_stream_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"A": {"type": "number"}},
                        "required": ["A"],
                    },
                },
            },
        },
        _BehaviorEngine(),
    )

    assert content == '{"A":1}'
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_streaming_structured_outputs_json_complex_schema_matches_content() -> None:
    events, content = _post_chat_stream_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "structured_outputs": {"json": _complex_json_schema()},
        },
        _BehaviorEngine(),
    )

    assert pyjson.loads(content) == {"A": 1}
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_streaming_response_format_complex_json_schema_matches_content() -> None:
    events, content = _post_chat_stream_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "nested_answer",
                    "schema": _complex_json_schema(),
                },
            },
        },
        _BehaviorEngine(),
    )

    assert pyjson.loads(content) == {"A": 1}
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_streaming_regex_content_matches_pattern() -> None:
    events, content = _post_chat_stream_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "structured_outputs": {"regex": "^(?:AB|AC)$"},
        },
        _BehaviorEngine(),
    )

    assert content in {"AB", "AC"}
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_streaming_grammar_content_matches_grammar() -> None:
    events, content = _post_chat_stream_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "structured_outputs": {"grammar": 'root ::= "x"'},
        },
        _BehaviorEngine(),
    )

    assert content == "x"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_fallback_choice_content_satisfies_constraint() -> None:
    response = _post_chat_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "structured_outputs": {"choice": ["AB", "AC"]},
        },
        _FallbackBehaviorEngine(),
    )

    assert response["choices"][0]["message"]["content"] == "AB"


def test_chat_completion_fallback_json_object_content_is_json_object() -> None:
    response = _post_chat_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "json_object"},
        },
        _FallbackBehaviorEngine(),
    )

    assert response["choices"][0]["message"]["content"] == '{"A":1}'


def test_chat_completion_fallback_json_schema_content_matches_schema() -> None:
    response = _post_chat_with_engine(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"A": {"type": "number"}},
                        "required": ["A"],
                    },
                },
            },
        },
        _FallbackBehaviorEngine(),
    )

    assert response["choices"][0]["message"]["content"] == '{"A":1}'


def test_chat_completion_fallback_regex_returns_server_error() -> None:
    status_code, body = _post_chat_expect_error(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "structured_outputs": {"regex": "^(?:A|B)$"},
        },
        _FallbackBehaviorEngine(),
    )

    assert status_code == 500
    assert "Internal Server Error" in body


def test_chat_completion_fallback_grammar_returns_server_error() -> None:
    status_code, body = _post_chat_expect_error(
        {
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "structured_outputs": {"grammar": 'root ::= "x"'},
        },
        _FallbackBehaviorEngine(),
    )

    assert status_code == 500
    assert "Internal Server Error" in body
