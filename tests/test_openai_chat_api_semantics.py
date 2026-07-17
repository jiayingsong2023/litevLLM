# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from vllm.engine.errors import RequestRejectedError
from vllm.entrypoints.openai import api_server
from vllm.outputs import CompletionOutput, RequestOutput


class _TemplateTokenizer:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] | None = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = list(messages)
        return "<chat>" + "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )


class _ChatEngine:
    def __init__(self, *, reject: str | None = None) -> None:
        self.tokenizer = _TemplateTokenizer()
        self.engine = SimpleNamespace(tokenizer=self.tokenizer)
        self.reject = reject
        self.submissions = []
        self.aborted: list[str] = []

    async def get_model_config(self):
        return SimpleNamespace(model="model-a")

    async def submit(self, prompt, sampling_params, request_id, **kwargs) -> None:
        if self.reject is not None:
            raise RequestRejectedError(self.reject)
        self.submissions.append((prompt, sampling_params, request_id, kwargs))

    async def stream(self, request_id):
        yield RequestOutput(
            request_id=request_id,
            prompt="ignored",
            prompt_token_ids=[7, 8, 9],
            outputs=[
                CompletionOutput(
                    index=0,
                    text="done",
                    token_ids=[11, 12],
                    cumulative_logprob=0.0,
                )
            ],
            finished=True,
        )

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


def test_chat_uses_full_history_template_sampling_and_usage() -> None:
    previous_engine = api_server.engine
    fake = _ChatEngine()
    api_server.engine = fake  # type: ignore[assignment]
    try:
        response = TestClient(api_server.app).post(
            "/v1/chat/completions",
            json={
                "model": "model-a",
                "messages": [
                    {"role": "system", "content": "rules"},
                    {"role": "user", "content": "hello"},
                ],
                "max_tokens": 2,
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 20,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
            },
        )
    finally:
        api_server.engine = previous_engine

    assert response.status_code == 200
    prompt, sampling_params, _request_id, _kwargs = fake.submissions[0]
    assert prompt == "<chat>system:rules|user:hello"
    assert fake.tokenizer.messages == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hello"},
    ]
    assert sampling_params.top_p == 0.8
    assert sampling_params.top_k == 20
    assert sampling_params.frequency_penalty == 0.1
    assert sampling_params.presence_penalty == 0.2
    assert response.json()["choices"][0]["finish_reason"] == "length"
    assert response.json()["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


def test_chat_rejects_unsupported_sampling_before_submit() -> None:
    previous_engine = api_server.engine
    fake = _ChatEngine()
    api_server.engine = fake  # type: ignore[assignment]
    try:
        response = TestClient(api_server.app).post(
            "/v1/chat/completions",
            json={
                "model": "model-a",
                "messages": [{"role": "user", "content": "hello"}],
                "n": 2,
            },
        )
    finally:
        api_server.engine = previous_engine

    assert response.status_code == 400
    assert fake.submissions == []


def test_chat_maps_admission_rejection_to_429() -> None:
    previous_engine = api_server.engine
    api_server.engine = _ChatEngine(reject="request queue full")  # type: ignore[assignment]
    try:
        response = TestClient(api_server.app).post(
            "/v1/chat/completions",
            json={
                "model": "model-a",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    finally:
        api_server.engine = previous_engine

    assert response.status_code == 429
