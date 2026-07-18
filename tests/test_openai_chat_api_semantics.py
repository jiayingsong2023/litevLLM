# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from vllm.engine.errors import (
    EngineFatalError,
    InvalidRequestError,
    RequestRejectedError,
)
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
    def __init__(self, *, reject: BaseException | None = None) -> None:
        self.tokenizer = _TemplateTokenizer()
        self.engine = SimpleNamespace(tokenizer=self.tokenizer)
        self.reject = reject
        self.submissions = []
        self.aborted: list[str] = []

    async def get_model_config(self):
        return SimpleNamespace(model="model-a")

    async def submit(self, prompt, sampling_params, request_id, **kwargs) -> None:
        if self.reject is not None:
            raise self.reject
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
    api_server.engine = _ChatEngine(reject=RequestRejectedError("request queue full"))  # type: ignore[assignment]
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


def test_chat_maps_invalid_request_and_fatal_errors_by_type() -> None:
    previous_engine = api_server.engine
    try:
        api_server.engine = _ChatEngine(reject=InvalidRequestError("invalid image"))  # type: ignore[assignment]
        invalid = TestClient(api_server.app).post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [{"role": "user", "content": "x"}]},
        )
        api_server.engine = _ChatEngine(reject=EngineFatalError("engine is fatal"))  # type: ignore[assignment]
        fatal = TestClient(api_server.app).post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [{"role": "user", "content": "x"}]},
        )
    finally:
        api_server.engine = previous_engine

    assert invalid.status_code == 400
    assert fatal.status_code == 503


def test_server_main_leaves_toml_model_options_unoverridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        model="model-a",
        host="127.0.0.1",
        port=8000,
        enable_debug_endpoints=False,
        trust_remote_code=None,
        revision=None,
        policy_mode="auto",
    )
    captured: dict[str, object] = {}
    previous_engine = api_server.engine
    monkeypatch.setattr(
        api_server.argparse.ArgumentParser, "parse_args", lambda _self: args
    )
    monkeypatch.setattr(
        api_server,
        "build_vllm_config",
        lambda model, **kwargs: (
            captured.update(model=model, **kwargs) or SimpleNamespace()
        ),
    )
    monkeypatch.setattr(api_server, "AsyncLLM", lambda _config: object())
    monkeypatch.setattr(api_server.uvicorn, "run", lambda *_args, **_kwargs: None)
    try:
        api_server.main()
    finally:
        api_server.engine = previous_engine

    assert captured == {"model": "model-a", "policy_mode": "auto"}
