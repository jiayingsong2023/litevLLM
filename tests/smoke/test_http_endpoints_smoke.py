# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator

from fastapi.testclient import TestClient

from vllm.entrypoints.openai import api_server as openai_api_server
from vllm.entrypoints.serve import register_vllm_serve_api_routers
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeResponse,
    TokenizeResponse,
)
from vllm.outputs import CompletionOutput, RequestOutput


class _FakeModelConfig:
    model = "fake-http-smoke-model"


class _FakeEngine:
    async def get_model_config(self):
        return _FakeModelConfig()

    async def generate(
        self, prompt: str, sampling_params, request_id: str, **kwargs
    ) -> AsyncGenerator[RequestOutput, None]:
        del sampling_params, kwargs
        text = f"echo:{prompt}"
        completion = CompletionOutput(
            index=0, text=text, token_ids=[1, 2, 3], cumulative_logprob=0.0
        )
        yield RequestOutput(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=[10, 11],
            outputs=[completion],
            finished=True,
        )


class _FakeTokenizationServing:
    async def create_tokenize(self, request, raw_request):
        del raw_request
        prompt = getattr(request, "prompt", None)
        if prompt is not None:
            token_count = len(prompt.split())
        else:
            token_count = len(getattr(request, "messages", []))
        tokens = list(range(1, token_count + 1))
        return TokenizeResponse(
            count=len(tokens),
            max_model_len=2048,
            tokens=tokens,
            token_strs=[str(token) for token in tokens],
        )

    async def create_detokenize(self, request, raw_request):
        del raw_request
        text = " ".join(str(token) for token in request.tokens)
        return DetokenizeResponse(prompt=text)


def _ensure_tokenize_router_registered(app) -> None:
    existing_paths = {route.path for route in app.router.routes}
    if "/tokenize" in existing_paths and "/detokenize" in existing_paths:
        return
    register_vllm_serve_api_routers(app)


def test_http_smoke_for_tokenize_detokenize_and_chat_completions():
    app = openai_api_server.app
    openai_api_server.engine = _FakeEngine()
    app.state.openai_serving_tokenization = _FakeTokenizationServing()
    _ensure_tokenize_router_registered(app)

    with TestClient(app) as client:
        tokenize_response = client.post(
            "/tokenize",
            json={"model": "fake-http-smoke-model", "prompt": "hello world"},
        )
        assert tokenize_response.status_code == 200
        assert tokenize_response.json()["count"] == 2

        detokenize_response = client.post(
            "/detokenize",
            json={"model": "fake-http-smoke-model", "tokens": [7, 8, 9]},
        )
        assert detokenize_response.status_code == 200
        assert detokenize_response.json()["prompt"] == "7 8 9"

        chat_response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-http-smoke-model",
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "max_tokens": 8,
            },
        )
        assert chat_response.status_code == 200
        body = chat_response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "echo:ping"
