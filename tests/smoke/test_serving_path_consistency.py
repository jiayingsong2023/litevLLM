# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve import register_vllm_serve_api_routers
from vllm.entrypoints.serve.tokenize.api_router import attach_router


def _route_paths(app: FastAPI) -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


def test_serve_registration_attaches_tokenize_routes() -> None:
    app = FastAPI()

    register_vllm_serve_api_routers(app)

    paths = _route_paths(app)
    assert "/tokenize" in paths
    assert "/detokenize" in paths


def test_direct_tokenize_attach_matches_serve_registration() -> None:
    direct = FastAPI()
    through_serve = FastAPI()

    attach_router(direct)
    register_vllm_serve_api_routers(through_serve)

    assert {"/tokenize", "/detokenize"} <= _route_paths(direct)
    assert _route_paths(direct) == _route_paths(through_serve)


class _Tokenizer:
    model_max_length = 32

    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        return [1] + [ord(char) for char in prompt] if add_special_tokens else [
            ord(char) for char in prompt
        ]

    def convert_ids_to_tokens(self, tokens: list[int]) -> list[str]:
        return [str(token) for token in tokens]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens if token != 1)


def test_tokenize_router_uses_attached_app_tokenizer() -> None:
    app = FastAPI()
    app.state.tokenizer = _Tokenizer()
    attach_router(app)
    client = TestClient(app)

    tokenized = client.post(
        "/tokenize", json={"prompt": "ab", "return_token_strs": True}
    )
    assert tokenized.status_code == 200
    assert tokenized.json() == {
        "tokens": [1, 97, 98],
        "token_strs": ["1", "97", "98"],
        "count": 3,
        "max_model_len": 32,
    }

    detokenized = client.post("/detokenize", json={"tokens": [1, 97, 98]})
    assert detokenized.status_code == 200
    assert detokenized.json() == {"prompt": "ab"}
