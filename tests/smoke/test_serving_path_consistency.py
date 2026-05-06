# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from fastapi import FastAPI

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
