# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from vllm.entrypoints.openai import api_server


def test_openai_chat_endpoint_reports_uninitialized_engine() -> None:
    old_engine = api_server.engine
    api_server.engine = None
    try:
        response = TestClient(api_server.app).post(
            "/v1/chat/completions",
            json={
                "model": "missing",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
            },
        )
    finally:
        api_server.engine = old_engine

    assert response.status_code == 503
    assert response.json()["detail"] == "engine not initialized"


def test_openai_models_endpoint_reports_uninitialized_engine() -> None:
    old_engine = api_server.engine
    api_server.engine = None
    try:
        response = TestClient(api_server.app).get("/v1/models")
    finally:
        api_server.engine = old_engine

    assert response.status_code == 503
    assert response.json()["detail"] == "engine not initialized"


def test_health_and_readiness_contract() -> None:
    previous_engine = api_server.engine
    api_server.engine = None
    try:
        client = TestClient(api_server.app)
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").status_code == 503
    finally:
        api_server.engine = previous_engine


def test_server_lifespan_shuts_down_initialized_engine() -> None:
    previous_engine = api_server.engine
    shutdowns: list[bool] = []
    api_server.engine = SimpleNamespace(shutdown=lambda: shutdowns.append(True))
    try:
        with TestClient(api_server.app):
            pass
    finally:
        api_server.engine = previous_engine

    assert shutdowns == [True]


def test_debug_endpoints_are_disabled_by_default() -> None:
    previous_debug = api_server.debug_endpoints_enabled
    api_server.debug_endpoints_enabled = False
    try:
        response = TestClient(api_server.app).get("/debug/runtime_stats")
    finally:
        api_server.debug_endpoints_enabled = previous_debug

    assert response.status_code == 404


def test_chat_request_body_limit_is_enforced_before_json_parsing() -> None:
    previous_engine = api_server.engine
    api_server.engine = object()  # type: ignore[assignment]
    try:
        response = TestClient(api_server.app).post(
            "/v1/chat/completions",
            content=b"x" * (api_server.MAX_REQUEST_BODY_BYTES + 1),
            headers={"content-type": "application/json"},
        )
    finally:
        api_server.engine = previous_engine

    assert response.status_code == 413
