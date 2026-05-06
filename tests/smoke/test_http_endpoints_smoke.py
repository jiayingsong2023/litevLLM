# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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
