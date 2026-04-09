# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from fastapi.testclient import TestClient

from vllm.engine import async_llm as async_llm_module
from vllm.entrypoints.llm import LLM
from vllm.entrypoints.openai import api_server


def test_async_llm_stats_delegates_to_engine(monkeypatch) -> None:
    class FakeEngine:
        def __init__(self, _vllm_config) -> None:
            self.tokenizer = None
            self.reset_calls = []

        def stats(self) -> dict[str, object]:
            return {"scheduler": {"active_request_count": 1}}

        def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
            self.reset_calls.append(clear_prefix_cache)

    class FakeDriver:
        def __init__(self, engine) -> None:
            self.engine = engine

        def shutdown(self) -> None:
            return None

    class DummyConfig:
        class ModelConfig:
            model = "dummy"

        model_config = ModelConfig()

    monkeypatch.setattr(async_llm_module, "LiteEngine", FakeEngine)
    monkeypatch.setattr(async_llm_module, "AsyncDriver", FakeDriver)
    monkeypatch.setattr(async_llm_module, "get_tokenizer", lambda *_args, **_kwargs: object())

    llm = async_llm_module.AsyncLLM(DummyConfig())

    assert llm.stats() == {"scheduler": {"active_request_count": 1}}
    llm.reset_stats(clear_prefix_cache=True)
    assert llm.engine.reset_calls == [True]


def test_llm_stats_delegates_to_engine() -> None:
    llm = LLM.__new__(LLM)

    class FakeEngine:
        def __init__(self) -> None:
            self.reset_calls = []

        def stats(self) -> dict[str, object]:
            return {"backend": {"backend_type": "fake"}}

        def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
            self.reset_calls.append(clear_prefix_cache)

    llm.engine = FakeEngine()

    assert llm.stats() == {"backend": {"backend_type": "fake"}}
    llm.reset_stats(clear_prefix_cache=True)
    assert llm.engine.reset_calls == [True]


def test_runtime_stats_endpoint_returns_engine_snapshot() -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.reset_calls = []

        async def get_model_config(self):
            return type("ModelConfig", (), {"model": "demo-model"})()

        def stats(self) -> dict[str, object]:
            return {
                "scheduler": {"active_request_count": 2},
                "observer": {"step_count": 3},
            }

        def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
            self.reset_calls.append(clear_prefix_cache)

    prev_engine = api_server.engine
    api_server.engine = FakeEngine()
    try:
        client = TestClient(api_server.app)
        response = client.get("/debug/runtime_stats")
    finally:
        api_server.engine = prev_engine

    assert response.status_code == 200
    assert response.json() == {
        "model": "demo-model",
        "stats": {
            "scheduler": {"active_request_count": 2},
            "observer": {"step_count": 3},
        },
    }


def test_runtime_stats_reset_endpoint_resets_engine_stats() -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.reset_calls = []

        async def get_model_config(self):
            return type("ModelConfig", (), {"model": "demo-model"})()

        def stats(self) -> dict[str, object]:
            return {
                "scheduler": {"active_request_count": 0},
                "observer": {"step_count": 0},
            }

        def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
            self.reset_calls.append(clear_prefix_cache)

    fake_engine = FakeEngine()
    prev_engine = api_server.engine
    api_server.engine = fake_engine
    try:
        client = TestClient(api_server.app)
        response = client.post("/debug/runtime_stats/reset?clear_prefix_cache=true")
    finally:
        api_server.engine = prev_engine

    assert response.status_code == 200
    assert fake_engine.reset_calls == [True]
    assert response.json() == {
        "model": "demo-model",
        "stats": {
            "scheduler": {"active_request_count": 0},
            "observer": {"step_count": 0},
        },
    }


def test_runtime_stats_endpoint_requires_initialized_engine() -> None:
    prev_engine = api_server.engine
    api_server.engine = None
    try:
        client = TestClient(api_server.app)
        response = client.get("/debug/runtime_stats")
    finally:
        api_server.engine = prev_engine

    assert response.status_code == 503
    assert response.json() == {"detail": "engine not initialized"}
