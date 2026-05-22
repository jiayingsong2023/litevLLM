# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from fastapi.testclient import TestClient

from vllm.engine import async_llm as async_llm_module
from vllm.engine.runtime_controller import RuntimeController
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

        def register_lora_adapter(
            self,
            *,
            lora_name: str,
            lora_path: str | None = None,
            lora_int_id: int | None = None,
        ):
            return {
                "lora_name": lora_name,
                "lora_path": lora_path,
                "lora_int_id": lora_int_id,
            }

        def unregister_lora_adapter(self, lora_name: str) -> bool:
            return lora_name == "demo"

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
    monkeypatch.setattr(
        async_llm_module, "get_tokenizer", lambda *_args, **_kwargs: object()
    )

    llm = async_llm_module.AsyncLLM(DummyConfig())

    assert llm.stats() == {"scheduler": {"active_request_count": 1}}
    llm.reset_stats(clear_prefix_cache=True)
    assert llm.engine.reset_calls == [True]
    assert llm.register_lora_adapter(
        lora_name="demo", lora_path="/tmp/demo", lora_int_id=7
    ) == {
        "lora_name": "demo",
        "lora_path": "/tmp/demo",
        "lora_int_id": 7,
    }
    assert llm.unregister_lora_adapter("demo") is True


def test_llm_stats_delegates_to_engine() -> None:
    llm = LLM.__new__(LLM)

    class FakeEngine:
        def __init__(self) -> None:
            self.reset_calls = []

        def stats(self) -> dict[str, object]:
            return {"backend": {"backend_type": "fake"}}

        def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
            self.reset_calls.append(clear_prefix_cache)

        def register_lora_adapter(
            self,
            *,
            lora_name: str,
            lora_path: str | None = None,
            lora_int_id: int | None = None,
        ):
            return {
                "lora_name": lora_name,
                "lora_path": lora_path,
                "lora_int_id": lora_int_id,
            }

        def unregister_lora_adapter(self, lora_name: str) -> bool:
            return lora_name == "demo"

    llm.engine = FakeEngine()

    assert llm.stats() == {"backend": {"backend_type": "fake"}}
    llm.reset_stats(clear_prefix_cache=True)
    assert llm.engine.reset_calls == [True]
    assert llm.register_lora_adapter(
        lora_name="demo", lora_path="/tmp/demo", lora_int_id=7
    ) == {
        "lora_name": "demo",
        "lora_path": "/tmp/demo",
        "lora_int_id": 7,
    }
    assert llm.unregister_lora_adapter("demo") is True


def test_runtime_stats_endpoint_returns_engine_snapshot() -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.reset_calls = []

        async def get_model_config(self):
            return type("ModelConfig", (), {"model": "demo-model"})()

        def stats(self) -> dict[str, object]:
            return {
                "profile": {
                    "requested": "auto",
                    "effective": "benchmark",
                    "kv_cache_dtype": "turbo_int4",
                },
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
        "summary": {
            "prefix_cache": {
                "hit_rate": 0.0,
                "avg_saved_prefill_tokens_per_request": 0.0,
            },
            "preemption": {
                "preempted_steps": 0,
                "preempted_prefill_requests": 0,
                "preempted_multimodal_prefill_requests": 0,
                "protected_multimodal_prefix_steps": 0,
                "protected_multimodal_prefix_prefill_requests": 0,
            },
            "fairness": {
                "p95_queue_wait_s": 0.0,
                "max_queue_wait_s": 0.0,
                "fairness_guardrail_triggered_steps": 0,
            },
            "lora": {
                "registered_adapters": 0,
                "active_adapters": 0,
                "total_routed_requests": 0,
                "prefill_locality_rate": 0.0,
                "decode_locality_rate": 0.0,
                "admit_relaxed_steps": 0,
                "admit_tightened_steps": 0,
                "prefill_relaxed_steps": 0,
                "prefill_tightened_steps": 0,
                "decode_relaxed_steps": 0,
                "decode_tightened_steps": 0,
            },
            "multimodal": {
                "requests": 0,
                "images": 0,
                "multimodal_lora_requests": 0,
                "queued_requests": 0,
                "admitted_requests": 0,
                "prefill_requests": 0,
                "prefill_multimodal_lora_requests": 0,
                "decode_requests": 0,
                "p95_queue_wait_s": 0.0,
                "prefix_cache_hits": 0,
                "prefix_cache_misses": 0,
                "prefix_cache_saved_prefill_tokens": 0,
                "prefix_cache_hit_rate": 0.0,
                "mixed_multimodal_lora_prefill_ratio": 0.0,
                "avg_effective_prefill_multimodal_limit": 0.0,
                "prefill_multimodal_limit_relaxed_steps": 0,
                "prefill_multimodal_limit_tightened_steps": 0,
                "prefill_multimodal_limit_triggered_steps": 0,
                "avg_effective_admit_multimodal_lora_limit": 0.0,
                "avg_effective_prefill_multimodal_lora_limit": 0.0,
                "avg_effective_decode_multimodal_lora_limit": 0.0,
                "admit_multimodal_lora_limit_triggered_steps": 0,
                "prefill_multimodal_lora_limit_relaxed_steps": 0,
                "prefill_multimodal_lora_limit_relaxed_by_fairness_steps": 0,
                "prefill_multimodal_lora_limit_tightened_steps": 0,
                "prefill_multimodal_lora_limit_tightened_by_locality_steps": 0,
                "prefill_multimodal_lora_limit_triggered_steps": 0,
                "avg_prefill_multimodal_lora_max_fairness_gap": 0.0,
                "decode_multimodal_lora_limit_relaxed_steps": 0,
                "decode_multimodal_lora_limit_relaxed_by_fairness_steps": 0,
                "decode_multimodal_lora_limit_tightened_steps": 0,
                "decode_multimodal_lora_limit_tightened_by_locality_steps": 0,
                "decode_multimodal_lora_limit_triggered_steps": 0,
                "avg_decode_multimodal_lora_max_fairness_gap": 0.0,
                "prepared_requests": 0,
                "prepared_images": 0,
                "embeddings_computed": 0,
                "avg_embedding_feature_dim": 0.0,
            },
        },
        "stats": {
            "profile": {
                "requested": "auto",
                "effective": "benchmark",
                "kv_cache_dtype": "turbo_int4",
            },
            "scheduler": {"active_request_count": 2},
            "observer": {"step_count": 3},
        },
    }


def test_runtime_controller_stats_include_profile() -> None:
    class FakeProfile:
        def stats(self) -> dict[str, object]:
            return {
                "requested": "auto",
                "effective": "benchmark",
                "kv_cache_dtype": "turbo_int4",
            }

    class FakeRuntimeConfig:
        profile = FakeProfile()

    class FakeScheduler:
        active_request_count = 2
        running_request_count = 1
        queued_request_count = 1
        available_slots = 3
        runtime_config = FakeRuntimeConfig()

    class FakeObserver:
        def stats(self) -> dict[str, object]:
            return {"step_count": 3}

    class FakeBackend:
        def stats(self) -> dict[str, object]:
            return {"backend_type": "fake"}

        def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
            del clear_prefix_cache

    controller = RuntimeController(
        scheduler=FakeScheduler(),
        step_scheduler=object(),
        observer=FakeObserver(),
        backend=FakeBackend(),
        queue_timeout_s=15.0,
    )

    stats = controller.stats()

    assert stats["profile"] == {
        "requested": "auto",
        "effective": "benchmark",
        "kv_cache_dtype": "turbo_int4",
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
        "summary": {
            "prefix_cache": {
                "hit_rate": 0.0,
                "avg_saved_prefill_tokens_per_request": 0.0,
            },
            "preemption": {
                "preempted_steps": 0,
                "preempted_prefill_requests": 0,
                "preempted_multimodal_prefill_requests": 0,
                "protected_multimodal_prefix_steps": 0,
                "protected_multimodal_prefix_prefill_requests": 0,
            },
            "fairness": {
                "p95_queue_wait_s": 0.0,
                "max_queue_wait_s": 0.0,
                "fairness_guardrail_triggered_steps": 0,
            },
            "lora": {
                "registered_adapters": 0,
                "active_adapters": 0,
                "total_routed_requests": 0,
                "prefill_locality_rate": 0.0,
                "decode_locality_rate": 0.0,
                "admit_relaxed_steps": 0,
                "admit_tightened_steps": 0,
                "prefill_relaxed_steps": 0,
                "prefill_tightened_steps": 0,
                "decode_relaxed_steps": 0,
                "decode_tightened_steps": 0,
            },
            "multimodal": {
                "requests": 0,
                "images": 0,
                "multimodal_lora_requests": 0,
                "queued_requests": 0,
                "admitted_requests": 0,
                "prefill_requests": 0,
                "prefill_multimodal_lora_requests": 0,
                "decode_requests": 0,
                "p95_queue_wait_s": 0.0,
                "prefix_cache_hits": 0,
                "prefix_cache_misses": 0,
                "prefix_cache_saved_prefill_tokens": 0,
                "prefix_cache_hit_rate": 0.0,
                "mixed_multimodal_lora_prefill_ratio": 0.0,
                "avg_effective_prefill_multimodal_limit": 0.0,
                "prefill_multimodal_limit_relaxed_steps": 0,
                "prefill_multimodal_limit_tightened_steps": 0,
                "prefill_multimodal_limit_triggered_steps": 0,
                "avg_effective_admit_multimodal_lora_limit": 0.0,
                "avg_effective_prefill_multimodal_lora_limit": 0.0,
                "avg_effective_decode_multimodal_lora_limit": 0.0,
                "admit_multimodal_lora_limit_triggered_steps": 0,
                "prefill_multimodal_lora_limit_relaxed_steps": 0,
                "prefill_multimodal_lora_limit_relaxed_by_fairness_steps": 0,
                "prefill_multimodal_lora_limit_tightened_steps": 0,
                "prefill_multimodal_lora_limit_tightened_by_locality_steps": 0,
                "prefill_multimodal_lora_limit_triggered_steps": 0,
                "avg_prefill_multimodal_lora_max_fairness_gap": 0.0,
                "decode_multimodal_lora_limit_relaxed_steps": 0,
                "decode_multimodal_lora_limit_relaxed_by_fairness_steps": 0,
                "decode_multimodal_lora_limit_tightened_steps": 0,
                "decode_multimodal_lora_limit_tightened_by_locality_steps": 0,
                "decode_multimodal_lora_limit_triggered_steps": 0,
                "avg_decode_multimodal_lora_max_fairness_gap": 0.0,
                "prepared_requests": 0,
                "prepared_images": 0,
                "embeddings_computed": 0,
                "avg_embedding_feature_dim": 0.0,
            },
        },
        "stats": {
            "scheduler": {"active_request_count": 0},
            "observer": {"step_count": 0},
        },
    }


def test_runtime_stats_summary_includes_multimodal_snapshot() -> None:
    summary = api_server._summarize_runtime_stats(
        {
            "observer": {
                "multimodal": {
                    "requests": 2,
                    "images": 2,
                    "multimodal_lora_requests": 1,
                    "queued_requests": 1,
                    "admitted_requests": 2,
                    "prefill_requests": 2,
                    "prefill_multimodal_lora_requests": 1,
                    "decode_requests": 1,
                    "p95_queue_wait_s": 0.4,
                    "mixed_multimodal_lora_prefill_ratio": 0.5,
                    "avg_effective_prefill_multimodal_limit": 2.0,
                    "prefill_multimodal_limit_relaxed_steps": 1,
                    "prefill_multimodal_limit_tightened_steps": 0,
                    "prefill_multimodal_limit_triggered_steps": 1,
                    "avg_effective_admit_multimodal_lora_limit": 1.0,
                    "avg_effective_prefill_multimodal_lora_limit": 1.0,
                    "avg_effective_decode_multimodal_lora_limit": 1.5,
                    "admit_multimodal_lora_limit_triggered_steps": 2,
                    "prefill_multimodal_lora_limit_relaxed_steps": 1,
                    "prefill_multimodal_lora_limit_relaxed_by_fairness_steps": 1,
                    "prefill_multimodal_lora_limit_tightened_steps": 0,
                    "prefill_multimodal_lora_limit_tightened_by_locality_steps": 0,
                    "prefill_multimodal_lora_limit_triggered_steps": 1,
                    "avg_prefill_multimodal_lora_max_fairness_gap": 0.25,
                    "decode_multimodal_lora_limit_relaxed_steps": 1,
                    "decode_multimodal_lora_limit_relaxed_by_fairness_steps": 1,
                    "decode_multimodal_lora_limit_tightened_steps": 0,
                    "decode_multimodal_lora_limit_tightened_by_locality_steps": 0,
                    "decode_multimodal_lora_limit_triggered_steps": 1,
                    "avg_decode_multimodal_lora_max_fairness_gap": 0.2,
                }
            },
            "backend": {
                "multimodal": {
                    "prepared_requests": 2,
                    "prepared_images": 2,
                    "embeddings_computed": 2,
                    "avg_embedding_feature_dim": 32.0,
                }
            },
        }
    )

    assert summary["multimodal"] == {
        "requests": 2,
        "images": 2,
        "multimodal_lora_requests": 1,
        "queued_requests": 1,
        "admitted_requests": 2,
        "prefill_requests": 2,
        "prefill_multimodal_lora_requests": 1,
        "decode_requests": 1,
        "p95_queue_wait_s": 0.4,
        "prefix_cache_hits": 0,
        "prefix_cache_misses": 0,
        "prefix_cache_saved_prefill_tokens": 0,
        "prefix_cache_hit_rate": 0.0,
        "mixed_multimodal_lora_prefill_ratio": 0.5,
        "avg_effective_prefill_multimodal_limit": 2.0,
        "prefill_multimodal_limit_relaxed_steps": 1,
        "prefill_multimodal_limit_tightened_steps": 0,
        "prefill_multimodal_limit_triggered_steps": 1,
        "avg_effective_admit_multimodal_lora_limit": 1.0,
        "avg_effective_prefill_multimodal_lora_limit": 1.0,
        "avg_effective_decode_multimodal_lora_limit": 1.5,
        "admit_multimodal_lora_limit_triggered_steps": 2,
        "prefill_multimodal_lora_limit_relaxed_steps": 1,
        "prefill_multimodal_lora_limit_relaxed_by_fairness_steps": 1,
        "prefill_multimodal_lora_limit_tightened_steps": 0,
        "prefill_multimodal_lora_limit_tightened_by_locality_steps": 0,
        "prefill_multimodal_lora_limit_triggered_steps": 1,
        "avg_prefill_multimodal_lora_max_fairness_gap": 0.25,
        "decode_multimodal_lora_limit_relaxed_steps": 1,
        "decode_multimodal_lora_limit_relaxed_by_fairness_steps": 1,
        "decode_multimodal_lora_limit_tightened_steps": 0,
        "decode_multimodal_lora_limit_tightened_by_locality_steps": 0,
        "decode_multimodal_lora_limit_triggered_steps": 1,
        "avg_decode_multimodal_lora_max_fairness_gap": 0.2,
        "prepared_requests": 2,
        "prepared_images": 2,
        "embeddings_computed": 2,
        "avg_embedding_feature_dim": 32.0,
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
