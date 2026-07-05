# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.engine.errors import RequestRejectedError
from vllm.engine.lite_engine import LiteEngine
from vllm.engine.request_state import RequestState
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams


class _FakeObserver:
    def __init__(self) -> None:
        self.rejections: list[tuple[str, str]] = []

    def on_request_rejected(self, request_id: str, reason: str) -> None:
        self.rejections.append((request_id, reason))

    def on_request_added(self, request_id: str, request_state: RequestState) -> None:
        del request_id, request_state


class _FakeScheduler:
    active_request_count = 0

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, RequestState]] = []

    def has_queue_capacity(self) -> bool:
        return True

    def enqueue_request(self, request_id: str, request_state: RequestState) -> None:
        self.enqueued.append((request_id, request_state))


class _FakeLoRARegistry:
    def __init__(self, request: LoRARequest) -> None:
        self.request = request
        self.added: list[str | None] = []

    def resolve_adapter(self, *, lora_id=None, lora_request=None):
        del lora_id, lora_request
        return self.request

    def on_request_added(self, lora_name: str | None) -> None:
        self.added.append(lora_name)


class _FakeLoRAManager:
    def __init__(self, *, has_adapter: bool) -> None:
        self._has_adapter = has_adapter
        self.registered: list[tuple[str, str | None]] = []

    def has_adapter(self, lora_name: str | None) -> bool:
        del lora_name
        return self._has_adapter

    def register_adapter(self, *, lora_name: str, lora_path: str | None) -> None:
        self.registered.append((lora_name, lora_path))
        self._has_adapter = True


class _FakeRequestBuilder:
    def build(self, **kwargs) -> RequestState:
        return RequestState(
            request_id=kwargs["request_id"],
            prompt=kwargs["prompt"],
            input_ids=[1, 2],
            sampling_params=kwargs["sampling_params"],
            lora_id=kwargs["lora_id"],
            lora_int_id=kwargs["lora_int_id"],
            lora_path=kwargs["lora_path"],
        )


class _FakeMultimodalProcessor:
    def prepare_request(self, request_state: RequestState) -> None:
        del request_state


class _FakeExecutionBackend:
    def maybe_apply_prefix_cache(self, request_state: RequestState) -> None:
        del request_state


def _engine_for_lora_contract(*, supports_lora: bool, manager_has_adapter: bool):
    engine = SimpleNamespace()
    engine.policies = object()
    engine.scheduler = _FakeScheduler()
    engine.adapter = SimpleNamespace()
    engine.lora_registry = _FakeLoRARegistry(
        LoRARequest(lora_name="adapter-a", lora_int_id=7, lora_path="/tmp/adapter")
    )
    engine.lora_manager = _FakeLoRAManager(has_adapter=manager_has_adapter)
    engine.model_capabilities = SimpleNamespace(
        supports_lora=supports_lora,
        supports_multimodal=False,
        supports_chunked_prefill=True,
    )
    engine.request_builder = _FakeRequestBuilder()
    engine.multimodal_processor = _FakeMultimodalProcessor()
    engine.execution_backend = _FakeExecutionBackend()
    engine.observer = _FakeObserver()
    engine.max_active_requests = 2
    return engine


def test_lite_engine_rejects_lora_when_model_does_not_support_it() -> None:
    engine = _engine_for_lora_contract(supports_lora=False, manager_has_adapter=False)

    with pytest.raises(RequestRejectedError, match="model does not support LoRA"):
        LiteEngine.add_request(
            engine,
            "req-1",
            "hello",
            SamplingParams(max_tokens=1),
            lora_request=LoRARequest(
                lora_name="adapter-a", lora_int_id=7, lora_path="/tmp/adapter"
            ),
        )

    assert engine.observer.rejections == [
        ("req-1", "model does not support LoRA")
    ]
    assert engine.scheduler.enqueued == []


def test_lite_engine_lora_request_loads_adapter_before_enqueue() -> None:
    engine = _engine_for_lora_contract(supports_lora=True, manager_has_adapter=False)

    LiteEngine.add_request(
        engine,
        "req-1",
        "hello",
        SamplingParams(max_tokens=1),
        lora_request=LoRARequest(
            lora_name="adapter-a", lora_int_id=7, lora_path="/tmp/adapter"
        ),
    )

    assert engine.lora_manager.registered == [("adapter-a", "/tmp/adapter")]
    assert len(engine.scheduler.enqueued) == 1
    _, request_state = engine.scheduler.enqueued[0]
    assert request_state.lora_id == "adapter-a"
    assert request_state.lora_int_id == 7
    assert request_state.lora_path == "/tmp/adapter"
    assert engine.lora_registry.added == ["adapter-a"]
