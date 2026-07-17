from __future__ import annotations

import torch

from vllm.model_executor.models.deepseek_v4_flash.kv_lifecycle import (
    DeepSeekKVLifecycleAdapter,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)


def test_model_request_state_map_is_idempotent_and_frees_kv() -> None:
    model = DeepSeekV4FlashForCausalLM()
    device = torch.device("cpu")

    first = model.ensure_request_state(
        request_id="req-1",
        context_length=256,
        device=device,
        max_active_requests=2,
    )
    again = model.ensure_request_state(
        request_id="req-1",
        context_length=256,
        device=device,
        max_active_requests=2,
    )

    assert again is first
    assert model.get_request_state("req-1") is first
    stats = model.kv_stats()
    assert stats["active_requests"] == 1
    assert stats["active_requests_high_water"] == 1

    model.free_request_state("req-1")

    stats = model.kv_stats()
    assert stats["active_requests"] == 0
    assert stats["active_requests_high_water"] == 1


def test_kv_lifecycle_adapter_creates_state_before_ensuring_capacity() -> None:
    model = DeepSeekV4FlashForCausalLM()
    adapter = DeepSeekKVLifecycleAdapter(
        model=model,
        context_length=256,
        device=torch.device("cpu"),
        max_active_requests=2,
    )

    adapter.ensure_blocks_for_requests(["req-1"], [3])

    state = model.get_request_state("req-1")
    assert state.request_id == "req-1"
    assert state.config.context_length == 256
    assert state.kv_cache._pool.max_requests == 2

    adapter.free_request_blocks("req-1")

    assert model.kv_stats()["active_requests"] == 0


def test_kv_lifecycle_adapter_records_family_allocation_event() -> None:
    class Observer:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def on_deepseek_event(self, event: str, **payload: object) -> None:
            self.events.append((event, payload))

    model = DeepSeekV4FlashForCausalLM()
    observer = Observer()
    adapter = DeepSeekKVLifecycleAdapter(
        model=model,
        context_length=256,
        device=torch.device("cpu"),
        max_active_requests=2,
        observer=observer,
    )

    adapter.ensure_blocks_for_requests(["req-1"], [3])

    assert observer.events == [
        (
            "kv_family_allocation",
            {
                "request_id": "req-1",
                "token_count": 3,
                "active_requests": 1,
                "active_requests_high_water": 1,
            },
        )
    ]
