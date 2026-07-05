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
    assert model.kv_stats()["active_requests"] == 1

    model.free_request_state("req-1")

    assert model.kv_stats()["active_requests"] == 0


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
