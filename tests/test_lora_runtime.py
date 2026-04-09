# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from vllm.engine.lora_runtime import LoRARuntimeRegistry
from vllm.lora.request import LoRARequest


def test_lora_runtime_registry_registers_and_routes_explicit_adapter() -> None:
    registry = LoRARuntimeRegistry()

    request = registry.register_adapter(
        lora_name="demo",
        lora_path="/tmp/demo",
        lora_int_id=7,
    )
    resolved = registry.resolve_adapter(lora_id="demo")

    assert request.lora_name == "demo"
    assert request.lora_int_id == 7
    assert request.lora_path == "/tmp/demo"
    assert resolved is request


def test_lora_runtime_registry_auto_registers_lora_request() -> None:
    registry = LoRARuntimeRegistry()

    resolved = registry.resolve_adapter(
        lora_request=LoRARequest(
            lora_name="adapter-a",
            lora_int_id=9,
            lora_path="/tmp/adapter-a",
        )
    )

    assert resolved is not None
    assert resolved.lora_name == "adapter-a"
    assert registry.stats()["registered_adapters"] == 1


def test_lora_runtime_registry_tracks_request_counts() -> None:
    registry = LoRARuntimeRegistry()
    registry.register_adapter(lora_name="demo")

    registry.on_request_added("demo")
    registry.on_request_added("demo")
    registry.on_request_removed("demo")

    assert registry.stats() == {
        "registered_adapters": 1,
        "active_adapters": 1,
        "active_requests": 1,
        "total_routed_requests": 2,
        "adapters": {
            "demo": {
                "lora_int_id": registry.resolve_adapter(lora_id="demo").lora_int_id,
                "lora_path": None,
                "active_requests": 1,
                "total_requests": 2,
            }
        },
    }


def test_lora_runtime_registry_rejects_unregistered_adapter() -> None:
    registry = LoRARuntimeRegistry()

    with pytest.raises(ValueError, match="not registered"):
        registry.resolve_adapter(lora_id="missing")


def test_lora_runtime_registry_rejects_unregister_while_active() -> None:
    registry = LoRARuntimeRegistry()
    registry.register_adapter(lora_name="demo")
    registry.on_request_added("demo")

    with pytest.raises(ValueError, match="while 1 requests are active"):
        registry.unregister_adapter("demo")
