# SPDX-License-Identifier: Apache-2.0
"""Lightweight import smoke for LoRA wiring (no adapter weights required)."""


def test_lora_request_symbol_importable() -> None:
    from vllm.lora.request import LoRARequest

    assert LoRARequest is not None
