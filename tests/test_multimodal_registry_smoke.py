# SPDX-License-Identifier: Apache-2.0
"""Lightweight import smoke for multimodal registry (no vision weights required)."""


def test_multimodal_registry_importable() -> None:
    from vllm.multimodal import MULTIMODAL_REGISTRY

    assert MULTIMODAL_REGISTRY is not None
