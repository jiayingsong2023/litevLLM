# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    _compressed_tensors_high_fidelity_enabled,
)


def test_compressed_tensors_high_fidelity_default_off() -> None:
    assert _compressed_tensors_high_fidelity_enabled() is False


def test_compressed_tensors_high_fidelity_uses_kernel_policy() -> None:
    assert (
        _compressed_tensors_high_fidelity_enabled(
            {"kernel_policy": {"compressed_tensors_high_fidelity": True}}
        )
        is True
    )
