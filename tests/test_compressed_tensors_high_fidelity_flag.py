# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    _compressed_tensors_high_fidelity_enabled,
)


def test_compressed_tensors_high_fidelity_default_off(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_COMPRESSED_TENSORS_HIGH_FIDELITY", raising=False)
    assert _compressed_tensors_high_fidelity_enabled() is False


def test_compressed_tensors_high_fidelity_on(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_COMPRESSED_TENSORS_HIGH_FIDELITY", "1")
    assert _compressed_tensors_high_fidelity_enabled() is True
