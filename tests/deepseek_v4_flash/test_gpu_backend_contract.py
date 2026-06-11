from __future__ import annotations

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
    DeepSeekV4FlashGPUCapabilities,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)


def test_gpu_backend_reports_missing_required_kernels() -> None:
    caps = DeepSeekV4FlashGPUCapabilities(
        q8_linear=True,
        attention=False,
        compressed_attention=False,
        cache_update=False,
        moe=False,
        output=False,
    )
    backend = DeepSeekV4FlashGPUBackend(capabilities=caps)

    assert backend.is_ready is False
    assert backend.missing_kernels == (
        "attention",
        "compressed_attention",
        "cache_update",
        "moe",
        "output",
    )
    with pytest.raises(RuntimeError, match="missing GPU kernels"):
        backend.require_ready()


def test_gpu_backend_can_report_ready_when_all_kernels_are_enabled() -> None:
    caps = DeepSeekV4FlashGPUCapabilities(
        q8_linear=True,
        attention=True,
        compressed_attention=True,
        cache_update=True,
        moe=True,
        output=True,
    )
    backend = DeepSeekV4FlashGPUBackend(capabilities=caps)

    assert backend.is_ready is True
    assert backend.missing_kernels == ()
    backend.require_ready()


def test_model_keeps_kernel_execution_disabled_until_backend_ready() -> None:
    model = DeepSeekV4FlashForCausalLM(gpu_backend=DeepSeekV4FlashGPUBackend())

    assert model.kernel_execution_available is False
    with pytest.raises(NotImplementedError, match="kernel execution is not available"):
        model.forward_full(torch.tensor([1], dtype=torch.long), use_kernel=True)
