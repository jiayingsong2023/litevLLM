from __future__ import annotations

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)


def test_deepseek_model_declares_reference_and_kernel_execution_boundaries() -> None:
    model = DeepSeekV4FlashForCausalLM()

    assert model.reference_execution_available is True
    assert model.kernel_execution_available is False


def test_deepseek_kernel_forward_fails_explicitly_until_wired() -> None:
    model = DeepSeekV4FlashForCausalLM()

    with pytest.raises(NotImplementedError, match="kernel execution"):
        model.forward_full(torch.tensor([1], dtype=torch.long), use_kernel=True)
