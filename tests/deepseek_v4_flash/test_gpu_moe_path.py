from __future__ import annotations

import pytest
import torch

from vllm.kernels.triton.deepseek_v4_flash.moe import (
    DeepSeekV4MoEKernelInputs,
    deepseek_v4_moe,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_deepseek_gpu_moe_accumulates_selected_experts_on_device() -> None:
    device = torch.device("cuda")
    hidden = torch.tensor([1.0, -2.0], device=device)
    expert_ids = torch.tensor([0, 2], dtype=torch.int32, device=device)
    expert_weights = torch.tensor([0.25, 0.75], dtype=torch.float32, device=device)
    expert_outputs = torch.tensor(
        [
            [2.0, 4.0],
            [100.0, 100.0],
            [-1.0, 3.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    output = deepseek_v4_moe(
        DeepSeekV4MoEKernelInputs(
            hidden=hidden,
            expert_ids=expert_ids,
            expert_weights=expert_weights,
            expert_outputs=expert_outputs,
        )
    )

    expected = 0.25 * expert_outputs[0] + 0.75 * expert_outputs[2]
    assert output.device.type == "cuda"
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_backend_moe_requires_gpu_tensors() -> None:
    backend = DeepSeekV4FlashGPUBackend()

    with pytest.raises(ValueError, match="must be CUDA tensors"):
        backend.routed_moe(
            hidden=torch.zeros(2),
            expert_ids=torch.tensor([0], dtype=torch.int32),
            expert_weights=torch.tensor([1.0]),
            expert_outputs=torch.zeros(1, 2),
        )
