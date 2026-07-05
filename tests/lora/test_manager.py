# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

from vllm.lora.manager import LoRAManager
from vllm.lora.weights import LoRALayerWeights


def test_manager_applies_single_adapter_delta() -> None:
    manager = LoRAManager()
    manager.add_adapter_weights(
        "adapter-a",
        {
            "proj": LoRALayerWeights(
                lora_name="adapter-a",
                rank=2,
                alpha=4,
                lora_a=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                lora_b=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            )
        },
    )

    x = torch.tensor([[[2.0, 3.0]]])

    delta = manager.compute_delta(
        target_name="proj",
        x=x,
        lora_mapping=["adapter-a"],
    )

    expected = x.reshape(-1, 2) @ torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = expected @ torch.tensor([[5.0, 6.0], [7.0, 8.0]]) * 2.0
    assert torch.allclose(delta, expected.reshape(1, 1, 2))


def test_manager_rejects_mixed_adapter_delta() -> None:
    manager = LoRAManager()

    with pytest.raises(ValueError, match="mixed LoRA"):
        manager.compute_delta(
            target_name="proj",
            x=torch.ones(1, 1, 2),
            lora_mapping=["adapter-a", "adapter-b"],
        )
