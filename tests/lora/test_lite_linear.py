# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.lora.manager import LoRAManager
from vllm.lora.weights import LoRALayerWeights
from vllm.model_executor.layers.lite_linear import LiteLinear


def test_lite_linear_adds_lora_delta() -> None:
    layer = LiteLinear(2, 2, bias=False, prefix="proj")
    layer.weight = torch.nn.Parameter(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        requires_grad=False,
    )
    manager = LoRAManager()
    manager.add_adapter_weights(
        "adapter-a",
        {
            "proj": LoRALayerWeights(
                lora_name="adapter-a",
                rank=1,
                alpha=2,
                lora_a=torch.tensor([[1.0], [1.0]]),
                lora_b=torch.tensor([[3.0, 5.0]]),
            )
        },
    )
    manager.bind_to_model(layer)

    x = torch.tensor([[[2.0, 4.0]]])

    out = layer(x, ["adapter-a"])

    base = torch.nn.functional.linear(x, layer.weight)
    delta = torch.tensor([[[36.0, 60.0]]])
    assert torch.allclose(out, base + delta)
