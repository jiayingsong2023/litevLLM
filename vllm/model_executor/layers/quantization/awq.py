# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, List
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class AWQConfig(QuantizationConfig):
    def __init__(self, weight_bits: int = 4, group_size: int = 128):
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.pack_factor = 32 // weight_bits

    def get_name(self) -> str: return "awq"

    def init_layer(self, layer: nn.Module):
        layer.qweight = None
        layer.scales = None
        layer.qzeros = None
        layer._quant_weight = None

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.tensor import AWQWeight
        
        # Check if already built
        weight = getattr(layer, "_quant_weight", None)
        if weight is None:
            qweight = getattr(layer, "qweight", None)
            if qweight is None:
                # Fallback to standard if possible
                if hasattr(layer, "weight") and layer.weight is not None and layer.weight.numel() > 1:
                    return torch.nn.functional.linear(x, layer.weight, layer.bias)
                raise RuntimeError(f"AWQ weight not ready for layer '{getattr(layer, 'prefix', '<unknown>')}'")

            # Build the wrapper once
            weight = AWQWeight(
                qweight, 
                getattr(layer, "scales"), 
                getattr(layer, "qzeros"),
                getattr(layer, "group_size", 128)
            )
            layer._quant_weight = weight

        return weight.matmul(x, layer.bias)

    def load_weights(self, layer: nn.Module, weights_iter, expert_idx=None, part=None):
        for name, loaded_weight in weights_iter:
            if "qweight" in name: layer.qweight = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif "scales" in name: layer.scales = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif "qzeros" in name: layer.qzeros = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif "bias" in name: layer.bias = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQConfig":
        return cls()
