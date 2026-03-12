# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, List
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.tensor import GGUFWeight, _GLOBAL_WEIGHT_CACHE

def clear_gguf_cache():
    _GLOBAL_WEIGHT_CACHE.clear()

class GGUFConfig(QuantizationConfig):
    def __init__(self, prefer_fused: bool = False):
        super().__init__()
        self.pack_factor = 1
        self.prefer_fused = prefer_fused

    def get_name(self) -> str: return "gguf"

    def init_layer(self, layer: nn.Module):
        layer.qweight = None
        layer.scales = None
        layer.gguf_quant_type = None
        layer._quant_weight = None

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        weight = getattr(layer, "_quant_weight", None)
        if weight is None:
            qweight = getattr(layer, "qweight", None)
            if qweight is None:
                # Fallback to standard linear if normal weight exists
                if hasattr(layer, "weight") and layer.weight is not None and layer.weight.numel() > 1:
                    return torch.nn.functional.linear(x, layer.weight, layer.bias)
                raise RuntimeError(f"GGUF weight not ready for layer '{getattr(layer, 'prefix', '<unknown>')}'")
            
            # Wrap with our balanced GGUFWeight
            weight = GGUFWeight(
                qweight, 
                getattr(layer, "scales", torch.ones(1, device=qweight.device)), 
                getattr(layer, "gguf_quant_type", None),
                prefer_fused=self.prefer_fused
            )
            layer._quant_weight = weight
            
        return weight.matmul(x, layer.bias)

    def load_weights(self, layer: nn.Module, weights_iter, expert_idx=None, part=None):
        for name, loaded_weight in weights_iter:
            if "weight" in name: 
                layer.qweight = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif "scales" in name: 
                layer.scales = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif "bias" in name: 
                layer.bias = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GGUFConfig":
        return cls()
