# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, List
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.tensor import GGUFWeight, _GLOBAL_WEIGHT_CACHE

def clear_gguf_cache():
    _GLOBAL_WEIGHT_CACHE.clear()

class GGUFConfig(QuantizationConfig):
    def __init__(self):
        super().__init__()
        self.pack_factor = 1

    def get_name(self) -> str: return "gguf"

    def init_layer(self, layer: nn.Module):
        layer.qweight = None
        layer.scales = None
        layer.gguf_quant_type = None
        # Explicitly don't allocate parameters here to save memory

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.tensor import GGUFWeight
        
        weight = getattr(layer, "weight", None)
        # 1. Direct standard attributes
        qweight = getattr(layer, "qweight", None)
        scales = getattr(layer, "scales", None)

        # 2. Attribute Discovery (DeepSeek-V2 / GLM-4.7 Flash Support)
        if qweight is None:
            for attr_name in dir(layer):
                if "weight" in attr_name:
                    val = getattr(layer, attr_name)
                    if isinstance(val, (torch.Tensor, nn.Parameter)) and val.dtype in (torch.int32, torch.uint8, torch.int8):
                        qweight = val; break
        if scales is None:
            for attr_name in dir(layer):
                if "scales" in attr_name:
                    val = getattr(layer, attr_name)
                    if isinstance(val, (torch.Tensor, nn.Parameter)):
                        scales = val; break

        if not isinstance(weight, GGUFWeight):
            if qweight is None:
                # Last resort fallback: check if it's an unquantized layer
                if hasattr(layer, "weight") and not isinstance(layer.weight, (GGUFWeight, type(None))):
                    if layer.weight.numel() > 1:
                        return torch.nn.functional.linear(x, layer.weight, layer.bias)

                raise RuntimeError(
                    f"GGUF weight is not ready for layer '{getattr(layer, 'prefix', '<unknown>')}'. "
                )

            weight = GGUFWeight(
                qweight, 
                scales if scales is not None else torch.ones(1, device=qweight.device), 
                getattr(layer, "gguf_quant_type", None)
            )
            if hasattr(layer, "weight"):
                layer.weight = weight

        return weight.matmul(x, layer.bias)


    def load_weights(self, layer: nn.Module, weights_iter, expert_idx=None, part=None):
        for name, loaded_weight in weights_iter:
            # Lazy Assignment: Directly use the loaded tensor to avoid copy overhead
            if "weight" in name: 
                layer.qweight = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif "scales" in name: 
                layer.scales = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif "bias" in name: 
                layer.bias = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GGUFConfig":
        return cls()
