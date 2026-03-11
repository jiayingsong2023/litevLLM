# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Union, Optional, List
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.tensor import AWQWeight

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

class AWQConfig(QuantizationConfig):
    def __init__(
        self,
        weight_bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []
        self.pack_factor = 32 // self.weight_bits

    def get_name(self) -> str:
        return "awq"

    def init_layer(self, layer: nn.Module):
        layer.qweight = None
        layer.qzeros = None
        layer.scales = None

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.tensor import AWQWeight
        
        weight = getattr(layer, "weight", None)
        if not isinstance(weight, AWQWeight):
            if getattr(layer, "qweight", None) is None:
                 # Fallback/Safety for empty layers during initialization
                 return torch.nn.functional.linear(x, torch.zeros((layer.output_size, layer.input_size), device=x.device, dtype=x.dtype), layer.bias)
            
            weight = AWQWeight(
                layer.qweight,
                layer.scales,
                getattr(layer, "qzeros", None),
                self.group_size
            )
            # If layer is a real LiteLinear, store it. If mock, just keep it for this call.
            if hasattr(layer, "weight"):
                layer.weight = weight
            
        return weight.matmul(x, layer.bias)

    def load_weights(self, layer: nn.Module, weights_iter, expert_idx=None, part=None):
        if expert_idx is not None and getattr(layer, "qweight", None) is None:
            K, N = layer.input_size, layer.output_size
            layer.qweight = nn.Parameter(torch.zeros((N, K // 8), device="cuda", dtype=torch.int32), requires_grad=False)
            layer.scales = nn.Parameter(torch.zeros((N, K // self.group_size), device="cuda", dtype=torch.float16), requires_grad=False)
            layer.qzeros = None

        for name, loaded_weight in weights_iter:
            if expert_idx is not None:
                num_experts = 256
                if "w1" in layer.prefix:
                    inter_size = layer.output_size // (num_experts * 2)
                    start_n = expert_idx * (inter_size * 2) + (part * inter_size if part is not None else 0)
                    end_n = start_n + inter_size
                else:
                    inter_size = layer.output_size // num_experts
                    start_n = expert_idx * inter_size
                    end_n = start_n + inter_size
                
                if any(x in name for x in ["qweight", "weight_packed"]):
                    layer.qweight.data[start_n : end_n, :].copy_(loaded_weight)
                elif any(x in name for x in ["scales", "weight_scale"]):
                    layer.scales.data[start_n : end_n, :].copy_(loaded_weight)
                elif any(x in name for x in ["qzeros", "weight_zeros"]):
                    if layer.qzeros is None:
                        layer.qzeros = nn.Parameter(torch.zeros((layer.output_size, layer.input_size // self.group_size // 8 + 1), device="cuda", dtype=torch.int32), requires_grad=False)
                    layer.qzeros.data[start_n : end_n, :].copy_(loaded_weight)
            else:
                if any(x in name for x in ["qweight", "weight_packed"]):
                    layer.qweight = nn.Parameter(loaded_weight, requires_grad=False)
                elif any(x in name for x in ["qzeros", "weight_zeros"]):
                    layer.qzeros = nn.Parameter(loaded_weight, requires_grad=False)
                elif any(x in name for x in ["scales", "weight_scale"]):
                    layer.scales = nn.Parameter(loaded_weight, requires_grad=False)
                elif "bias" in name:
                    layer.bias = nn.Parameter(loaded_weight, requires_grad=False)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AWQConfig":
        return cls(
            weight_bits=config.get("bits", 4),
            group_size=config.get("group_size", 128),
            zero_point=config.get("zero_point", True),
        )
