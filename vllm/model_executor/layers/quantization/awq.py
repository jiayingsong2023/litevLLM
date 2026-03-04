# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Union, Optional, List
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton, awq_gemm_triton

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
        layer.weight_id = id(layer)

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.gguf import _GLOBAL_GGUF_CACHE
        
        cached_weight = _GLOBAL_GGUF_CACHE.get(layer.weight_id)
        
        if cached_weight is None:
            # High-performance path: Hybrid Selection
            if x.shape[0] > 1:
                # AMD Stability Patch: Ensure metadata before dequant
                # Handle potential None if weights weren't loaded correctly
                if layer.qweight is None:
                    return torch.nn.functional.linear(x, torch.zeros((layer.output_size, layer.input_size), device=x.device, dtype=x.dtype), layer.bias)
                
                qweight = layer.qweight.contiguous()
                scales = layer.scales.contiguous()
                # Some AWQ models don't have zeros (symmetric)
                qzeros = layer.qzeros.contiguous() if layer.qzeros is not None else torch.zeros((qweight.shape[0] // self.group_size, qweight.shape[1]), device=qweight.device, dtype=torch.int32)
                
                cached_weight = awq_dequantize_triton(
                    qweight, scales, qzeros
                ).transpose(0, 1) 
                _GLOBAL_GGUF_CACHE.put(layer.weight_id, cached_weight)
            else:
                # Single-token fused path
                if layer.qweight is None: return torch.zeros_like(x)
                orig_shape = x.shape
                x_2d = x.reshape(-1, orig_shape[-1])
                qzeros = layer.qzeros if layer.qzeros is not None else torch.zeros((layer.qweight.shape[0] // self.group_size, layer.qweight.shape[1]), device=x.device, dtype=torch.int32)
                out = awq_gemm_triton(x_2d, layer.qweight, layer.scales, qzeros, split_k_iters=1)
                return out.reshape(orig_shape[:-1] + (out.shape[-1],))

        return torch.nn.functional.linear(x, cached_weight, layer.bias)

    def load_weights(self, layer: nn.Module, weights_iter):
        for name, loaded_weight in weights_iter:
            # Support both standard AWQ and Qwen-style Safetensors naming
            if any(x in name for x in ["qweight", "weight_packed"]):
                layer.qweight = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif any(x in name for x in ["qzeros", "weight_zeros"]): # Some use weight_zeros
                layer.qzeros = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif any(x in name for x in ["scales", "weight_scale"]):
                layer.scales = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif "bias" in name:
                layer.bias = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AWQConfig":
        return cls(
            weight_bits=config.get("bits", 4),
            group_size=config.get("group_size", 128),
            zero_point=config.get("zero_point", True),
        )
