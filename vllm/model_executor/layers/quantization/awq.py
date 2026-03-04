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
            if layer.qweight is None:
                return torch.nn.functional.linear(x, torch.zeros((layer.output_size, layer.input_size), device=x.device, dtype=x.dtype), layer.bias)
            
            qweight = layer.qweight.contiguous()
            scales = layer.scales.contiguous()
            K, M = layer.input_size, layer.output_size
            n_rows_q = qweight.shape[0]
            n_cols_q = qweight.shape[1] * 8
            
            if layer.qzeros is not None:
                qzeros = layer.qzeros.contiguous()
            else:
                fill_value = -2004318072
                if scales.shape[0] == n_rows_q:
                    qzeros = torch.full((n_rows_q, n_cols_q // self.group_size // 8 + 1), fill_value, device=qweight.device, dtype=torch.int32)
                else:
                    qzeros = torch.full((n_rows_q // self.group_size, n_cols_q // 8), fill_value, device=qweight.device, dtype=torch.int32)
            
            dequantized = awq_dequantize_triton(qweight, scales, qzeros, self.group_size)
            if dequantized.shape[0] == K:
                cached_weight = dequantized.transpose(0, 1).contiguous()
            else:
                cached_weight = dequantized.contiguous()
            _GLOBAL_GGUF_CACHE.put(layer.weight_id, cached_weight)

        return torch.nn.functional.linear(x, cached_weight, layer.bias)

    def load_weights(self, layer: nn.Module, weights_iter, expert_idx=None, part=None):
        if expert_idx is not None and layer.qweight is None:
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
