# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, List
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.kernels.triton.gguf_dequant import gguf_dequantize
from vllm.model_executor.layers.quantization.gguf_kernels import ggml_mul_mat_a8_fallback

class GGUFWeightCache:
    def __init__(self, max_size=128):
        self.cache = {}
        self.max_size = max_size
    def get(self, key): return self.cache.get(key)
    def put(self, key, value):
        if len(self.cache) >= self.max_size: self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
    def clear(self): self.cache.clear()

_GLOBAL_GGUF_CACHE = GGUFWeightCache(max_size=256)

def clear_gguf_cache(): _GLOBAL_GGUF_CACHE.clear()

class GGUFConfig(QuantizationConfig):
    def __init__(self):
        super().__init__()
        self.pack_factor = 1

    def get_name(self) -> str: return "gguf"

    def init_layer(self, layer: nn.Module):
        layer.qweight = None; layer.qzeros = None; layer.scales = None; layer.weight_id = id(layer)
        layer.gguf_quant_type = None
        layer.gguf_shape = None

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if getattr(layer, "gguf_quant_type", None) is not None:
            if layer.qweight is None:
                raise RuntimeError(
                    f"GGUF packed tensor is not ready for layer '{getattr(layer, 'prefix', '<unknown>')}'."
                )
            if layer.qweight.dim() != 2:
                raise RuntimeError(
                    f"GGUF packed tensor for layer '{getattr(layer, 'prefix', '<unknown>')}' "
                    "is not 2D and cannot be applied via LiteLinear."
                )
            original_shape = x.shape
            x_2d = x.view(-1, original_shape[-1])
            compute_dtype = getattr(layer, "gguf_compute_dtype", x.dtype)
            if not isinstance(compute_dtype, torch.dtype):
                compute_dtype = x.dtype
            out_2d = ggml_mul_mat_a8_fallback(
                layer.qweight,
                x_2d.to(compute_dtype),
                int(layer.gguf_quant_type),
                int(layer.qweight.shape[0]),
            )
            if layer.bias is not None:
                out_2d = out_2d + layer.bias.to(out_2d.dtype)
            output_dtype = getattr(layer, "gguf_output_dtype", x.dtype)
            out_2d = out_2d.to(output_dtype)
            return out_2d.view(*original_shape[:-1], out_2d.shape[-1])

        cached_weight = _GLOBAL_GGUF_CACHE.get(layer.weight_id)
        if cached_weight is None:
            if layer.qweight is None or layer.scales is None:
                raise RuntimeError(
                    f"GGUF weight is not ready for layer '{getattr(layer, 'prefix', '<unknown>')}'. "
                    "Refusing to run fallback computation."
                )
            cached_weight = gguf_dequantize(layer.qweight, layer.scales, 2) # Default Q4_K
            _GLOBAL_GGUF_CACHE.put(layer.weight_id, cached_weight)
        return torch.nn.functional.linear(x, cached_weight, layer.bias)

    def load_weights(self, layer: nn.Module, weights_iter, expert_idx=None, part=None):
        for name, loaded_weight in weights_iter:
            if "weight" in name: layer.qweight = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif "scales" in name: layer.scales = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)
            elif "bias" in name: layer.bias = nn.Parameter(loaded_weight.contiguous(), requires_grad=False)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GGUFConfig":
        return cls()
