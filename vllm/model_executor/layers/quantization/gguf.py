# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, List
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.kernels.triton.gguf_dequant import gguf_dequantize

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

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        cached_weight = _GLOBAL_GGUF_CACHE.get(layer.weight_id)
        if cached_weight is None:
            if layer.qweight is None: return torch.zeros((x.shape[0], layer.output_size), device=x.device, dtype=x.dtype)
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
