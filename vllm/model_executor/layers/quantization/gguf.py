# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Dict, OrderedDict
from collections import OrderedDict
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class GGUFWeightCache:
    """Global LRU Cache for dequantized GGUF weights to prevent OOM."""
    def __init__(self, max_cache_size: int = 256):
        self.cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.max_size = max_cache_size

    def get(self, weight_id: int) -> torch.Tensor:
        if weight_id in self.cache:
            self.cache.move_to_end(weight_id)
            return self.cache[weight_id]
        return None

    def put(self, weight_id: int, weight: torch.Tensor):
        if weight_id in self.cache:
            self.cache.move_to_end(weight_id)
            return
        
        if len(self.cache) >= self.max_size:
            # print(f"DEBUG: GGUF Cache Full, evicting oldest")
            self.cache.popitem(last=False)
            
        self.cache[weight_id] = weight
    
    def clear(self):
        self.cache.clear()
        torch.cuda.empty_cache()

# Global instance
_GLOBAL_GGUF_CACHE = GGUFWeightCache(max_cache_size=384) # Optimized for large MoE at BS=32

def clear_gguf_cache():
    _GLOBAL_GGUF_CACHE.clear()

class GGUFConfig(QuantizationConfig):
    def get_name(self) -> str: return "gguf"

    def init_layer(self, layer: nn.Module):
        layer.qweight = None
        layer.qscales = None
        layer.qtype = "q4_0"
        layer.weight_id = id(layer) 

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        cached_weight = _GLOBAL_GGUF_CACHE.get(layer.weight_id)
        
        if cached_weight is None:
            if layer.qweight is None:
                in_features = x.shape[-1]
                out_features = layer.output_size
                layer.qweight = nn.Parameter(
                    torch.zeros((out_features, in_features // 2), device=x.device, dtype=torch.uint8),
                    requires_grad=False
                )
                layer.qscales = nn.Parameter(
                    torch.ones((out_features,), device=x.device, dtype=torch.float16),
                    requires_grad=False
                )

            from vllm.kernels.triton.gguf_dequant import gguf_dequantize
            cached_weight = gguf_dequantize(layer.qweight, layer.qscales, layer.qtype)
            _GLOBAL_GGUF_CACHE.put(layer.weight_id, cached_weight)
            
        return torch.nn.functional.linear(x, cached_weight, layer.bias)

    def load_weights(self, layer: nn.Module, weights_iter):
        for name, loaded_weight in weights_iter:
            if "weight" in name:
                layer.qweight = nn.Parameter(loaded_weight, requires_grad=False)
            elif "bias" in name:
                layer.bias = nn.Parameter(loaded_weight, requires_grad=False)
            elif "scales" in name:
                layer.qscales = nn.Parameter(loaded_weight, requires_grad=False)
