# SPDX-License-Identifier: Apache-2.0
"""
LiteLinear: A unified Linear layer for single-GPU inference.
This layer replaces ColumnParallelLinear and RowParallelLinear, removing
all distributed/TP overhead and supporting multiple quantization formats.
"""
import torch
import torch.nn as nn
from typing import Optional, Any
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class LiteLinear(nn.Module):
    # Global LRU Cache for Dequantized Weights (Shared across layers to manage memory)
    # Key: (id(layer), qtype), Value: dequantized_weight_tensor
    _DEQUANT_CACHE = {}
    _CACHE_ORDER = []
    _MAX_CACHE_SIZE = 128 # Covers most 7B-13B model layers

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.params_dtype = params_dtype or torch.get_default_dtype()
        self.quant_config = quant_config
        self.prefix = prefix

        if self.quant_config is None:
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, device="cuda", dtype=self.params_dtype)
            )
        else:
            self.quant_config.init_layer(self)

        if bias:
            self.bias = nn.Parameter(
                torch.empty(output_size, device="cuda", dtype=self.params_dtype)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_config is None:
            return torch.nn.functional.linear(x, self.weight, self.bias)
        
        # GGUF Weight Caching Logic with LRU
        if self.quant_config.get_name() == "gguf":
            layer_id = id(self)
            if layer_id in LiteLinear._DEQUANT_CACHE:
                # Cache Hit: Move to end (most recent)
                weight = LiteLinear._DEQUANT_CACHE[layer_id]
                LiteLinear._CACHE_ORDER.remove(layer_id)
                LiteLinear._CACHE_ORDER.append(layer_id)
            else:
                # Cache Miss: Dequantize and Cache
                from vllm.kernels.triton.gguf_dequant import gguf_dequantize
                weight = gguf_dequantize(self.qweight, self.qscales, self.qtype)
                
                # Manage Cache Size
                if len(LiteLinear._DEQUANT_CACHE) >= LiteLinear._MAX_CACHE_SIZE:
                    oldest_id = LiteLinear._CACHE_ORDER.pop(0)
                    del LiteLinear._DEQUANT_CACHE[oldest_id]
                
                LiteLinear._DEQUANT_CACHE[layer_id] = weight
                LiteLinear._CACHE_ORDER.append(layer_id)
            
            return torch.nn.functional.linear(x, weight, self.bias)

        return self.quant_config.apply(self, x)

    def load_weights(self, weights_iter):
        if self.quant_config is None:
            for name, loaded_weight in weights_iter:
                if "weight" in name:
                    self.weight.data.copy_(loaded_weight)
                elif "bias" in name and self.bias is not None:
                    self.bias.data.copy_(loaded_weight)
        else:
            self.quant_config.load_weights(self, weights_iter)
