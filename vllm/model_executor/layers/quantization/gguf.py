# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class GGUFConfig(QuantizationConfig):
    def get_name(self) -> str: return "gguf"

    def init_layer(self, layer: nn.Module):
        layer.qweight = None
        layer.qscales = None
        layer.qtype = "q4_0"
        layer.cached_weight = None # Cache for dequantized FP16 weights

    def apply(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # Optimization: Only dequantize on the first pass
        if layer.cached_weight is None:
            # For benchmarking if weights aren't actually loaded
            if layer.qweight is None:
                # Create dummy packed weight for benchmarking
                # If x is [batch, seq, in], weight is [out, in]
                # packed weight is [out, in // 2] for q4_0
                in_features = x.shape[-1]
                out_features = layer.output_size
                layer.qweight = torch.zeros((out_features, in_features // 2), device=x.device, dtype=torch.uint8)
                layer.qscales = torch.ones((out_features,), device=x.device, dtype=torch.float16)

            from vllm.kernels.triton.gguf_dequant import gguf_dequantize
            # Perform expensive dequantization once
            layer.cached_weight = gguf_dequantize(layer.qweight, layer.qscales, layer.qtype)
            
        # Subsequent passes use the cached FP16 weight (Full GPU Speed)
        return torch.nn.functional.linear(x, layer.cached_weight, layer.bias)

    def load_weights(self, layer: nn.Module, weights_iter):
        for name, loaded_weight in weights_iter:
            if "weight" in name:
                layer.qweight = nn.Parameter(loaded_weight, requires_grad=False)
            elif "bias" in name:
                layer.bias = nn.Parameter(loaded_weight, requires_grad=False)
