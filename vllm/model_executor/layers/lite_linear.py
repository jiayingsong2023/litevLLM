# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, Any, Dict, Tuple
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class LiteLinear(nn.Module):
    """
    High-performance pure Python Linear layer.
    Optimized for GGUF and AWQ backends.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.prefix = prefix
        self.quant_config = quant_config
        
        if self.quant_config is None:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            if bias: self.bias = nn.Parameter(torch.empty(output_size))
            else: self.register_parameter('bias', None)
        else:
            self.quant_config.init_layer(self)
            if bias: self.bias = nn.Parameter(torch.empty(output_size))
            else: self.register_parameter('bias', None)

        self.weight_id = id(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_config is None:
            return torch.nn.functional.linear(x, self.weight, self.bias)
        return self.quant_config.apply(self, x)

    def load_weights(self, weights_iter, expert_idx=None, part=None):
        if self.quant_config is None:
            for name, loaded_weight in weights_iter:
                if "weight" in name: self.weight.data.copy_(loaded_weight)
                elif "bias" in name and self.bias is not None: self.bias.data.copy_(loaded_weight)
        else:
            # Direct delegate to quantization backend
            if hasattr(self.quant_config, "load_weights"):
                import inspect
                sig = inspect.signature(self.quant_config.load_weights)
                if "expert_idx" in sig.parameters:
                    self.quant_config.load_weights(self, weights_iter, expert_idx=expert_idx, part=part)
                else:
                    self.quant_config.load_weights(self, weights_iter)
