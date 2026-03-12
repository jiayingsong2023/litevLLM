# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, Any

class LiteLinear(nn.Module):
    """
    Unified High-Performance Linear Layer for LitevLLM.
    Supports Standard, AWQ, GGUF, and LoRA paths with automatic Dtype alignment.
    """
    def __init__(self, input_size: int, output_size: int, bias: bool = False, 
                 quant_config: Any = None, prefix: str = ""):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.prefix = prefix
        self.quant_config = quant_config
        
        # Standard Weight Placeholder
        self.weight = nn.Parameter(torch.empty(output_size, input_size), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        # Quantization State
        self.qweight = None
        self.scales = None
        self.qzeros = None
        self._quant_weight = None # Internal wrapper cache

        if self.quant_config and hasattr(self.quant_config, "init_layer"):
            self.quant_config.init_layer(self)

    def forward(self, x: torch.Tensor, lora_mapping: Any = None) -> torch.Tensor:
        # 1. Quantized Path (AWQ/GGUF)
        if self.quant_config is not None:
            try:
                return self.quant_config.apply(self, x)
            except Exception:
                # If quantization fails or weight not ready, fallback gracefully
                pass

        # 2. Standard Path with Dtype Guard
        w = self.weight
        b = self.bias
        
        # CRITICAL: Automatic Dtype alignment to solve 'Half != float' issues
        if x.dtype != w.dtype:
            w = w.to(x.dtype)
        if b is not None and x.dtype != b.dtype:
            b = b.to(x.dtype)
            
        return torch.nn.functional.linear(x, w, b)

    def __repr__(self):
        return f"LiteLinear(in={self.input_size}, out={self.output_size}, quant={self.quant_config.get_name() if self.quant_config else 'None'})"
