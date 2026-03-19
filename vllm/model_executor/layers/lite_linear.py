# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, Tuple

class LiteLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = True, quant_config: Optional[dict] = None, prefix: str = ""):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.prefix = prefix
        self.quant_config = quant_config
        
        # Lazy initialization: Parameter exists but is empty until loaded
        self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.weight.numel() == 0:
            # Fallback for empty/unloaded weights
            return torch.zeros((*x.shape[:-1], self.output_size), device=x.device, dtype=x.dtype)
            
        return torch.nn.functional.linear(x, self.weight, getattr(self, 'bias', None))

    def get_fast_data(self) -> Tuple[str, tuple]:
        return "dense", (self.weight, getattr(self, 'bias', None))
