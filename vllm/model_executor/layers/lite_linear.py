# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, Any

class LiteLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False, 
                 quant_config: Any = None, prefix: str = ""):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.prefix = prefix
        self.quant_config = quant_config
        
        self.weight = nn.Parameter(torch.empty(output_size, input_size), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        self._quant_weight = None 

        if self.quant_config and hasattr(self.quant_config, "init_layer"):
            self.quant_config.init_layer(self)

    def forward(self, x: torch.Tensor, lora_mapping: Any = None) -> torch.Tensor:
        if self.quant_config is not None:
            try: return self.quant_config.apply(self, x)
            except Exception: pass

        w = self.weight
        if x.dtype != w.dtype: w = w.to(x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)

    @torch.inference_mode()
    def get_fast_data(self):
        """Returns unified raw data and type tag."""
        from vllm.model_executor.layers.quantization.tensor import AWQWeight, GGUFWeight
        qw = getattr(self, "_quant_weight", None)
        if isinstance(qw, AWQWeight):
            return "awq", (qw.qweight, qw.scales, qw.qzeros, qw.group_size)
        if isinstance(qw, GGUFWeight):
            # GGUF d: (qweight, scales, quant_type, prefer_fused)
            return "gguf", (qw.qweight, qw.scales, qw.quant_type, qw.prefer_fused)
        return "fp16", (self.weight, None, None, None)

    def __repr__(self):
        return f"LiteLinear(in={self.input_size}, out={self.output_size}, quant={self.quant_config.get_name() if self.quant_config else 'None'})"
