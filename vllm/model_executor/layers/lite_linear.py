# SPDX-License-Identifier: Apache-2.0
"""LitevLLM: Unified Linear Layer with Quantization and Triton support."""

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any, Type
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class LiteLinear(nn.Module):
    """
    A single-GPU optimized linear layer that handles all quantization types.
    Removes distributed overhead and uses direct Triton/Torch kernels.
    """
    __slots__ = ("input_size", "output_size", "bias", "weight", "quant_config", "xtype")

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = ""
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.quant_config = quant_config
        
        # Standard Weight Initialization (will be overwritten by loader)
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float16),
            requires_grad=False
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(output_size, dtype=torch.float16),
                requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass. 
        Dispatches to Triton kernels if quantized, else use native torch.
        """
        if self.quant_config is None:
            # Native high-performance path for unquantized weights
            return torch.matmul(x, self.weight.t()) if self.bias is None else torch.addmm(self.bias, x, self.weight.t())
        
        # Quantized path: delegated to quantization-specific kernels
        return self.quant_config.apply(self.weight, x, bias=self.bias)

    def extra_repr(self) -> str:
        return f"in={self.input_size}, out={self.output_size}, bias={self.bias is not None}, quant={self.quant_config is not None}"
