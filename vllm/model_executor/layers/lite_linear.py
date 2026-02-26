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

        # Initialization logic for different quantization types
        if self.quant_config is None:
            # Standard unquantized weight
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size, device="cuda", dtype=self.params_dtype)
            )
        else:
            # For GGUF/GPTQ/AWQ, weights are managed by their respective configs
            self.quant_config.init_layer(self)

        if bias:
            self.bias = nn.Parameter(
                torch.empty(output_size, device="cuda", dtype=self.params_dtype)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass logic:
        1. If unquantized: Use standard torch.addmm or F.linear.
        2. If quantized: Dispatch to the specific quantization kernel (Triton).
        """
        if self.quant_config is None:
            return torch.nn.functional.linear(x, self.weight, self.bias)
        
        # Quantized path: delegated to the quantization config implementation
        # This typically calls a Triton kernel (e.g., for GGUF or AWQ)
        return self.quant_config.apply(self, x)

    def load_weights(self, weights_iter):
        """Standardized weight loader for LitevLLM."""
        if self.quant_config is None:
            # Basic loading for unquantized weights
            for name, loaded_weight in weights_iter:
                if "weight" in name:
                    self.weight.data.copy_(loaded_weight)
                elif "bias" in name and self.bias is not None:
                    self.bias.data.copy_(loaded_weight)
        else:
            # Quantization-specific loading (e.g., loading packed int32 for GPTQ)
            self.quant_config.load_weights(self, weights_iter)