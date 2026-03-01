# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, Any
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class LiteLinear(nn.Module):
    _DEQUANT_CACHE = {}
    _CACHE_ORDER = []
    _MAX_CACHE_SIZE = 128

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
        self.qweight = None
        self.qscales = None
        self.qtype = "q4_0"

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
        
        # Dispatch to quantization config apply
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
