# SPDX-License-Identifier: Apache-2.0
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
        
        # LoRA 权重占位符
        self.lora_a: Optional[torch.Tensor] = None
        self.lora_b: Optional[torch.Tensor] = None
        self.lora_scaling: float = 1.0

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

    def set_lora(self, lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float = 1.0):
        """为当前层注入 LoRA 权重"""
        self.lora_a = lora_a # [rank, in]
        self.lora_b = lora_b # [out, rank]
        self.lora_scaling = scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 主路径 (Base Weight)
        if self.quant_config is None:
            res = torch.nn.functional.linear(x, self.weight, self.bias)
        else:
            res = self.quant_config.apply(self, x)
            
        # 2. LoRA 旁路路径 (X @ A.T @ B.T * scaling)
        if self.lora_a is not None and self.lora_b is not None:
            # 这里的计算量非常小 (rank 通常为 8 或 16)
            lora_res = torch.nn.functional.linear(x, self.lora_a)
            lora_res = torch.nn.functional.linear(lora_res, self.lora_b)
            res += lora_res * self.lora_scaling
            
        return res

    def load_weights(self, weights_iter):
        if self.quant_config is None:
            for name, loaded_weight in weights_iter:
                if "weight" in name:
                    self.weight.data.copy_(loaded_weight)
                elif "bias" in name and self.bias is not None:
                    self.bias.data.copy_(loaded_weight)
        else:
            self.quant_config.load_weights(self, weights_iter)
