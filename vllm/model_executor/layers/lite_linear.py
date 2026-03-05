# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, Any, Dict, Tuple, Callable
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class LiteLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        on_load_callback: Optional[Callable] = None,
    ):
        super().__init__()
        self.input_size = input_size; self.output_size = output_size; self.prefix = prefix
        self.quant_config = quant_config; self._bias_enabled = bias; self.on_load_callback = on_load_callback
        
        if self.quant_config is None:
            self.weight = nn.Parameter(torch.empty(output_size, input_size), requires_grad=False)
            if bias: self.bias = nn.Parameter(torch.empty(output_size), requires_grad=False)
            else: self.register_parameter('bias', None)
        else:
            self.quant_config.init_layer(self)
            if bias: self.bias = nn.Parameter(torch.empty(output_size), requires_grad=False)
            else: self.register_parameter('bias', None)
        self.weight_id = id(self)

    def forward(self, x: torch.Tensor, lora_mapping: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            if x.shape[-1] < self.input_size: x = torch.nn.functional.pad(x, (0, self.input_size - x.shape[-1]))
            else: x = x[..., :self.input_size]
        if self.quant_config is None: return torch.nn.functional.linear(x, self.weight, self.bias)
        return self.quant_config.apply(self, x)

    def load_weights(self, weights_iter, expert_idx=None, part=None):
        # Peak at weight dimensions to detect forced quantization
        for name, loaded_weight in list(weights_iter):
            # If loaded weight is 1/8 size of expected, it's packed AWQ
            if self.quant_config is None and (loaded_weight.shape[0] * 8 == self.output_size or loaded_weight.shape[1] * 8 == self.input_size):
                # EMERGENCY: Layer was initialized as FP16 but weights are AWQ
                print(f">>> LiteLinear: Emergency Quantization conversion for {self.prefix}")
                from vllm.model_executor.layers.quantization.awq import AWQConfig
                self.quant_config = AWQConfig()
                self.quant_config.init_layer(self)

        if self.quant_config is not None:
            if hasattr(self.quant_config, "load_weights"):
                import inspect; sig = inspect.signature(self.quant_config.load_weights)
                if "expert_idx" in sig.parameters: self.quant_config.load_weights(self, weights_iter, expert_idx=expert_idx, part=part)
                else: self.quant_config.load_weights(self, weights_iter)
            return

        for name, loaded_weight in weights_iter:
            # Dimension Healing
            w_out, w_in = loaded_weight.shape[0], loaded_weight.shape[1]
            if w_out != self.output_size and expert_idx is None:
                self.output_size = w_out
                if self._bias_enabled and self.bias is not None: self.bias = nn.Parameter(torch.zeros(w_out, device=loaded_weight.device, dtype=torch.float16), requires_grad=False)
            if w_in != self.input_size: self.input_size = w_in
            if self.on_load_callback: self.on_load_callback(self.output_size, self.input_size)
            if "weight" in name: self.weight.data.copy_(loaded_weight)
            elif "bias" in name and self.bias is not None: self.bias.data.copy_(loaded_weight)
