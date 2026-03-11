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
        device: Optional[str] = None,
    ):
        super().__init__()
        self.input_size = input_size; self.output_size = output_size; self.prefix = prefix
        self.quant_config = quant_config; self._bias_enabled = bias; self.on_load_callback = on_load_callback
        
        target_device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        if "embed_tokens" in prefix or "output" in prefix or "lm_head" in prefix:
            if input_size > 32000 or output_size > 32000: target_device = "cpu"

        if self.quant_config is None:
            self.weight = nn.Parameter(torch.empty(output_size, input_size, device=target_device), requires_grad=False)
        else:
            self.quant_config.init_layer(self)
        
        if not hasattr(self, "weight") and hasattr(self, "qweight"): self.weight = self.qweight
        elif not hasattr(self, "weight"): self.weight = nn.Parameter(torch.empty(output_size, input_size, device=target_device), requires_grad=False)
        if not hasattr(self, "qweight"): self.qweight = self.weight

        if bias: self.bias = nn.Parameter(torch.empty(output_size, device=target_device), requires_grad=False)
        else: self.register_parameter('bias', None)
        
        self.weight_id = id(self); self.adapters: Dict[int, Tuple[torch.Tensor, torch.Tensor, float]] = {}

    def forward(self, x: torch.Tensor, lora_mapping: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Input Tiling / Padding
        if x.shape[-1] != self.input_size:
            if x.shape[-1] < self.input_size: x = torch.nn.functional.pad(x, (0, self.input_size - x.shape[-1]))
            else: x = x[..., :self.input_size]
        
        # 2. Unified Quantized Matmul Path
        from vllm.model_executor.layers.quantization.tensor import QuantizedLinearWeight
        if isinstance(self.weight, QuantizedLinearWeight):
            return self.weight.matmul(x, self.bias)

        if self.quant_config is None:
            w, b = self.weight, self.bias
            if w.device != x.device:
                w = w.to(x.device); b = b.to(x.device) if b is not None else None
            
            # 3. Output Tiling for Large Matrix (Memory Efficiency)
            # If output_size is massive (e.g. LM Head), we split the matmul to keep intermediate memory low
            if self.output_size > 65536 and x.numel() // self.input_size > 1:
                tile_size = 32768
                outputs = []
                for start in range(0, self.output_size, tile_size):
                    end = min(start + tile_size, self.output_size)
                    w_tile = w[start:end, :]
                    b_tile = b[start:end] if b is not None else None
                    outputs.append(torch.nn.functional.linear(x, w_tile, b_tile))
                return torch.cat(outputs, dim=-1)
            
            return torch.nn.functional.linear(x, w, b)
        else:
            return self.quant_config.apply(self, x)
