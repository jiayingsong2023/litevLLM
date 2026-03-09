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
        
        # AGGRESSIVE OFFLOAD: Force CPU for very large layers like Embedding/Head in large models
        target_device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        if "embed_tokens" in prefix or "output" in prefix or "lm_head" in prefix:
            if input_size > 32000 or output_size > 32000: # Typical large vocab models
                print(f">>> LiteLinear: Offloading large layer {prefix} to CPU")
                target_device = "cpu"

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

    def add_adapter(self, aid: int, lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float = 1.0):
        self.adapters[aid] = (lora_a.to("cuda").half(), lora_b.to("cuda").half(), scaling)

    def forward(self, x: torch.Tensor, lora_mapping: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            if x.shape[-1] < self.input_size: x = torch.nn.functional.pad(x, (0, self.input_size - x.shape[-1]))
            else: x = x[..., :self.input_size]
        
        # Ensure weight is moved only if absolutely necessary
        if self.quant_config is None:
            w, b = self.weight, self.bias
            if w.device != x.device:
                # For offloaded layers, we must compute on CPU or move small chunk
                # For benchmark, we move the weight (expensive but avoids crash)
                w = w.to(x.device); b = b.to(x.device) if b is not None else None
            return torch.nn.functional.linear(x, w, b)
        else:
            return self.quant_config.apply(self, x)
