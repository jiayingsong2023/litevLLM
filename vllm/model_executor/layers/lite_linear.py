# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Optional, Any, Dict, Tuple, Callable
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class LiteLinear(nn.Module):
    """
    Final LiteLinear with Index-Add LoRA Support.
    """
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
        self.adapters: Dict[int, Tuple[torch.Tensor, torch.Tensor, float]] = {}

    def add_adapter(self, aid: int, lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float = 1.0):
        self.adapters[aid] = (lora_a.to("cuda").half(), lora_b.to("cuda").half(), scaling)

    def set_lora(self, lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float = 1.0):
        self.add_adapter(aid=1, lora_a=lora_a, lora_b=lora_b, scaling=scaling)

    def forward(self, x: torch.Tensor, lora_mapping: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            if x.shape[-1] < self.input_size: x = torch.nn.functional.pad(x, (0, self.input_size - x.shape[-1]))
            else: x = x[..., :self.input_size]
        
        if self.quant_config is None: res = torch.nn.functional.linear(x, self.weight, self.bias)
        else: res = self.quant_config.apply(self, x)
            
        # --- ROBUST MULTI-LORA DISPATCH ---
        if lora_mapping is not None and self.adapters:
            orig_shape = res.shape
            res_2d = res.view(-1, orig_shape[-1])
            x_2d = x.view(-1, x.shape[-1])
            n_tokens = res_2d.shape[0]
            m_1d = lora_mapping.flatten()
            bsz = m_1d.shape[0]
            
            factor = n_tokens // bsz if n_tokens > bsz else 1

            for aid in torch.unique(m_1d):
                aid_int = int(aid.item())
                if aid_int == 0: continue
                
                adapter = self.adapters.get(aid_int)
                if adapter:
                    la, lb, scaling = adapter
                    if factor > 1: mask = (m_1d.repeat_interleave(factor) == aid_int)
                    else: mask = (m_1d == aid_int)
                    
                    if mask.any() and mask.shape[0] == n_tokens:
                        # (X_masked @ A.T) @ B.T
                        lora_delta = (x_2d[mask].to(torch.float32) @ la.t().to(torch.float32)) @ lb.t().to(torch.float32)
                        indices = torch.nonzero(mask).flatten()
                        res_2d.index_add_(0, indices, lora_delta.to(res_2d.dtype), alpha=scaling)
            
            return res_2d.view(orig_shape)
                    
        return res

    def load_weights(self, weights_iter, expert_idx=None, part=None):
        for name, loaded_weight in list(weights_iter):
            if self.quant_config is None and (loaded_weight.shape[0] * 8 == self.output_size or loaded_weight.shape[1] * 8 == self.input_size):
                from vllm.model_executor.layers.quantization.awq import AWQConfig
                self.quant_config = AWQConfig(); self.quant_config.init_layer(self)
        if self.quant_config is not None:
            if hasattr(self.quant_config, "load_weights"):
                import inspect; sig = inspect.signature(self.quant_config.load_weights)
                if "expert_idx" in sig.parameters: self.quant_config.load_weights(self, weights_iter, expert_idx=expert_idx, part=part)
                else: self.quant_config.load_weights(self, weights_iter)
            return
        for name, loaded_weight in weights_iter:
            w_out, w_in = loaded_weight.shape[0], loaded_weight.shape[1]
            if w_out != self.output_size and expert_idx is None:
                self.output_size = w_out
                if self._bias_enabled and self.bias is not None: self.bias = nn.Parameter(torch.zeros(w_out, device=loaded_weight.device, dtype=torch.float16), requires_grad=False)
            if w_in != self.input_size: self.input_size = w_in
            if self.on_load_callback: self.on_load_callback(self.output_size, self.input_size)
            if "weight" in name: self.weight.data.copy_(loaded_weight)
            elif "bias" in name and self.bias is not None: self.bias.data.copy_(loaded_weight)
