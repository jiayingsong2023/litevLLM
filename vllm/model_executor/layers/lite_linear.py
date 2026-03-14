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

    def register_lora(self, lora_id: str, lora_a: torch.Tensor, lora_b: torch.Tensor, alpha: float = 1.0):
        """Registers a LoRA adapter for this layer."""
        if not hasattr(self, "_loras"):
            self._loras = {}
        
        rank = lora_a.shape[0]
        scaling = alpha / rank
        self._loras[lora_id] = (lora_a.to(self.weight.device), lora_b.to(self.weight.device), scaling)

    def forward(self, x: torch.Tensor, lora_mapping: Optional[Any] = None) -> torch.Tensor:
        # 1. Base Linear / Quantized Linear
        if getattr(self, "_quant_weight", None) is not None:
            try:
                res = self._quant_weight.matmul(x, getattr(self, 'bias', None))
            except Exception as e: 
                print(f"!!! LiteLinear Quantized Matmul Failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to fp16 matmul if quant fails
                if self.weight.device != x.device or self.weight.dtype != x.dtype:
                    self.weight.data = self.weight.data.to(device=x.device, dtype=x.dtype)
                if getattr(self, "bias", None) is not None and (self.bias.device != x.device or self.bias.dtype != x.dtype):
                    self.bias.data = self.bias.data.to(device=x.device, dtype=x.dtype)
                
                res = torch.nn.functional.linear(x, self.weight, getattr(self, 'bias', None))
        else:
            # AUTO-ALIGNMENT (Lazy & Permanent)
            if self.weight.device != x.device or self.weight.dtype != x.dtype:
                self.weight.data = self.weight.data.to(device=x.device, dtype=x.dtype)
            if getattr(self, "bias", None) is not None and (self.bias.device != x.device or self.bias.dtype != x.dtype):
                self.bias.data = self.bias.data.to(device=x.device, dtype=x.dtype)
            res = torch.nn.functional.linear(x, self.weight, getattr(self, 'bias', None))

        # 2. Apply LoRA if mapping provided
        # lora_mapping: List of lora_id (one per request in batch)
        if lora_mapping is not None and hasattr(self, "_loras"):
            # x shape: [B, S, D] or [Total_Tokens, D]
            # Since LiteEngine handles batching, x is often [B, 1, D] for decode or [1, S, D] for prefill
            
            # For multi-batch LoRA, we check if we have multiple IDs
            if isinstance(lora_mapping, list) and len(lora_mapping) > 1:
                # Multi-batch LoRA logic
                # For each token in the batch, apply its specific LoRA
                # Optimization: Group by LoRA ID
                for l_id in set(lora_mapping):
                    if l_id in self._loras:
                        indices = [i for i, mid in enumerate(lora_mapping) if mid == l_id]
                        indices_t = torch.tensor(indices, device=x.device)
                        
                        a, b, scaling = self._loras[l_id]
                        # Apply LoRA: (x @ a.T) @ b.T * scaling
                        lora_out = (x[indices_t].to(a.dtype) @ a.t()) @ b.t()
                        res[indices_t] += lora_out.to(res.dtype) * scaling
            elif isinstance(lora_mapping, str) or (isinstance(lora_mapping, list) and len(lora_mapping) == 1):
                l_id = lora_mapping if isinstance(lora_mapping, str) else lora_mapping[0]
                if l_id in self._loras:
                    a, b, scaling = self._loras[l_id]
                    lora_out = (x.to(a.dtype) @ a.t()) @ b.t()
                    res += lora_out.to(res.dtype) * scaling

        return res

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
        return f"LiteLinear(in={self.input_size}, out={self.output_size}, quant={self.quant_config.get_name() if self.quant_config else 'None'}, loras={list(self._loras.keys()) if hasattr(self, '_loras') else []})"
