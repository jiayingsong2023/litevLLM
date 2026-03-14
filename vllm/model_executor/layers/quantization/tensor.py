# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

class LRUWeightCache:
    def __init__(self, max_size=256):
        self.cache = {}
        self.max_size = max_size
    def get(self, key): return self.cache.get(key)
    def put(self, key, value):
        if len(self.cache) >= self.max_size: self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
    def clear(self): self.cache.clear()

_GLOBAL_WEIGHT_CACHE = LRUWeightCache(max_size=256)

def clear_global_weight_cache():
    _GLOBAL_WEIGHT_CACHE.clear()
    torch.cuda.empty_cache()

class QuantizedLinearWeight(nn.Module, ABC):
    def __init__(self): super().__init__()
    @abstractmethod
    def matmul(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor: pass

class GGUFWeight(QuantizedLinearWeight):
    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, quant_type: int = 2, 
                 prefer_fused: bool = True, original_shape: Optional[torch.Size] = None,
                 slice_offset: int = 0):
        super().__init__()
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.scales = nn.Parameter(scales, requires_grad=False)
        self.quant_type = quant_type
        self.prefer_fused = prefer_fused
        self.weight_id = id(self)
        self.original_shape = original_shape 
        self.slice_offset = slice_offset

    def matmul(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Determine logical shape
        n_rows = self.qweight.shape[0]
        # Q4_0: 18 bytes per 32 weights. Q4_K: 144 bytes per 256 weights.
        if self.quant_type == 2: # Q4_0
            n_cols = (self.qweight.shape[1] // 18) * 32
        elif self.quant_type == 12: # Q4_K
            n_cols = (self.qweight.shape[1] // 144) * 256
        else:
            # Fallback for unknown types (assuming similar dense packing)
            n_cols = self.qweight.shape[1] * 2 
        
        bs = x.shape[0] if x.dim() > 1 else 1
        
        # 1. Performance Path: Use Fused Kernels for large batches or if preferred
        # SLICING IN FUSED KERNELS NOT SUPPORTED YET - FALLBACK TO CACHE IF SHAPE MISMATCH OR OFFSET
        shape_matches = (self.original_shape is None or (n_rows == self.original_shape[0] and n_cols == self.original_shape[1]))
        no_offset = (self.slice_offset == 0)
        
        if (self.prefer_fused or bs > 16) and shape_matches and no_offset:
            try:
                if self.quant_type == 2:
                    from vllm.kernels.triton.gguf_q4_0_dequant import gguf_q4_0_dequant
                    w = gguf_q4_0_dequant(self.qweight, n_rows, n_cols)
                    return torch.nn.functional.linear(x, w, bias)
                elif self.quant_type == 12:
                    from vllm.kernels.triton.gguf_dequant import dequant_q4_k_triton
                    w = dequant_q4_k_triton(self.qweight, n_rows, n_cols)
                    return torch.nn.functional.linear(x, w, bias)
            except Exception:
                pass # Fallback to Cache

        # 2. Cache Path: Use LRU Cache for decoded weights
        cached_w = _GLOBAL_WEIGHT_CACHE.get(self.weight_id)
        if cached_w is None:
            # Dequantize to GPU Memory
            if self.quant_type == 2:
                from vllm.kernels.triton.gguf_q4_0_dequant import gguf_q4_0_dequant
                cached_w = gguf_q4_0_dequant(self.qweight, n_rows, n_cols)
            elif self.quant_type == 12:
                from vllm.kernels.triton.gguf_dequant import dequant_q4_k_triton
                cached_w = dequant_q4_k_triton(self.qweight, n_rows, n_cols)
            else:
                # GPU-Native Fallback
                q_flat = self.qweight.view(-1)
                low = (q_flat & 0x0F).to(torch.float16)
                high = (q_flat >> 4).to(torch.float16)
                cached_w = torch.stack([low, high], dim=1).view(n_rows, n_cols)
                cached_w = (cached_w - 8.0) * self.scales.unsqueeze(-1)
            
            # SLICE IF NEEDED
            if self.original_shape is not None or self.slice_offset > 0:
                out_size = self.original_shape[0] if self.original_shape else n_rows
                in_size = self.original_shape[1] if self.original_shape else n_cols
                out_start = self.slice_offset
                out_end = out_start + out_size
                if out_end <= cached_w.shape[0] and in_size <= cached_w.shape[1]:
                    cached_w = cached_w[out_start:out_end, :in_size].contiguous()
            
            _GLOBAL_WEIGHT_CACHE.put(self.weight_id, cached_w)
        
        return torch.nn.functional.linear(x, cached_w, bias)

class AWQWeight(QuantizedLinearWeight):
    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, qzeros: Optional[torch.Tensor], group_size: int = 128):
        super().__init__()
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.scales = nn.Parameter(scales, requires_grad=False)
        self.qzeros = nn.Parameter(qzeros, requires_grad=False) if qzeros is not None else None
        self.group_size = group_size
        self.weight_id = id(self)

    def matmul(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        bs = x.shape[0] if x.dim() > 1 else 1
        if bs >= 16:
            from vllm.kernels.triton.awq_fused_gemm import awq_fused_gemm
            qz = self.qzeros
            if qz is None: qz = torch.full((self.qweight.shape[0], self.qweight.shape[1]*8//self.group_size//8+1), -2004318072, device=self.qweight.device, dtype=torch.int32)
            out = awq_fused_gemm(x.view(-1, x.shape[-1]), self.qweight, self.scales, qz, self.group_size)
            return out.view(*x.shape[:-1], out.shape[-1]) + (bias if bias is not None else 0)

        cached_w = _GLOBAL_WEIGHT_CACHE.get(self.weight_id)
        if cached_w is None:
            from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton
            qz = self.qzeros
            if qz is None: qz = torch.full((self.qweight.shape[0], self.qweight.shape[1]*8//self.group_size//8+1), -2004318072, device=self.qweight.device, dtype=torch.int32)
            dequant = awq_dequantize_triton(self.qweight, self.scales, qz, self.group_size, out_dtype=torch.float16)
            cached_w = dequant.transpose(0, 1).contiguous() if dequant.shape[0] == x.shape[-1] else dequant.contiguous()
            _GLOBAL_WEIGHT_CACHE.put(self.weight_id, cached_w)
        return torch.nn.functional.linear(x, cached_w, bias)
