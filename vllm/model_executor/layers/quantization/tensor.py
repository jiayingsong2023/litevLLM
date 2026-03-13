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

_GLOBAL_WEIGHT_CACHE = LRUWeightCache(max_size=1024)

class QuantizedLinearWeight(nn.Module, ABC):
    def __init__(self): super().__init__()
    @abstractmethod
    def matmul(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor: pass

class GGUFWeight(QuantizedLinearWeight):
    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, quant_type: int = 2, prefer_fused: bool = True):
        super().__init__()
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.scales = nn.Parameter(scales, requires_grad=False)
        self.quant_type = quant_type
        self.prefer_fused = prefer_fused
        self.weight_id = id(self)

    def matmul(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        from vllm.kernels.triton.gguf_q4_0_dequant import gguf_q4_0_dequant
        
        # Determine logical shape from physical qweight
        n_rows = self.qweight.shape[0]
        # 18 bytes = 32 weights
        n_cols = (self.qweight.shape[1] // 18) * 32
        
        # PURE GPU PATH: Dequantize on-the-fly using Triton
        # For small batch, we use the cache. For large batch or large model, we dequantize every step.
        bs = x.shape[0] if x.dim() > 1 else 1
        
        if self.prefer_fused or bs > 8:
            # Full GPU dequantization to avoid Error 700 from CPU sync
            # Note: In future v2.2, this will be a fused GEMM kernel.
            # Currently it is a fused Tiling closure in model.py.
            w = gguf_q4_0_dequant(self.qweight, n_rows, n_cols)
            return torch.nn.functional.linear(x, w, bias)

        cached_w = _GLOBAL_WEIGHT_CACHE.get(self.weight_id)
        if cached_w is None:
            cached_w = gguf_q4_0_dequant(self.qweight, n_rows, n_cols)
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
