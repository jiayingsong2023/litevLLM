# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

class LRUWeightCache:
    def __init__(self, max_size=256):
        self.cache = {}
        self.max_size = max_size

    def get(self, key):
        return self.cache.get(key)

    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

    def clear(self):
        self.cache.clear()

_GLOBAL_WEIGHT_CACHE = LRUWeightCache(max_size=1024)

class QuantizedLinearWeight(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def matmul(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

class GGUFWeight(QuantizedLinearWeight):
    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, quant_type: Optional[int] = None, prefer_fused: bool = False):
        super().__init__()
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.scales = nn.Parameter(scales, requires_grad=False)
        self.quant_type = quant_type
        self.prefer_fused = prefer_fused
        self.weight_id = id(self)

    def matmul(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        from vllm.kernels.triton.gguf_dequant import gguf_dequantize
        from vllm.model_executor.layers.quantization.gguf_kernels import ggml_mul_mat_a8_fallback

        bs = x.shape[0] if x.dim() > 1 else 1
        
        # Memory-Saving Strategy: Real-time Dequantization for Large Models or Large Batch
        if self.prefer_fused or bs > 8:
            # We treat quant_type as Q4_K (2) if None
            qtype = int(self.quant_type) if self.quant_type is not None else 2
            original_shape = x.shape
            x_2d = x.view(-1, original_shape[-1])
            out_2d = ggml_mul_mat_a8_fallback(
                self.qweight,
                x_2d,
                qtype,
                int(self.qweight.shape[0]),
            )
            out = out_2d.view(*original_shape[:-1], out_2d.shape[-1])
            return out + bias if bias is not None else out

        # Performance Strategy: FP8 Cache for Latency-sensitive Small Models
        cached_weight = _GLOBAL_WEIGHT_CACHE.get(self.weight_id)
        if cached_weight is None:
            # Default to Q4_K (2)
            cached_weight = gguf_dequantize(self.qweight, self.scales, 2)
            _GLOBAL_WEIGHT_CACHE.put(self.weight_id, cached_weight)
        
        return torch.nn.functional.linear(x, cached_weight, bias)

class AWQWeight(QuantizedLinearWeight):
    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, qzeros: Optional[torch.Tensor], group_size: int = 128):
        super().__init__()
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.scales = nn.Parameter(scales, requires_grad=False)
        if qzeros is not None:
            self.qzeros = nn.Parameter(qzeros, requires_grad=False)
        else:
            self.register_parameter('qzeros', None)
        self.group_size = group_size
        self.weight_id = id(self)

    def matmul(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        bs = x.shape[0] if x.dim() > 1 else 1
        
        if bs >= 16:
            from vllm.kernels.triton.awq_fused_gemm import awq_fused_gemm
            qzeros = self.qzeros
            if qzeros is None:
                n_rows_q = self.qweight.shape[0]
                n_cols_q = self.qweight.shape[1] * 8
                qzeros = torch.full((n_rows_q, n_cols_q // self.group_size // 8 + 1), -2004318072, device=self.qweight.device, dtype=torch.int32)
            out = awq_fused_gemm(x.view(-1, x.shape[-1]), self.qweight, self.scales, qzeros, self.group_size)
            out = out.view(*x.shape[:-1], out.shape[-1])
            return out + bias if bias is not None else out

        from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton
        cached_weight = _GLOBAL_WEIGHT_CACHE.get(self.weight_id)
        if cached_weight is None:
            qweight = self.qweight.contiguous(); scales = self.scales.contiguous()
            input_size = x.shape[-1]; qzeros = self.qzeros.contiguous() if self.qzeros is not None else None
            if qzeros is None:
                n_rows_q = qweight.shape[0]; n_cols_q = qweight.shape[1] * 8
                qzeros = torch.full((n_rows_q, n_cols_q // self.group_size // 8 + 1), -2004318072, device=qweight.device, dtype=torch.int32)
            out_dtype = torch.float16 # Fixed for Stability on APU
            dequantized = awq_dequantize_triton(qweight, scales, qzeros, self.group_size, out_dtype=out_dtype)
            cached_weight = dequantized.transpose(0, 1).contiguous() if dequantized.shape[0] == input_size else dequantized.contiguous()
            _GLOBAL_WEIGHT_CACHE.put(self.weight_id, cached_weight)
        
        return torch.nn.functional.linear(x, cached_weight, bias)
