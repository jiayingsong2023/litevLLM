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
    """
    Unified interface for quantized weights that can perform matmul.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def matmul(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

class GGUFWeight(QuantizedLinearWeight):
    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, quant_type: Optional[int] = None):
        super().__init__()
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.scales = nn.Parameter(scales, requires_grad=False)
        self.quant_type = quant_type
        self.weight_id = id(self)

    def matmul(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        from vllm.kernels.triton.gguf_dequant import gguf_dequantize
        from vllm.model_executor.layers.quantization.gguf_kernels import ggml_mul_mat_a8_fallback

        cached_weight = _GLOBAL_WEIGHT_CACHE.get(self.weight_id)
        if cached_weight is None:
            # print(f"[CACHE] MISS GGUF {self.weight_id}")
            if self.quant_type is not None:
                # Path for specialized fused kernels (e.g. Q4_K fallback to a8)
                original_shape = x.shape
                x_2d = x.view(-1, original_shape[-1])
                out_2d = ggml_mul_mat_a8_fallback(
                    self.qweight,
                    x_2d,
                    int(self.quant_type),
                    int(self.qweight.shape[0]),
                )
                if bias is not None:
                    out_2d = out_2d + bias.to(out_2d.dtype)
                return out_2d.view(*original_shape[:-1], out_2d.shape[-1])

            # Default path: Dequantize and Cache
            # We assume Q4_K (2) as default for now if no specialized path
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
        # Strategy: Fused Kernel for Batch throughput, Cache for Latency
        bs = x.shape[0] if x.dim() > 1 else 1
        
        # Use Fused Kernel for BS >= 16 to save memory bandwidth
        if bs >= 16:
            from vllm.kernels.triton.awq_fused_gemm import awq_fused_gemm
            # Ensure qzeros is not None for fused kernel
            qzeros = self.qzeros
            if qzeros is None:
                # Column grouping (Qwen style)
                n_rows_q = self.qweight.shape[0]
                n_cols_q = self.qweight.shape[1] * 8
                qzeros = torch.full((n_rows_q, n_cols_q // self.group_size // 8 + 1), -2004318072, device=self.qweight.device, dtype=torch.int32)
            
            # A: [M, K], B: [N, K // 8] -> Output: [M, N]
            # Standard AWQ Layout in LitevLLM: [N, K // 8]
            out = awq_fused_gemm(x.view(-1, x.shape[-1]), self.qweight, self.scales, qzeros, self.group_size)
            out = out.view(*x.shape[:-1], out.shape[-1])
            if bias is not None:
                out += bias
            return out

        from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton
        
        cached_weight = _GLOBAL_WEIGHT_CACHE.get(self.weight_id)
        if cached_weight is None:
            qweight = self.qweight.contiguous()
            scales = self.scales.contiguous()
            input_size = x.shape[-1]
            qzeros = self.qzeros.contiguous() if self.qzeros is not None else None
            
            if qzeros is None:
                n_rows_q = qweight.shape[0]
                n_cols_q = qweight.shape[1] * 8
                if scales.shape[0] == n_rows_q:
                    qzeros = torch.full((n_rows_q, n_cols_q // self.group_size // 8 + 1), -2004318072, device=qweight.device, dtype=torch.int32)
                else:
                    qzeros = torch.full((n_rows_q // self.group_size, n_cols_q // 8), -2004318072, device=qweight.device, dtype=torch.int32)
            
            # Optimization: Dequantize into float8 to save cache bandwidth
            out_dtype = torch.float8_e4m3fn if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9 or "gfx11" in torch.cuda.get_device_name().lower() else torch.float16
            dequantized = awq_dequantize_triton(qweight, scales, qzeros, self.group_size, out_dtype=out_dtype)
            
            cached_weight = dequantized.transpose(0, 1).contiguous() if dequantized.shape[0] == input_size else dequantized.contiguous()
            _GLOBAL_WEIGHT_CACHE.put(self.weight_id, cached_weight)

        if cached_weight.dtype == torch.float8_e4m3fn:
            # We cast to input dtype for the final matmul. In future, we can use fused FP8 matmul.
            cached_weight = cached_weight.to(x.dtype)
            
        return torch.nn.functional.linear(x, cached_weight, bias)
