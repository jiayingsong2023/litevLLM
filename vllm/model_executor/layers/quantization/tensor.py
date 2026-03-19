# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

class LRUWeightCache:
    def __init__(self, max_size=256):
        self.cache: Dict[int, torch.Tensor] = {}
        self.keys = []
        self.max_size = max_size
    def get(self, key: int) -> Optional[torch.Tensor]:
        if key in self.cache:
            self.keys.remove(key); self.keys.append(key)
            return self.cache[key]
        return None
    def put(self, key: int, value: torch.Tensor):
        if key in self.cache: return
        if len(self.keys) >= self.max_size:
            old_key = self.keys.pop(0); del self.cache[old_key]
        self.cache[key] = value; self.keys.append(key)

_GLOBAL_WEIGHT_CACHE = LRUWeightCache(max_size=512)

def dequantize_q4k_pytorch(qweight: torch.Tensor, n_rows: int, n_cols: int) -> torch.Tensor:
    try:
        bytes_per_row = qweight.numel() // n_rows; n_blocks = bytes_per_row // 144
        qw_view = qweight.view(n_rows, n_blocks, 144)
        d = qw_view[:, :, 0:2].contiguous().view(torch.float16).to(torch.float32)
        dmin = qw_view[:, :, 2:4].contiguous().view(torch.float16).to(torch.float32)
        qs = qw_view[:, :, 16:144]
        qs_low = (qs & 0x0F).to(torch.float32); qs_high = ((qs >> 4) & 0x0F).to(torch.float32)
        w_low = qs_low * d - dmin; w_high = qs_high * d - dmin
        res = torch.stack([w_low, w_high], dim=-1).view(n_rows, n_blocks * 256)
        if res.shape[1] > n_cols: res = res[:, :n_cols]
        elif res.shape[1] < n_cols: res = torch.cat([res, torch.zeros((n_rows, n_cols - res.shape[1]), device=res.device, dtype=res.dtype)], dim=1)
        return res.to(torch.float16)
    except Exception as e: raise RuntimeError(f"Q4_K Dequant Error: {e} ({qweight.shape}, R:{n_rows}, C:{n_cols})")

def dequantize_q6k_pytorch(qweight: torch.Tensor, n_rows: int, n_cols: int) -> torch.Tensor:
    try:
        n_blocks = qweight.numel() // (n_rows * 210); qw_view = qweight.view(n_rows, n_blocks, 210)
        ql = qw_view[:, :, 0:128]; qh = qw_view[:, :, 128:192]; d = qw_view[:, :, 208:210].contiguous().view(torch.float16).to(torch.float32)
        ql_low = (ql & 0x0F).to(torch.float32); ql_high = ((ql >> 4) & 0x0F).to(torch.float32)
        res_low = (ql_low - 32.0) * d; res_high = (ql_high - 32.0) * d
        res = torch.stack([res_low, res_high], dim=-1).view(n_rows, n_blocks * 256)
        if res.shape[1] > n_cols: res = res[:, :n_cols]
        return res.to(torch.float16)
    except Exception as e: raise RuntimeError(f"Q6_K Dequant Error: {e}")

def dequantize_awq_pytorch(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    try:
        n_rows, n_cols_packed = qweight.shape; n_cols = n_cols_packed * 8
        shifts = torch.arange(0, 32, 4, device=qweight.device)
        qs = ((qweight.unsqueeze(-1) >> shifts) & 0x0F).view(n_rows, n_cols).to(torch.float32)
        zs = ((qzeros.unsqueeze(-1) >> shifts) & 0x0F).view(qzeros.shape[0], -1).to(torch.float32)
        n_groups = n_cols // group_size; qs = qs.view(n_rows, n_groups, group_size)
        res = (qs - zs.unsqueeze(-1)) * scales.to(torch.float32).unsqueeze(-1)
        return res.view(n_rows, n_cols).to(torch.float16)
    except Exception as e: raise RuntimeError(f"AWQ PyTorch Dequant Error: {e}")

class QuantizedLinearWeight(nn.Module, ABC):
    def __init__(self): super().__init__(); self.weight_id = id(self)
    @abstractmethod
    def matmul(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor: pass

class GGUFWeight(QuantizedLinearWeight):
    def __init__(self, qweight, scales, quant_type=2, prefer_fused=True, original_shape=None, slice_offset=0):
        super().__init__(); self.qweight = nn.Parameter(qweight, requires_grad=False); self.scales = nn.Parameter(scales, requires_grad=False); self.quant_type = quant_type; self.prefer_fused = prefer_fused; self.original_shape = original_shape; self.slice_offset = slice_offset
    def matmul(self, x, bias=None):
        n_rows = self.qweight.shape[0]; n_cols = (self.qweight.shape[1] // 144 * 256) if self.quant_type >= 12 else (self.qweight.shape[1] // 18 * 32)
        bs = x.shape[0] if x.dim() > 1 else 1
        cached_w = _GLOBAL_WEIGHT_CACHE.get(self.weight_id)
        if cached_w is None:
            if self.quant_type == 2: from vllm.kernels.triton.gguf_q4_0_dequant import gguf_q4_0_dequant; cached_w = gguf_q4_0_dequant(self.qweight, n_rows, n_cols)
            elif self.quant_type == 12: cached_w = dequantize_q4k_pytorch(self.qweight, n_rows, n_cols)
            elif self.quant_type == 14: cached_w = dequantize_q6k_pytorch(self.qweight, n_rows, n_cols)
            if self.original_shape is not None or self.slice_offset > 0:
                os0 = self.original_shape[0] if self.original_shape else n_rows; os1 = self.original_shape[1] if self.original_shape else n_cols
                cached_w = cached_w[self.slice_offset : self.slice_offset + os0, :os1].contiguous()
            _GLOBAL_WEIGHT_CACHE.put(self.weight_id, cached_w)
        return torch.nn.functional.linear(x, cached_w, bias)

class AWQWeight(QuantizedLinearWeight):
    def __init__(self, qweight, scales, qzeros, group_size=128):
        super().__init__(); self.qweight = nn.Parameter(qweight, requires_grad=False); self.scales = nn.Parameter(scales, requires_grad=False); self.qzeros = nn.Parameter(qzeros, requires_grad=False); self.group_size = group_size
    def matmul(self, x, bias=None):
        cached_w = _GLOBAL_WEIGHT_CACHE.get(self.weight_id)
        if cached_w is None:
            try:
                from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton
                cached_w = awq_dequantize_triton(self.qweight, self.scales, self.qzeros, self.group_size)
            except: cached_w = dequantize_awq_pytorch(self.qweight, self.scales, self.qzeros, self.group_size)
            _GLOBAL_WEIGHT_CACHE.put(self.weight_id, cached_w)
        return torch.nn.functional.linear(x, cached_w, bias)
