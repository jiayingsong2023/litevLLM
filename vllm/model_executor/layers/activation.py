# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from vllm.kernels.triton.activation import silu as triton_silu, silu_and_mul as triton_silu_and_mul, gelu as triton_gelu

class Silu(nn.Module):
    def forward(self, x):
        # Restore Triton performance for common shapes
        if x.is_cuda and x.numel() > 0:
            try:
                return triton_silu(x)
            except Exception:
                return torch.nn.functional.silu(x)
        return torch.nn.functional.silu(x)

class SiluAndMul(nn.Module):
    def forward(self, x):
        if x.is_cuda and x.numel() > 0:
            try:
                return triton_silu_and_mul(x)
            except Exception:
                d = x.shape[-1] // 2
                return torch.nn.functional.silu(x[..., :d]) * x[..., d:]
        d = x.shape[-1] // 2
        return torch.nn.functional.silu(x[..., :d]) * x[..., d:]

class Gelu(nn.Module):
    def forward(self, x):
        if x.is_cuda and x.numel() > 0:
            try:
                return triton_gelu(x)
            except Exception:
                return torch.nn.functional.gelu(x)
        return torch.nn.functional.gelu(x)

__all__ = ["Silu", "SiluAndMul", "Gelu"]
