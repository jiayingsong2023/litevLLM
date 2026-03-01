# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
            
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        out = self.weight * x.to(input_dtype)
        
        if residual is not None:
            return out, x.to(input_dtype) # 简化 residual 返回
        return out
