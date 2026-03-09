# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x, residual=None):
        if residual is not None: x = x + residual
        
        # AUTO-DEVICE ALIGNMENT
        if self.weight.device != x.device:
            self.weight.data = self.weight.to(x.device)
            
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x_fp32.to(input_dtype)
