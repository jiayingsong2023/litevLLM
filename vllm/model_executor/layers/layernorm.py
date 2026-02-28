# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from vllm.kernels.triton.rms_norm import rms_norm, fused_add_rms_norm

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x, residual=None):
        if residual is not None:
            # 使用 Triton 融合的 Add + RMSNorm
            fused_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            return x, residual
        
        # 使用 Triton 融合的 RMSNorm
        out = torch.empty_like(x)
        rms_norm(out, x, self.weight, self.variance_epsilon)
        return out
