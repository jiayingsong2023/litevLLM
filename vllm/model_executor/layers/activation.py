# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F

class Silu(nn.Module):
    def forward(self, x):
        return F.silu(x)

class SiluAndMul(nn.Module):
    def forward(self, x):
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

__all__ = ["Silu", "SiluAndMul"]