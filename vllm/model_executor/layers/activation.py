# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from vllm.kernels.triton.activation import silu as triton_silu, silu_and_mul as triton_silu_and_mul, gelu as triton_gelu

class Silu(nn.Module):
    def forward(self, x):
        return triton_silu(x)

class SiluAndMul(nn.Module):
    def forward(self, x):
        return triton_silu_and_mul(x)

class Gelu(nn.Module):
    def forward(self, x):
        return triton_gelu(x)

__all__ = ["Silu", "SiluAndMul", "Gelu"]
