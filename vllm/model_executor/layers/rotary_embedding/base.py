# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Union, List
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

class RotaryEmbedding(nn.Module, ABC):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        
        self.use_flashinfer = False
        cache = self._compute_cos_sin_cache()
        self.register_buffer("cos_sin_cache", cache.to(dtype), persistent=False)
        self.apply_rotary_emb = ApplyRotaryEmb(is_neox_style=self.is_neox_style)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return self.apply_rotary_emb.forward_native(query, cos, sin), \
               self.apply_rotary_emb.forward_native(key, cos, sin)

def get_rope(head_size, rotary_dim, max_position, base, is_neox_style, dtype=torch.float16):
    class DefaultRoPE(RotaryEmbedding):
        pass
    return DefaultRoPE(head_size, rotary_dim, max_position, base, is_neox_style, dtype)
