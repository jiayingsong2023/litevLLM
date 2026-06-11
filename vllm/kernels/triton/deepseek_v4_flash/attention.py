# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeepSeekV4AttentionKernelInputs:
    """Kernel input contract for DeepSeek V4 sliding attention.

    Memory layout:
    - hidden is [hidden_size] for batch=1 decode.
    - kv_rows is [num_rows, latent_kv_dim] and is addressed by logical row.
    - output is [hidden_size].

    Tiling:
    - Future kernels should tile by attention head and selected KV rows.
    - RoPE tail handling must stay inside the attention kernel boundary so the
      model layer can dispatch without owning per-head vector math.
    """

    hidden: torch.Tensor
    kv_rows: torch.Tensor
    token_idx: int

    def __post_init__(self) -> None:
        if self.hidden.ndim != 1:
            raise ValueError(f"hidden must be 1-D; got {self.hidden.ndim}-D")
        if self.kv_rows.ndim != 2:
            raise ValueError(f"kv_rows must be 2-D; got {self.kv_rows.ndim}-D")
        if self.token_idx < 0:
            raise ValueError("token_idx must be non-negative")


def deepseek_v4_attention(inputs: DeepSeekV4AttentionKernelInputs) -> torch.Tensor:
    del inputs
    raise NotImplementedError("DeepSeek V4 Flash attention kernel is not implemented")
