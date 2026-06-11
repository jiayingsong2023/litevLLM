# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
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
    attn_sinks: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.hidden.ndim != 1:
            raise ValueError(f"hidden must be 1-D; got {self.hidden.ndim}-D")
        if self.kv_rows.ndim != 2:
            raise ValueError(f"kv_rows must be 2-D; got {self.kv_rows.ndim}-D")
        if self.kv_rows.shape[0] == 0:
            raise ValueError("kv_rows must contain at least one row")
        if self.kv_rows.shape[1] != self.hidden.numel():
            raise ValueError(
                "kv_rows width must match hidden size; "
                f"got {self.kv_rows.shape[1]} and {self.hidden.numel()}"
            )
        if self.token_idx < 0:
            raise ValueError("token_idx must be non-negative")
        if self.attn_sinks is not None and self.attn_sinks.ndim != 1:
            raise ValueError(
                f"attn_sinks must be 1-D; got {self.attn_sinks.ndim}-D"
            )


def deepseek_v4_attention(inputs: DeepSeekV4AttentionKernelInputs) -> torch.Tensor:
    tensors = [inputs.hidden, inputs.kv_rows, inputs.attn_sinks]
    if any(tensor is not None and not tensor.is_cuda for tensor in tensors):
        raise ValueError("DeepSeek V4 attention inputs must be CUDA tensors")
    scores = inputs.kv_rows.to(torch.float32).matmul(inputs.hidden.to(torch.float32))
    scores = scores / math.sqrt(float(inputs.hidden.numel()))
    if inputs.attn_sinks is not None:
        logits = torch.cat([scores, inputs.attn_sinks.to(torch.float32)])
        probs = torch.softmax(logits, dim=0)[: scores.numel()]
    else:
        probs = torch.softmax(scores, dim=0)
    return probs.matmul(inputs.kv_rows.to(torch.float32))
