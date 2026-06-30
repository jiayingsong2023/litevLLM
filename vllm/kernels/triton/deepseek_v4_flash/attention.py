# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


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
            raise ValueError(f"attn_sinks must be 1-D; got {self.attn_sinks.ndim}-D")


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


@triton.jit
def _deepseek_v4_fused_swa_attention_kernel(
    query_ptr,
    kv_rows_ptr,
    attn_sinks_ptr,
    output_ptr,
    stride_qh,
    stride_qd,
    stride_kvw,
    stride_kvd,
    stride_oh,
    stride_od,
    num_heads,
    head_dim,
    window,
    scale,
    HAS_SINKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused sliding-window attention over shared K=V rows.

    Memory layout:
    - query is [num_heads, head_dim] row-major fp32.
    - kv_rows is [window, head_dim] row-major; loaded as fp32.
    - attn_sinks is [num_heads] fp32; contributes a softmax logit with no value.
    - output is [num_heads, head_dim] fp32.

    Tiling:
    - grid (num_heads,): one program per head.
    - BLOCK_D threads cover the head dimension; masked when head_dim < BLOCK_D.
    - Each program streams over `window` KV rows in a loop, accumulating the
      weighted value using online softmax to keep only O(head_dim) SRAM.
    """
    # Memory layout (row-major):
    #
    #   query  : [num_heads, head_dim]
    #   kv_rows: [window, head_dim]
    #   output : [num_heads, head_dim]
    #
    # Tiling:
    #
    #   grid   : (num_heads,)
    #   block  : (BLOCK_D,)
    #   head   : one program per head
    #   threads: each program loads head_dim elements via a BLOCK_D-wide vector
    head_idx = tl.program_id(0)
    if head_idx >= num_heads:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    q_ptrs = query_ptr + head_idx * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    m = float("-inf")
    d = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for w in range(window):
        kv_ptrs = kv_rows_ptr + w * stride_kvw + offs_d * stride_kvd
        kv = tl.load(kv_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        score = tl.sum(q * kv) * scale

        new_m = tl.maximum(m, score)
        exp_old = tl.exp(m - new_m)
        exp_score = tl.exp(score - new_m)
        d = d * exp_old + exp_score
        acc = acc * exp_old + exp_score * kv
        m = new_m

    if HAS_SINKS:
        sink = tl.load(attn_sinks_ptr + head_idx).to(tl.float32)
        new_m = tl.maximum(m, sink)
        exp_old = tl.exp(m - new_m)
        d = d * exp_old + tl.exp(sink - new_m)
        acc = acc * exp_old
        m = new_m

    out = acc / d
    out_ptrs = output_ptr + head_idx * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, out, mask=mask_d)


def deepseek_v4_fused_sliding_window_attention(
    query: torch.Tensor,
    kv_rows: torch.Tensor,
    attn_sinks: torch.Tensor | None,
) -> torch.Tensor:
    """Fused sliding-window attention over shared K=V latent rows.

    Memory layout:
    - query is [num_heads, head_dim] fp32.
    - kv_rows is [window, head_dim] and is used as both keys and values.
    - attn_sinks is [num_heads] optional; each sink contributes one extra
      softmax logit but has no associated value row.
    - output is [num_heads, head_dim] fp32.

    Tiling:
    - grid (num_heads,); one block per head.
    - BLOCK_D is the next power of two >= head_dim.
    """
    if query.ndim != 2:
        raise ValueError(f"query must be 2-D; got {query.ndim}-D")
    if kv_rows.ndim != 2:
        raise ValueError(f"kv_rows must be 2-D; got {kv_rows.ndim}-D")
    if query.shape[1] != kv_rows.shape[1]:
        raise ValueError(
            "query and kv_rows widths must match; "
            f"got {query.shape[1]} and {kv_rows.shape[1]}"
        )
    if not query.is_cuda or not kv_rows.is_cuda:
        raise ValueError("fused sliding attention inputs must be CUDA tensors")
    if attn_sinks is not None:
        if attn_sinks.shape != (query.shape[0],):
            raise ValueError(
                f"attn_sinks shape must be {(query.shape[0],)}; "
                f"got {tuple(attn_sinks.shape)}"
            )
        if not attn_sinks.is_cuda:
            raise ValueError("attn_sinks must be a CUDA tensor")

    num_heads, head_dim = query.shape
    window = kv_rows.shape[0]
    output = torch.empty_like(query)
    scale = 1.0 / math.sqrt(float(head_dim))
    has_sinks = attn_sinks is not None
    sink_ptr = (
        attn_sinks
        if attn_sinks is not None
        else torch.empty(0, dtype=query.dtype, device=query.device)
    )
    block_d = triton.next_power_of_2(head_dim)

    _deepseek_v4_fused_swa_attention_kernel[(num_heads,)](
        query,
        kv_rows,
        sink_ptr,
        output,
        query.stride(0),
        query.stride(1),
        kv_rows.stride(0),
        kv_rows.stride(1),
        output.stride(0),
        output.stride(1),
        num_heads,
        head_dim,
        window,
        scale,
        HAS_SINKS=has_sinks,
        BLOCK_D=block_d,
    )
    return output
