# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

import torch

from vllm.triton_utils import tl, triton


def _attn_section(
    section: Callable[[str], AbstractContextManager[None]] | None, name: str
) -> AbstractContextManager[None]:
    if section is None:
        return nullcontext()
    return section(name)


@triton.jit
def _compressed_attention_scores_kernel(
    query_ptr,
    rows_ptr,
    selected_ptr,
    scores_ptr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    # Memory layout:
    # - query_ptr is a contiguous [HEAD_DIM] fp32 vector.
    # - rows_ptr is a contiguous [N_ROWS, HEAD_DIM] fp32 matrix.
    # - selected_ptr is a contiguous [N_SELECTED] int32/int64 vector of row ids.
    # - scores_ptr is a contiguous [N_SELECTED] fp32 output.
    # Tiling:
    # - one program per selected row.
    # - each program loads the selected row and the full query, then reduces.
    row_idx = tl.program_id(0)
    selected = tl.load(selected_ptr + row_idx).to(tl.int64)

    offsets = tl.arange(0, BLOCK_DIM)
    mask = offsets < HEAD_DIM

    q = tl.load(query_ptr + offsets, mask=mask, other=0.0)
    k = tl.load(
        rows_ptr + selected * HEAD_DIM + offsets,
        mask=mask,
        other=0.0,
    )
    score = tl.sum(q * k, axis=0)
    scale = 1.0 / tl.sqrt(tl.full((), float(HEAD_DIM), tl.float32))
    tl.store(scores_ptr + row_idx, score * scale)


@triton.jit
def _compressed_attention_reduce_kernel(
    rows_ptr,
    selected_ptr,
    probs_ptr,
    out_ptr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    N_SELECTED: tl.constexpr,
):
    # Memory layout:
    # - rows_ptr is [N_ROWS, HEAD_DIM] fp32.
    # - selected_ptr is [N_SELECTED] int32/int64 row ids.
    # - probs_ptr is [N_SELECTED] fp32 softmax probabilities.
    # - out_ptr is [HEAD_DIM] fp32 output.
    # Tiling:
    # - one program per BLOCK_DIM chunk of HEAD_DIM.
    # - the program loops over selected rows and accumulates prob * row.
    dim_start = tl.program_id(0) * BLOCK_DIM
    offsets = dim_start + tl.arange(0, BLOCK_DIM)
    mask = offsets < HEAD_DIM

    acc = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
    for i in tl.range(0, N_SELECTED):
        selected = tl.load(selected_ptr + i).to(tl.int64)
        row = tl.load(
            rows_ptr + selected * HEAD_DIM + offsets,
            mask=mask,
            other=0.0,
        )
        prob = tl.load(probs_ptr + i)
        acc = acc + prob * row
    tl.store(out_ptr + offsets, acc, mask=mask)


def deepseek_v4_compressed_attention_triton(
    query: torch.Tensor,
    compressed_rows: torch.Tensor,
    selected_rows: torch.Tensor,
    *,
    section: Callable[[str], AbstractContextManager[None]] | None = None,
) -> torch.Tensor:
    """Fused compressed attention: softmax(selected_rows @ query) @ selected_rows.

    Memory layout:
    - query is [HEAD_DIM] fp32/fp16/bf16.
    - compressed_rows is [N_ROWS, HEAD_DIM] fp32/fp16/bf16.
    - selected_rows is [N_SELECTED] int32/int64 row ids.

    Tiling:
    - Launch 1 program per selected row to compute scores.
    - Launch ceil(HEAD_DIM / BLOCK_DIM) programs to reduce weighted rows.

    Constraints:
    - BLOCK_DIM is fixed at 512, matching the DS4 Flash HEAD_DIM. The kernel
      masks unused lanes for smaller head dimensions.
    """
    if not query.is_cuda or not compressed_rows.is_cuda or not selected_rows.is_cuda:
        raise ValueError("DeepSeek V4 compressed attention inputs must be CUDA tensors")
    if query.ndim != 1:
        raise ValueError(f"query must be 1-D; got {query.ndim}-D")
    if compressed_rows.ndim != 2:
        raise ValueError(f"compressed_rows must be 2-D; got {compressed_rows.ndim}-D")
    if compressed_rows.shape[1] != query.numel():
        raise ValueError("compressed_rows width must match query size")
    if selected_rows.ndim != 1:
        raise ValueError(f"selected_rows must be 1-D; got {selected_rows.ndim}-D")
    if selected_rows.numel() == 0:
        raise ValueError("selected_rows must contain at least one row")

    query_f32 = query.to(torch.float32).contiguous()
    rows_f32 = compressed_rows.to(torch.float32).contiguous()
    selected = selected_rows.contiguous()

    head_dim = query.numel()
    n_selected = selected.numel()
    block_dim = 512  # HEAD_DIM is 512 in DS4 Flash

    scores = torch.empty((n_selected,), dtype=torch.float32, device=query.device)
    with _attn_section(section, "attn_backend_scores"):
        _compressed_attention_scores_kernel[(n_selected,)](
            query_f32,
            rows_f32,
            selected,
            scores,
            HEAD_DIM=head_dim,
            BLOCK_DIM=block_dim,
        )

    with _attn_section(section, "attn_backend_softmax_reduce"):
        probs = torch.softmax(scores, dim=0)

        output = torch.empty((head_dim,), dtype=torch.float32, device=query.device)
        grid = ((head_dim + block_dim - 1) // block_dim,)
        _compressed_attention_reduce_kernel[grid](
            rows_f32,
            selected,
            probs,
            output,
            HEAD_DIM=head_dim,
            BLOCK_DIM=block_dim,
            N_SELECTED=n_selected,
        )
    return output
