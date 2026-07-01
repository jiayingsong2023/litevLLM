# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _indexer_select_scores_kernel(
    query_ptr,
    rows_ptr,
    weights_ptr,
    scores_ptr,
    HEADS: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Memory layout:
    # - query_ptr is [HEADS, ROW_WIDTH] fp32.
    # - rows_ptr is [N_ROWS, ROW_WIDTH] fp32.
    # - weights_ptr is [HEADS] fp32.
    # - scores_ptr is [N_ROWS] fp32.
    # Tiling:
    # - one program per row.
    # - each program loads one compressed row once, then loops over heads and
    #   reduces ROW_WIDTH per head.
    # Constraint:
    # - BLOCK_WIDTH is fixed at 128 (the DS4 indexer_head_dim). row_width must
    #   be <= BLOCK_WIDTH; the host wrapper raises ValueError otherwise.
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_WIDTH)
    mask = offsets < ROW_WIDTH

    # Load the compressed row once outside the head loop to avoid repeated
    # global memory traffic.
    r = tl.load(
        rows_ptr + row * ROW_WIDTH + offsets,
        mask=mask,
        other=0.0,
    )

    acc = tl.full((), 0.0, tl.float32)
    for h in tl.range(0, HEADS):
        q = tl.load(
            query_ptr + h * ROW_WIDTH + offsets,
            mask=mask,
            other=0.0,
        )
        dot = tl.sum(q * r, axis=0)
        dot = tl.maximum(dot, 0.0)
        weight = tl.load(weights_ptr + h)
        acc = acc + weight * dot
    tl.store(scores_ptr + row, acc * SCALE)


def deepseek_v4_indexer_select_scores(
    query: torch.Tensor,
    indexer_rows: torch.Tensor,
    index_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted head scores for indexer-based compressed row selection.

    Memory layout:
    - query is [HEADS, ROW_WIDTH] fp32.
    - indexer_rows is [N_ROWS, ROW_WIDTH] fp32.
    - index_weights is [HEADS] fp32.

    Tiling:
    - Launch one program per row; loop over heads inside each program.

    Constraints:
    - BLOCK_WIDTH is fixed at 128. row_width must be <= 128; otherwise a
      ValueError is raised.
    - Inputs are cast to torch.float32 if necessary.
    - The per-head dot product is clamped to >= 0 inside the kernel (ReLU),
      matching the original PyTorch reference behavior.
    """
    if not query.is_cuda or not indexer_rows.is_cuda or not index_weights.is_cuda:
        raise ValueError("indexer select inputs must be CUDA tensors")
    if query.ndim != 2:
        raise ValueError(f"query must be 2-D; got {query.ndim}-D")
    if indexer_rows.ndim != 2:
        raise ValueError(f"indexer_rows must be 2-D; got {indexer_rows.ndim}-D")
    if index_weights.ndim != 1:
        raise ValueError(f"index_weights must be 1-D; got {index_weights.ndim}-D")
    if query.shape[1] != indexer_rows.shape[1]:
        raise ValueError("query row width must match indexer row width")
    if query.shape[0] != index_weights.shape[0]:
        raise ValueError("query heads must match index_weights length")

    n_rows = indexer_rows.shape[0]
    row_width = indexer_rows.shape[1]
    block_width = 128  # DS4 indexer_head_dim is 128
    if row_width > block_width:
        raise ValueError(f"row_width ({row_width}) exceeds BLOCK_WIDTH ({block_width})")

    query_f32 = query.to(torch.float32).contiguous()
    rows_f32 = indexer_rows.to(torch.float32).contiguous()
    weights_f32 = index_weights.to(torch.float32).contiguous()
    scale = 1.0 / float(query.shape[0] * row_width) ** 0.5

    scores = torch.empty((n_rows,), dtype=torch.float32, device=query.device)
    _indexer_select_scores_kernel[(n_rows,)](
        query_f32,
        rows_f32,
        weights_f32,
        scores,
        HEADS=query.shape[0],
        ROW_WIDTH=row_width,
        BLOCK_WIDTH=block_width,
        SCALE=scale,
    )
    return scores
