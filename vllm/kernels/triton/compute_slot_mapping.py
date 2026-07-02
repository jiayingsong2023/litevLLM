# SPDX-License-Identifier: Apache-2.0
"""
Compute slot mapping from positions and block tables.

Memory layout:
  query_start_loc: (num_reqs + 1,) int32, prefix sum of token counts.
  positions:       (num_tokens,) int64, token positions within each request.
  block_table:     (max_reqs, max_blocks_per_req) int32, block IDs per request.
  slot_mapping:    (num_tokens,) int64, output linear slot index.

Tiling:
  Grid: (num_tokens,)
  Each program handles one token.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compute_slot_mapping_kernel(
    query_start_loc_ptr,
    positions_ptr,
    block_table_ptr,
    slot_mapping_ptr,
    num_tokens,
    num_reqs,
    block_size,
    max_num_reqs,
    stride_bt_req,
    stride_bt_block,
    PAD_ID: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        tl.store(slot_mapping_ptr + token_idx, PAD_ID)
        return

    # Find request index by scanning query_start_loc. num_reqs equals the
    # actual number of requests (query_start_loc.shape[0] - 1), and is small
    # (<= max active sequences), so a linear scan is fine.
    req_idx = 0
    found = False
    for r in range(num_reqs):
        start = tl.load(query_start_loc_ptr + r).to(tl.int64)
        end = tl.load(query_start_loc_ptr + r + 1).to(tl.int64)
        if token_idx >= start and token_idx < end:
            req_idx = r
            found = True

    if not found:
        tl.store(slot_mapping_ptr + token_idx, PAD_ID)
        return

    pos = tl.load(positions_ptr + token_idx).to(tl.int64)
    block_idx = pos // block_size
    block_id = tl.load(
        block_table_ptr + req_idx * stride_bt_req + block_idx * stride_bt_block
    ).to(tl.int64)
    # Block ID 0 is a reserved zeroed/null block; map it to the pad value.
    if block_id == 0:
        slot_id = tl.cast(PAD_ID, tl.int64)
    else:
        slot_id = block_id * block_size + (pos % block_size)
    tl.store(slot_mapping_ptr + token_idx, slot_id)


def compute_slot_mapping(
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    pad_id: int = -1,
) -> None:
    """Fill ``slot_mapping`` in-place."""
    num_tokens = positions.shape[0]
    if num_tokens == 0:
        return
    if slot_mapping.shape[0] < positions.shape[0]:
        raise ValueError(
            f"slot_mapping ({slot_mapping.shape[0]}) must have length >= "
            f"positions ({positions.shape[0]})"
        )
    num_reqs = query_start_loc.shape[0] - 1

    if positions.is_cpu:
        # Pure-PyTorch fallback for CPU testing. Triton kernels cannot access
        # host memory, so replicate the kernel logic with tensor ops.
        token_indices = torch.arange(slot_mapping.shape[0], dtype=torch.int64)
        starts = query_start_loc[:-1].to(torch.int64)
        ends = query_start_loc[1:].to(torch.int64)
        req_indices = torch.searchsorted(ends, token_indices, right=True).clamp_(
            0, num_reqs - 1
        )
        valid = (
            (token_indices >= starts[req_indices])
            & (token_indices < ends[req_indices])
            & (token_indices < num_tokens)
        )

        pos_indices = torch.where(valid, token_indices, torch.zeros_like(token_indices))
        pos = positions[pos_indices]
        block_idx = (pos // block_size).clamp_(0, block_table.shape[1] - 1)
        block_ids = block_table[req_indices, block_idx].to(torch.int64)
        slots = block_ids * block_size + (pos % block_size)
        pad_tensor = torch.full_like(slots, pad_id, dtype=torch.int64)
        slots = torch.where(block_ids != 0, slots, pad_tensor)
        slots = torch.where(valid, slots, pad_tensor)
        slot_mapping.copy_(slots)
        return

    grid = (slot_mapping.shape[0],)
    _compute_slot_mapping_kernel[grid](
        query_start_loc,
        positions,
        block_table,
        slot_mapping,
        num_tokens,
        num_reqs,
        block_size,
        block_table.shape[0],
        block_table.stride(0),
        block_table.stride(1),
        PAD_ID=pad_id,
    )
