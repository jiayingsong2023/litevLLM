# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

def moe_permute(
    hidden_states: torch.Tensor,
    a1q_scale: torch.Tensor | None,
    topk_ids: torch.Tensor,
    n_expert: int,
    n_local_expert: int = -1,
    expert_map: torch.Tensor | None = None,
    align_block_size: int | None = None,
    fill_invalid_expert: int = -1,
    permuted_hidden_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_token, n_hidden = hidden_states.size()
    topk = topk_ids.size(1)
    assert (n_hidden * hidden_states.element_size()) % 16 == 0, (
        "permue kernel need hidden dim align to 16B"
    )
    permuted_row_size = n_token * topk
    if align_block_size is not None:
        permuted_row_size = (
            (
                permuted_row_size
                + n_expert * (align_block_size - 1)
                + align_block_size
                - 1
            )
            // align_block_size
            * align_block_size
        )
    if n_local_expert == -1:
        n_local_expert = n_expert
    if permuted_hidden_states is None:
        permuted_hidden_states = torch.empty(
            (permuted_row_size, n_hidden),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    assert permuted_hidden_states.size() == (permuted_row_size, n_hidden), (
        f"Expected permuted hidden states to be {(permuted_row_size, n_hidden)}"
        f" but got {permuted_hidden_states.size()}"
    )

    token_expert_indices = torch.arange(
        0, n_token * topk, dtype=torch.int32, device=hidden_states.device
    ).reshape((n_token, topk))

    m_indices = torch.full(
        (permuted_row_size,),
        fill_invalid_expert,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    expert_first_token_offset = torch.empty(
        n_local_expert + 1, dtype=torch.int64, device=hidden_states.device
    )
    permuted_idx = torch.full(
        (permuted_row_size,),
        n_token * topk,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    inv_permuted_idx = torch.empty(
        (n_token, topk), dtype=torch.int32, device=hidden_states.device
    )
    topk_ids = topk_ids.to(torch.int32)
    torch.ops._moe_C.moe_permute(
        hidden_states,
        topk_ids,
        token_expert_indices,
        expert_map,
        n_expert,
        n_local_expert,
        topk,
        align_block_size,
        permuted_hidden_states,
        expert_first_token_offset,
        inv_permuted_idx,
        permuted_idx,
        m_indices,
    )

    if a1q_scale is not None and a1q_scale.dim() > 1:
        a1q_scale = a1q_scale[permuted_idx.clamp(max=n_token * topk - 1) // topk]
    return (
        permuted_hidden_states,
        a1q_scale,
        expert_first_token_offset,
        inv_permuted_idx.flatten(),
        m_indices,
    )

def moe_unpermute(
    out: torch.Tensor,
    permuted_hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    expert_first_token_offset: torch.Tensor | None = None,
) -> None:
    topk = topk_weights.size(1)
    n_hidden = permuted_hidden_states.size(-1)
    assert (n_hidden * permuted_hidden_states.element_size()) % 16 == 0, (
        "unpermue kernel need hidden dim align to 16B"
    )

    torch.ops._moe_C.moe_unpermute(
        permuted_hidden_states,
        topk_weights,
        inv_permuted_idx,
        expert_first_token_offset,
        topk,
        out,
    )

def moe_permute_unpermute_supported():
    return torch.ops._moe_C.moe_permute_unpermute_supported()
