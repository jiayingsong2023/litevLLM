# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Optional

def fused_moe(hidden_states, w1, w2, gating_output, topk, renormalize=True):
    """
    Dual-Mode MoE Dispatcher for FastInference.
    - Performance Mode: Uses sorted grouping for high-throughput on medium models.
    - Stability Mode: Uses sequential dispatch for large models to avoid HIP errors.
    """
    M, K = hidden_states.shape
    E = w1.shape[0]
    
    # 1. Routing
    routing_weights = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
    output = torch.zeros_like(hidden_states)
    
    flattened_ids = topk_ids.view(-1)
    flattened_weights = topk_weights.view(-1)
    token_indices = torch.arange(M, device=hidden_states.device).repeat_interleave(topk)

    # --- DUAL MODE LOGIC ---
    # If model is extremely large (E >= 256 experts or hidden_size is large), use Stability Mode.
    # Otherwise, use Performance Mode (Sorted Grouping).
    use_stability_mode = (E >= 256)
    
    if not use_stability_mode:
        # PERFORMANCE MODE: Sorted Contiguous Dispatch
        # Cluster tokens by expert ID to maximize GPU compute density
        sorted_ids, sorting_indices = torch.sort(flattened_ids)
        sorted_token_indices = token_indices[sorting_indices]
        sorted_weights = flattened_weights[sorting_indices]
        
        unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        unique_ids_list = unique_ids.tolist()
        counts_list = counts.tolist()
        
        curr_offset = 0
        for i, expert_idx in enumerate(unique_ids_list):
            count = counts_list[i]
            start, end = curr_offset, curr_offset + count
            curr_offset += count
            
            group_token_indices = sorted_token_indices[start:end]
            group_weights = sorted_weights[start:end].unsqueeze(-1)
            
            tokens = hidden_states.index_select(0, group_token_indices)
            res = torch.nn.functional.linear(tokens, w1[expert_idx])
            res = torch.nn.functional.silu(res)
            res = torch.nn.functional.linear(res, w2[expert_idx])
            
            output.index_add_(0, group_token_indices, res * group_weights)
    else:
        # STABILITY MODE: Simple Sequential Dispatch
        # Avoids complex sorting kernels that trigger HIP errors on massive models
        active_experts = flattened_ids.unique().tolist()
        for expert_idx in active_experts:
            mask = (flattened_ids == expert_idx)
            expert_token_indices = token_indices[mask]
            expert_weights = flattened_weights[mask].unsqueeze(-1)
            
            tokens = hidden_states.index_select(0, expert_token_indices)
            res = torch.nn.functional.linear(tokens, w1[expert_idx])
            res = torch.nn.functional.silu(res)
            res = torch.nn.functional.linear(res, w2[expert_idx])
            output.index_add_(0, expert_token_indices, res * expert_weights)
                
    return output

__all__ = ["fused_moe"]
