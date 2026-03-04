# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Optional

def fused_moe(hidden_states, w1, w2, gating_output, topk, renormalize=True):
    """
    Stabilized MoE Dispatcher for AMD APU.
    Ensures zero illegal memory access by using sequential-batch dispatch.
    """
    M, K = hidden_states.shape
    
    # 1. Routing
    routing_weights = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
    # 2. Sequential-Batch Dispatch (Most stable on AMD)
    output = torch.zeros_like(hidden_states)
    
    flattened_ids = topk_ids.view(-1)
    flattened_weights = topk_weights.view(-1)
    token_indices = torch.arange(M, device=hidden_states.device).repeat_interleave(topk)
    
    # Using unique() on GPU is safe if results are moved to CPU for loop control
    active_experts = flattened_ids.unique().tolist()
    
    for expert_idx in active_experts:
        # Masking tokens for this expert
        mask = (flattened_ids == expert_idx)
        expert_token_indices = token_indices[mask]
        expert_weights = flattened_weights[mask].unsqueeze(-1)
        
        # Pull tokens
        tokens = hidden_states.index_select(0, expert_token_indices)
        
        # Expert Forward
        res = torch.nn.functional.linear(tokens, w1[expert_idx])
        res = torch.nn.functional.silu(res)
        res = torch.nn.functional.linear(res, w2[expert_idx])
        
        # Accumulated Update
        output.index_add_(0, expert_token_indices, res * expert_weights)
                
    return output

__all__ = ["fused_moe"]
