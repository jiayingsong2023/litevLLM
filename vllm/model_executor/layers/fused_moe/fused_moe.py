# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Optional

def fused_moe(hidden_states, w1, w2, gating_output, topk, renormalize=True):
    """
    Stabilized MoE Dispatcher for AMD APU.
    Handles both standard (w1, w2) and SwiGLU (gate_up, down) experts.
    """
    M, K = hidden_states.shape
    
    # 1. Routing
    routing_weights = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
    output = torch.zeros_like(hidden_states)
    
    # 2. Sequential-Batch Dispatch (AMD Stabilized)
    flattened_ids = topk_ids.view(-1)
    flattened_weights = topk_weights.view(-1)
    token_indices = torch.arange(M, device=hidden_states.device).repeat_interleave(topk)
    
    # Move control flow to CPU
    ids_cpu = flattened_ids.detach().cpu()
    active_experts = torch.unique(ids_cpu, sorted=True).tolist()
    
    for expert_idx in active_experts:
        mask_cpu = (ids_cpu == expert_idx)
        expert_token_indices = token_indices[mask_cpu.to(hidden_states.device)]
        expert_weights = flattened_weights[mask_cpu.to(hidden_states.device)].unsqueeze(-1)
        
        tokens = hidden_states.index_select(0, expert_token_indices)
        
        # Expert Forward
        # w1 might be [N, K] or [2*N, K] (SwiGLU)
        res = torch.nn.functional.linear(tokens, w1[expert_idx])
        
        # SwiGLU logic if w1 is 2x intermediate
        if res.shape[-1] == 2 * w2.shape[-1]:
            d = res.shape[-1] // 2
            res = torch.nn.functional.silu(res[:, :d]) * res[:, d:]
        else:
            res = torch.nn.functional.silu(res)
            
        res = torch.nn.functional.linear(res, w2[expert_idx])
        
        # Accumulated Update
        output.index_add_(0, expert_token_indices, res * expert_weights)
                
    return output

__all__ = ["fused_moe"]
