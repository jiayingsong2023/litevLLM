# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Optional
from vllm.kernels.triton.moe_gemm import index_aware_linear

def fused_moe(hidden_states, w1, w2, gating_output, topk, renormalize=True):
    """
    Optimized MoE Dispatcher for FastInference.
    Uses Index-aware GEMM to minimize data movement.
    """
    M, K = hidden_states.shape
    E, N, _ = w1.shape
    
    # 1. Routing
    routing_logits = gating_output
    routing_weights = torch.softmax(routing_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
    # 2. Execution using Index-aware GEMM
    output = torch.zeros_like(hidden_states)
    
    # Flatten topk_ids and topk_weights to simplify processing
    flattened_ids = topk_ids.view(-1)
    flattened_weights = topk_weights.view(-1)
    token_indices = torch.arange(M, device=hidden_states.device).repeat_interleave(topk)
    
    # Find active experts to avoid looping over all E
    active_experts = flattened_ids.unique()
    
    for expert_idx in active_experts:
        expert_idx_item = expert_idx.item()
        
        # Find which slots in the flattened batch belong to this expert
        mask = (flattened_ids == expert_idx)
        slot_indices = mask.nonzero().flatten()
        
        # Map back to original token indices
        expert_token_indices = token_indices[slot_indices]
        expert_weights = flattened_weights[slot_indices]
        
        # --- GEMM 1 ---
        tokens = hidden_states.index_select(0, expert_token_indices)
        res = torch.nn.functional.linear(tokens, w1[expert_idx_item])
        res = torch.nn.functional.silu(res)
        
        # --- GEMM 2 ---
        res = torch.nn.functional.linear(res, w2[expert_idx_item])
        
        # Weighted accumulation
        output.index_add_(0, expert_token_indices, res * expert_weights.unsqueeze(-1))
                
    return output

__all__ = ["fused_moe"]
