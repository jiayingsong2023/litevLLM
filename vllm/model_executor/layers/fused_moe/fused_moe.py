# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Optional

def fused_moe(hidden_states, w1, w2, gating_output, topk, renormalize=True):
    """
    ULTRA-STABLE MoE Dispatcher for AMD APU.
    Removed torch.unique to avoid driver-level illegal memory access errors on 35B+ models.
    """
    M, K = hidden_states.shape
    E = w1.shape[0]
    
    # 1. Routing
    routing_weights = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
    output = torch.zeros_like(hidden_states)
    
    # 2. Stability Path: Vectorized Masking (Avoids 'unique' and 'sort')
    # We iterate over experts, but the mask is fully vectorized on GPU.
    # This is slightly slower than sorted grouping but 100% stable.
    
    flattened_ids = topk_ids.view(-1)
    flattened_weights = topk_weights.view(-1)
    token_indices = torch.arange(M, device=hidden_states.device).repeat_interleave(topk)

    # For large models, the number of experts E can be 64 or 256.
    # Masking is highly robust because it doesn't involve dynamic GPU index generation.
    for expert_idx in range(E):
        # 1. Find tokens for this expert via boolean masking (Atomic & Stable)
        mask = (flattened_ids == expert_idx)
        if not mask.any(): continue
        
        # 2. Select indices
        expert_token_indices = token_indices[mask]
        expert_weights = flattened_weights[mask].unsqueeze(-1)
        
        # 3. Pull and Compute
        tokens = hidden_states.index_select(0, expert_token_indices)
        res = torch.nn.functional.linear(tokens, w1[expert_idx])
        
        # SwiGLU check
        if res.shape[-1] == 2 * w2.shape[-1]:
            d = res.shape[-1] // 2
            res = torch.nn.functional.silu(res[:, :d]) * res[:, d:]
        else:
            res = torch.nn.functional.silu(res)
            
        res = torch.nn.functional.linear(res, w2[expert_idx])
        
        # 4. Scatter Add
        output.index_add_(0, expert_token_indices, res * expert_weights)
                
    return output

__all__ = ["fused_moe"]
