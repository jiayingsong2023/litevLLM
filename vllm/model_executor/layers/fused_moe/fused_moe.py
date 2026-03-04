# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Optional

def fused_moe(hidden_states, w1, w2, gating_output, topk, renormalize=True):
    """
    ULTRA-STABLE DUAL-MODE DISPATCHER (Final Version)
    Optimized for AMD Strix Point (gfx1151).
    - Tier 1/2 (< 256 experts): Full GPU performance.
    - Tier 3 (>= 256 experts): Serial CPU-GPU synchronized stability.
    """
    M, K = hidden_states.shape
    E = w1.shape[0]
    
    # 1. Routing
    routing_weights = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
    output = torch.zeros_like(hidden_states)
    
    # 2. Tier Selection
    use_ultra_stability = (E >= 256)
    
    if not use_ultra_stability:
        # --- PERFORMANCE PATH ---
        flattened_ids = topk_ids.view(-1)
        flattened_weights = topk_weights.view(-1)
        token_indices = torch.arange(M, device=hidden_states.device).repeat_interleave(topk)
        
        sorted_ids, sorting_indices = torch.sort(flattened_ids)
        sorted_token_indices = token_indices[sorting_indices]
        sorted_weights = flattened_weights[sorting_indices]
        
        unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        
        curr_offset = 0
        for i in range(len(unique_ids)):
            expert_idx = unique_ids[i].item()
            count = counts[i].item()
            start, end = curr_offset, curr_offset + count
            curr_offset += count
            
            idx = sorted_token_indices[start:end]
            w = sorted_weights[start:end].unsqueeze(-1)
            
            tokens = hidden_states.index_select(0, idx)
            res = torch.nn.functional.linear(tokens, w1[expert_idx])
            # SwiGLU
            if res.shape[-1] == 2 * w2.shape[-1]:
                d = res.shape[-1] // 2
                res = torch.nn.functional.silu(res[:, :d]) * res[:, d:]
            else:
                res = torch.nn.functional.silu(res)
            res = torch.nn.functional.linear(res, w2[expert_idx])
            output.index_add_(0, idx, res * w)
    else:
        # --- ULTRA-STABILITY PATH (For 35B/48B) ---
        # 1. Force Sync and Move to CPU to avoid dynamic GPU index errors
        torch.cuda.synchronize()
        ids_cpu = topk_ids.detach().cpu()
        weights_cpu = topk_weights.detach().cpu()
        active_experts = torch.unique(ids_cpu).tolist()
        
        for expert_idx in active_experts:
            # 2. Find indices on CPU (Guaranteed stable)
            mask_cpu = (ids_cpu == expert_idx)
            # Find which row in the batch and which top-k slot
            matched_indices = mask_cpu.nonzero(as_tuple=True)
            batch_indices = matched_indices[0].to(hidden_states.device)
            slot_indices = matched_indices[1]
            
            # Get weights for this expert
            expert_weights = weights_cpu[mask_cpu].to(hidden_states.device).unsqueeze(-1)
            
            # 3. Step-by-step GPU execution with explicit barriers
            tokens = hidden_states.index_select(0, batch_indices)
            res = torch.nn.functional.linear(tokens, w1[expert_idx])
            
            # SwiGLU logic
            if res.shape[-1] == 2 * w2.shape[-1]:
                d = res.shape[-1] // 2
                res = torch.nn.functional.silu(res[:, :d]) * res[:, d:]
            else:
                res = torch.nn.functional.silu(res)
                
            res = torch.nn.functional.linear(res, w2[expert_idx])
            
            # 4. Safe accumulation
            output.index_add_(0, batch_indices, res * expert_weights)
            # Final barrier for this expert
            torch.cuda.synchronize()
                
    return output

__all__ = ["fused_moe"]
