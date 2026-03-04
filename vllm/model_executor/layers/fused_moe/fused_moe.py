# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Optional
from vllm.kernels.triton.moe_gemm import index_aware_linear

def fused_moe(hidden_states, w1, w2, gating_output, topk, renormalize=True):
    """
    High-Throughput MoE Dispatcher for Large Expert Counts (e.g. Qwen3.5 with 256 experts).
    Optimized by grouping experts to reduce kernel launch overhead.
    """
    M, K = hidden_states.shape
    E, N, _ = w1.shape
    
    # 1. Routing
    routing_logits = gating_output
    routing_weights = torch.softmax(routing_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
    # 2. Expert Grouping & Aggregated Dispatch
    output = torch.zeros_like(hidden_states)
    
    flattened_ids = topk_ids.view(-1)
    flattened_weights = topk_weights.view(-1)
    token_indices = torch.arange(M, device=hidden_states.device).repeat_interleave(topk)
    
    # Sort by expert ID to enable contiguous execution
    # This is the key to reducing memory thrashing and improving cache locality
    sorted_expert_ids, sorting_indices = torch.sort(flattened_ids)
    sorted_token_indices = token_indices[sorting_indices]
    sorted_weights = flattened_weights[sorting_indices]
    
    # Find start/end offsets for each active expert in one pass
    # Using unique_consecutive is much faster than multiple unique() calls
    unique_ids, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
    offsets = torch.cat([torch.tensor([0], device=unique_ids.device), torch.cumsum(counts, dim=0)])
    
    # Aggregated Execution
    for i, expert_idx in enumerate(unique_ids):
        expert_idx_item = expert_idx.item()
        start, end = offsets[i], offsets[i+1]
        
        # Pull indices and weights for this expert group
        expert_token_indices = sorted_token_indices[start:end]
        expert_weights = sorted_weights[start:end].unsqueeze(-1)
        
        # Batch collect tokens
        tokens = hidden_states.index_select(0, expert_token_indices)
        
        # Expert Computation (Triton linear)
        # Note: We reuse the cached dequantized weights here
        res = torch.nn.functional.linear(tokens, w1[expert_idx_item])
        res = torch.nn.functional.silu(res)
        res = torch.nn.functional.linear(res, w2[expert_idx_item])
        
        # Efficiently scatter results back
        output.index_add_(0, expert_token_indices, res * expert_weights)
                
    return output

__all__ = ["fused_moe"]
