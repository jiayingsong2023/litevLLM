# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Optional

def fused_moe(hidden_states, w1, w2, gating_output, topk, renormalize=True):
    """
    使用 PyTorch 稳定算子实现的 MoE，用于排除 BS=32 时的硬件异常。
    """
    M, K = hidden_states.shape
    E, N, _ = w1.shape
    
    # 1. Routing
    routing_weights = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
    # 2. Execution using stable PyTorch operations
    output = torch.zeros_like(hidden_states)
    
    for k in range(topk):
        weights = topk_weights[:, k]
        ids = topk_ids[:, k]
        
        for expert_idx in range(E):
            mask = (ids == expert_idx)
            if not mask.any():
                continue
            
            # 使用 PyTorch 原生高效索引，这在所有架构上都是最稳定的
            expert_indices = mask.nonzero().flatten()
            tokens = hidden_states.index_select(0, expert_indices)
            
            # 执行专家计算
            res = torch.nn.functional.linear(tokens, w1[expert_idx])
            res = torch.nn.functional.silu(res) 
            res = torch.nn.functional.linear(res, w2[expert_idx])
            
            # 应用权重并加回结果
            weighted_res = res * weights[expert_indices].unsqueeze(-1)
            output.index_add_(0, expert_indices, weighted_res)
                
    return output

__all__ = ["fused_moe"]
