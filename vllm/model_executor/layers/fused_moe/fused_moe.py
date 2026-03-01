# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Any, Optional
from vllm.kernels.triton.moe_gemm import index_aware_linear

def _fused_moe_inner(hidden_states, w1, w2, gating_output, topk, renormalize=True):
    M, K = hidden_states.shape
    E, N, _ = w1.shape
    
    routing_weights = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
    output = torch.zeros_like(hidden_states)
    intermediate = torch.zeros((M, N), device=hidden_states.device, dtype=hidden_states.dtype)
    
    BLOCK_SIZE_M = 32
    
    for k in range(topk):
        weights = topk_weights[:, k]
        ids = topk_ids[:, k]
        
        for expert_idx in range(E):
            mask = (ids == expert_idx)
            if not mask.any():
                continue
            
            expert_indices = mask.nonzero().flatten()
            num_actual = expert_indices.shape[0]
            num_padded = ((num_actual + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * BLOCK_SIZE_M
            
            padded_indices = torch.zeros(num_padded, device=expert_indices.device, dtype=expert_indices.dtype)
            padded_indices[:num_actual] = expert_indices

            index_aware_linear(hidden_states, w1[expert_idx], padded_indices, intermediate)
            intermediate[expert_indices] = torch.nn.functional.silu(intermediate[expert_indices])
            index_aware_linear(intermediate, w2[expert_idx], padded_indices, output, weight=weights[expert_indices].mean().item())
                
    return output

def fused_moe(hidden_states, w1, w2, gating_output, topk, renormalize=True):
    M = hidden_states.shape[0]
    MAX_SAFE_BATCH_SIZE = 8
    
    if M <= MAX_SAFE_BATCH_SIZE:
        return _fused_moe_inner(hidden_states, w1, w2, gating_output, topk, renormalize)
    
    # 终极稳定性修复：预分配输出缓冲区，逐片写入，配合显式同步
    output = torch.zeros_like(hidden_states)
    
    for i in range(0, M, MAX_SAFE_BATCH_SIZE):
        chunk_size = min(MAX_SAFE_BATCH_SIZE, M - i)
        chunk_h = hidden_states[i : i + chunk_size]
        chunk_g = gating_output[i : i + chunk_size]
        
        # 执行子任务并同步
        out_chunk = _fused_moe_inner(chunk_h, w1, w2, chunk_g, topk, renormalize)
        output[i : i + chunk_size] = out_chunk
        
        # 核心：通过同步强制 GPU 完成当前指令，防止指令堆叠触发异常
        torch.cuda.synchronize()
        
    return output

__all__ = ["fused_moe"]
