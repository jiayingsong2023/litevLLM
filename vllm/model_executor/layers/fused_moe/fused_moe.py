# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import os
from typing import Any, Optional

def _convert_to_fp8_with_scales(w, BS=64):
    """Utility to convert weight to FP8 with block scales."""
    f8_type = torch.float8_e4m3fn
    out_dim, in_dim = w.shape
    scales = torch.ones((out_dim // BS, in_dim // BS), device="cuda", dtype=torch.float32)
    fp8_w = torch.empty_like(w, dtype=f8_type)
    
    for r in range(scales.shape[0]):
        for c in range(scales.shape[1]):
            block = w[r*BS:(r+1)*BS, c*BS:(c+1)*BS]
            s = block.abs().max().clamp(min=1e-12).item() / 448.0
            scales[r, c] = s
            fp8_w[r*BS:(r+1)*BS, c*BS:(c+1)*BS].copy_((block / s).to(f8_type))
    return fp8_w, scales

_moe_fp8_cache = {}

def fused_moe(hidden_states, w1, w2, gating_output, topk, renormalize=True):
    """
    Intelligent Tiered MoE Dispatcher with FP8 Support.
    """
    M, K = hidden_states.shape
    E = w1.shape[0]
    
    fp8_enabled = os.environ.get("FASTINFERENCE_DEEPSEEK_FP8", "0") == "1"
    
    # 1. Routing
    routing_weights = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
    output = torch.zeros_like(hidden_states)
    
    # 2. FP8 Pre-conversion (Lazy Initialization)
    if fp8_enabled and id(w1) not in _moe_fp8_cache:
        print(f"[fused_moe] Pre-converting {E} experts to FP8 for acceleration...")
        w1_f8_list, w1_s_list = [], []
        w2_f8_list, w2_s_list = [], []
        for i in range(E):
            w1f8, w1s = _convert_to_fp8_with_scales(w1[i])
            w2f8, w2s = _convert_to_fp8_with_scales(w2[i])
            w1_f8_list.append(w1f8); w1_s_list.append(w1s)
            w2_f8_list.append(w2f8); w2_s_list.append(w2s)
        
        _moe_fp8_cache[id(w1)] = (
            torch.stack(w1_f8_list), torch.stack(w1_s_list),
            torch.stack(w2_f8_list), torch.stack(w2_s_list)
        )

    # 3. Strategy Selection
    use_serialized_mode = (E >= 256 or K > 4096 or M > 128)
    
    if not use_serialized_mode:
        # --- ASYNC VECTORIZED PATH ---
        from vllm.kernels.triton.fp8_gemm import fp8_block_gemm
        
        flattened_ids = topk_ids.view(-1)
        flattened_weights = topk_weights.view(-1)
        token_indices = torch.arange(M, device=hidden_states.device).repeat_interleave(topk)
        
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
            
            if fp8_enabled and id(w1) in _moe_fp8_cache:
                # FP8 Path
                w1_f8, w1_s, w2_f8, w2_s = _moe_fp8_cache[id(w1)]
                
                # Input scaling
                t_f8 = tokens.to(torch.float8_e4m3fn)
                t_s = torch.ones((count, tokens.shape[1] // 64), device="cuda", dtype=torch.float32) * (tokens.abs().max() / 448.0)
                
                res = fp8_block_gemm(t_f8, w1_f8[expert_idx].T, t_s, w1_s[expert_idx])
                
                if res.shape[-1] == 2 * w2.shape[-1]:
                    d = res.shape[-1] // 2
                    res = torch.nn.functional.silu(res[:, :d]) * res[:, d:]
                else:
                    res = torch.nn.functional.silu(res)
                
                # Intermediate scaling
                res_f8 = res.to(torch.float8_e4m3fn)
                res_s = torch.ones((count, res.shape[1] // 64), device="cuda", dtype=torch.float32) * (res.abs().max() / 448.0)
                
                res = fp8_block_gemm(res_f8, w2_f8[expert_idx].T, res_s, w2_s[expert_idx])
            else:
                # FP16 Fallback
                res = torch.nn.functional.linear(tokens, w1[expert_idx])
                if res.shape[-1] == 2 * w2.shape[-1]:
                    d = res.shape[-1] // 2
                    res = torch.nn.functional.silu(res[:, :d]) * res[:, d:]
                else:
                    res = torch.nn.functional.silu(res)
                res = torch.nn.functional.linear(res, w2[expert_idx])
                
            output.index_add_(0, group_token_indices, res * group_weights)
    else:
        # --- OLLAMA-STYLE SERIALIZED PATH (Stay in FP16 for stability) ---
        torch.cuda.synchronize()
        for m in range(M):
            token_expert_ids = topk_ids[m]
            token_expert_weights = topk_weights[m]
            token_hidden = hidden_states[m:m+1]
            token_output = torch.zeros_like(token_hidden)
            
            for k in range(topk):
                expert_idx = token_expert_ids[k].item()
                weight = token_expert_weights[k].item()
                
                res = torch.nn.functional.linear(token_hidden, w1[expert_idx])
                if res.shape[-1] == 2 * w2.shape[-1]:
                    d = res.shape[-1] // 2
                    res = torch.nn.functional.silu(res[:, :d]) * res[:, d:]
                else:
                    res = torch.nn.functional.silu(res)
                res = torch.nn.functional.linear(res, w2[expert_idx])
                token_output += res * weight
            output[m] = token_output
                
    return output

__all__ = ["fused_moe"]
