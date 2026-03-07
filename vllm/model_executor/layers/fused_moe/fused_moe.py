import torch
import torch.nn as nn
import os
import collections
from typing import Any, Optional

def _convert_to_fp8_with_scales(w, BS=64):
    """Utility to convert weight to FP8 with block scales (Vectorized)."""
    f8_type = torch.float8_e4m3fn
    out_dim, in_dim = w.shape
    
    # 1. Reshape to blocks: [out_dim//BS, BS, in_dim//BS, BS]
    # Then permute to [out_dim//BS, in_dim//BS, BS, BS]
    w_blocks = w.view(out_dim // BS, BS, in_dim // BS, BS).permute(0, 2, 1, 3)
    
    # 2. Calculate scales: max(abs) per block
    # scales shape: [out_dim//BS, in_dim//BS]
    scales = (w_blocks.abs().max(dim=-1)[0].max(dim=-1)[0].clamp(min=1e-12) / 448.0)
    
    # 3. Quantize: w_fp8 = (w / scales)
    # Broadcast scales back to block shape
    w_fp8 = (w_blocks / scales.unsqueeze(-1).unsqueeze(-1)).to(f8_type)
    
    # 4. Reshape back to original 2D shape [out_dim, in_dim]
    # Reverse permute and reshape
    w_fp8 = w_fp8.permute(0, 2, 1, 3).reshape(out_dim, in_dim)
    
    return w_fp8, scales

class FP8ExpertCache:
    """Dynamic LRU Cache for MoE Experts to save VRAM."""
    def __init__(self, max_size=32):
        self.cache = collections.OrderedDict()
        self.max_size = max_size

    def get(self, expert_id, w1_orig, w2_orig):
        # We use (id(w1_orig), expert_id) as the key to support multiple layers
        if isinstance(expert_id, (int, torch.Tensor)):
            key = (id(w1_orig), int(expert_id))
        else:
            key = expert_id
            
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        
        # Convert on the fly (This is the "Balanced Mode")
        if w2_orig is not None:
            w1_f8, w1_s = _convert_to_fp8_with_scales(w1_orig[expert_id] if isinstance(expert_id, (int, torch.Tensor)) else w1_orig)
            w2_f8, w2_s = _convert_to_fp8_with_scales(w2_orig[expert_id] if isinstance(expert_id, (int, torch.Tensor)) else w2_orig)
            val = (w1_f8, w1_s, w2_f8, w2_s)
        else:
            w1_f8, w1_s = _convert_to_fp8_with_scales(w1_orig)
            val = (w1_f8, w1_s)
        
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
            
        self.cache[key] = val
        return val

# LRU Cache Size controllable by Env Var
_lru_cache_size = int(os.environ.get("FASTINFERENCE_MOE_LRU_SIZE", "32"))
_global_fp8_cache = FP8ExpertCache(max_size=_lru_cache_size)

def fused_moe(hidden_states, w1, w2, gating_output, topk, renormalize=True):
    """
    Intelligent Tiered MoE Dispatcher with FP8 + Balanced Cache Mode.
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
    
    # 2. Strategy Selection
    # For massive models or large tokens, use Serialized Dispatch to ensure stability.
    use_serialized_mode = (E >= 256 or K > 4096 or M > 128)
    
    if not use_serialized_mode:
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
            
            if fp8_enabled:
                # Balanced Mode: Get from LRU Cache
                w1_f8, w1_s, w2_f8, w2_s = _global_fp8_cache.get(expert_idx, w1, w2)
                
                t_f8 = tokens.to(torch.float8_e4m3fn)
                t_s = torch.ones((count, tokens.shape[1] // 64), device="cuda", dtype=torch.float32) * (tokens.abs().max() / 448.0)
                
                # Forward Pass
                res = fp8_block_gemm(t_f8, w1_f8.T, t_s, w1_s)
                
                if res.shape[-1] == 2 * w2.shape[-1]:
                    d = res.shape[-1] // 2
                    res = torch.nn.functional.silu(res[:, :d]) * res[:, d:]
                else:
                    res = torch.nn.functional.silu(res)
                
                res_f8 = res.to(torch.float8_e4m3fn)
                res_s = torch.ones((count, res.shape[1] // 64), device="cuda", dtype=torch.float32) * (res.abs().max() / 448.0)
                res = fp8_block_gemm(res_f8, w2_f8.T, res_s, w2_s)
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
        # Serialized path for stability
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
