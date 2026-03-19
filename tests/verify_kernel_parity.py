# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
from vllm.kernels.triton.paged_attention import paged_attention_v1
import numpy as np

def test_paged_attention_parity(num_heads=32, head_dim=64):
    print("="*60)
    print(f"Kernel Parity Test: Triton vs PyTorch (H={num_heads}, D={head_dim})")
    print("="*60)
    
    num_kv_heads = num_heads
    seq_len = 16
    block_size = 16
    device = "cuda"
    dtype = torch.float16
    
    q = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    
    # SDPA for reference
    ref_out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False)
    ref_out = ref_out.transpose(1, 2).reshape(-1, num_heads, head_dim)
    
    # Manually build cache to match paged_attention_v1 EXPECTATIONS
    # K layout in kernel: kv_head_idx * s_head + (dim//x) * s_dim + token * s_token + (dim%x) * s_x
    # If we use simple [block, head, dim, token]... wait, the kernel code is complex.
    # Let's use the most standard vLLM layout: [num_blocks, num_kv_heads, head_dim, block_size]
    # But let's check strides.
    
    k_cache = torch.zeros(1, num_kv_heads, head_dim, block_size, device=device, dtype=dtype)
    v_cache = torch.zeros(1, num_kv_heads, block_size, head_dim, device=device, dtype=dtype)
    
    # Fill K [1, H, D, B]
    for h in range(num_kv_heads):
        for d in range(head_dim):
            for b in range(seq_len):
                k_cache[0, h, d, b] = k[0, b, h, d]
                
    # Fill V [1, H, B, D]
    for h in range(num_kv_heads):
        for b in range(seq_len):
            for d in range(head_dim):
                v_cache[0, h, b, d] = v[0, b, h, d]
                
    block_tables = torch.zeros(1, 1, device=device, dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    
    triton_out = torch.empty_like(ref_out)
    try:
        # Note: scale is usually 1/sqrt(head_dim)
        paged_attention_v1(triton_out, q.view(-1, num_heads, head_dim).contiguous(), k_cache, v_cache, num_heads, head_dim**-0.5, block_tables, seq_lens, block_size, 4096, None, "auto")
        
        cos_sim = F.cosine_similarity(triton_out.flatten(), ref_out.flatten(), dim=0).item()
        print(f"  Cosine Similarity: {cos_sim:.6f}")
        if cos_sim > 0.99: print("  ✅ PASS.")
        else: print("  ❌ FAIL. Logic still mismatched.")
    except Exception as e:
        print(f"  ❌ CRASH: {e}")

if __name__ == "__main__":
    test_paged_attention_parity(32, 64)
