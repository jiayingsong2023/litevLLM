# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def paged_attention_kernel(
    Q, K_cache, V_cache, Out,
    stride_qt, stride_qh, stride_qd,
    stride_kct, stride_kcs, stride_kch, stride_kcd,
    stride_vct, stride_vcs, stride_vch, stride_vcd,
    block_tables,
    stride_bts, stride_btb,
    scale,
    seq_lens,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    cur_seq_idx = tl.program_id(0)
    cur_head_idx = tl.program_id(1)
    
    cur_seq_len = tl.load(seq_lens + cur_seq_idx)
    num_blocks = (cur_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    off_d = tl.arange(0, HEAD_DIM)
    q_ptr = Q + cur_seq_idx * stride_qt + cur_head_idx * stride_qh + off_d
    q = tl.load(q_ptr)
    
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    l_i = 0.0
    m_i = -float('inf')
    
    # Offsets for within-block access
    off_s = tl.arange(0, BLOCK_SIZE)
    
    for b_idx in range(num_blocks):
        block_number = tl.load(block_tables + cur_seq_idx * stride_bts + b_idx * stride_btb)
        
        # Load the whole block of K [BLOCK_SIZE, HEAD_DIM]
        # K_cache shape: [num_blocks, block_size, num_heads, head_dim]
        k_ptr = K_cache + block_number * stride_kct + \
                off_s[:, None] * stride_kcs + \
                cur_head_idx * stride_kch + \
                off_d[None, :] * stride_kcd
        
        k_block = tl.load(k_ptr) # [BLOCK_SIZE, HEAD_DIM]
        
        # Compute scores for the block
        # q: [HEAD_DIM], k_block: [BLOCK_SIZE, HEAD_DIM]
        qk = tl.sum(q[None, :] * k_block, axis=1) * scale # [BLOCK_SIZE]
        
        # Masking for last block
        if b_idx == num_blocks - 1:
            boundary = cur_seq_len - b_idx * BLOCK_SIZE
            mask = off_s < boundary
            qk = tl.where(mask, qk, -float('inf'))
        
        m_ij = tl.max(qk, axis=0)
        m_ij = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_ij)
        
        l_ij = tl.sum(p, axis=0)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        # Load V block
        v_ptr = V_cache + block_number * stride_vct + \
                off_s[:, None] * stride_vcs + \
                cur_head_idx * stride_vch + \
                off_d[None, :] * stride_vcd
        v_block = tl.load(v_ptr) # [BLOCK_SIZE, HEAD_DIM]
        
        acc = acc * alpha + tl.sum(p[:, None] * v_block, axis=0)
        m_i = m_ij

    acc = acc / l_i
    out_ptr = Out + cur_seq_idx * stride_qt + cur_head_idx * stride_qh + off_d
    tl.store(out_ptr, acc.to(Out.dtype.element_ty))

def triton_paged_attention(q, k, v, kv_cache, slot_mapping, seq_lens, block_tables, scale):
    if isinstance(kv_cache, (list, tuple)):
        k_cache, v_cache = kv_cache
    else:
        return torch.zeros_like(q)

    # 1. Update Cache
    num_blocks, block_size, num_heads, head_dim = k_cache.shape
    b_indices = slot_mapping // block_size
    b_offsets = slot_mapping % block_size
    
    # Optimized cache update
    k_cache[b_indices, b_offsets] = k
    v_cache[b_indices, b_offsets] = v
    
    # 2. Run Kernel
    num_seqs = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    out = torch.empty_like(q)
    
    grid = (num_seqs, num_heads)
    
    if block_tables is None:
         return out

    paged_attention_kernel[grid](
        q, k_cache, v_cache, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables,
        block_tables.stride(0), block_tables.stride(1),
        scale,
        seq_lens,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        num_warps=4 if head_dim <= 64 else 8
    )
    return out

__all__ = ["triton_paged_attention"]
