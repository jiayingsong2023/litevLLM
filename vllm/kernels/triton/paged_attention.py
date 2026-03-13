# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def _paged_attention_kernel(
    Out_ptr, Q_ptr, K_ptr, V_ptr, BlockTables_ptr, SeqLens_ptr,
    scale, num_seqs, num_heads, num_kv_heads, head_size: tl.constexpr,
    block_size: tl.constexpr, max_num_blocks_per_seq,
    stride_out_seq, stride_out_head, stride_out_dim,
    stride_q_seq, stride_q_head, stride_q_dim,
    stride_k_block, stride_k_head, stride_k_dim, stride_k_token, stride_k_x,
    stride_v_block, stride_v_head, stride_v_dim, stride_v_token,
    stride_bt_seq, stride_bt_block, stride_sl_seq,
    k_scale, v_scale,
    IS_FP8: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr, KV_X: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_idx = pid // num_heads; head_idx = pid % num_heads
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    seq_len = tl.load(SeqLens_ptr + seq_idx * stride_sl_seq)
    
    off_q = seq_idx * stride_q_seq + head_idx * stride_q_head + tl.arange(0, BLOCK_D) * stride_q_dim
    q = tl.load(Q_ptr + off_q).to(tl.float32)
    
    m_i = -float('inf'); l_i = 1.0; acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    num_blocks = (seq_len + block_size - 1) // block_size
    
    for i in range(0, num_blocks):
        block_idx_ptr = BlockTables_ptr + seq_idx * stride_bt_seq + i * stride_bt_block
        physical_block_idx = tl.load(block_idx_ptr)
        
        offs_n = tl.arange(0, BLOCK_N); global_token_idx = i * block_size + offs_n
        block_mask = global_token_idx < seq_len
        offs_d = tl.arange(0, BLOCK_D); offs_n_2d = offs_n[:, None]; offs_d_2d = offs_d[None, :]
        
        # Load K (with optional FP8 Dequant)
        off_k = physical_block_idx * stride_k_block + kv_head_idx * stride_k_head + \
                (offs_d_2d // KV_X) * stride_k_dim + offs_n_2d * stride_k_token + (offs_d_2d % KV_X) * stride_k_x
        k = tl.load(K_ptr + off_k, mask=block_mask[:, None], other=0.0)
        if IS_FP8: k = (k.to(tl.float32) * k_scale)
        
        qk = tl.sum(q[None, :] * k, axis=1) * scale
        qk = tl.where(block_mask, qk, -float('inf'))
        m_curr = tl.max(qk, axis=0); m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new); exp_qk = tl.exp(qk - m_new)
        l_curr = tl.sum(exp_qk, axis=0); l_new = alpha * l_i + l_curr
        
        # Load V (with optional FP8 Dequant)
        off_v = physical_block_idx * stride_v_block + kv_head_idx * stride_v_head + \
                offs_d_2d * stride_v_dim + offs_n_2d * stride_v_token
        v = tl.load(V_ptr + off_v, mask=block_mask[:, None], other=0.0)
        if IS_FP8: v = (v.to(tl.float32) * v_scale)
        
        acc = acc * alpha + tl.sum(exp_qk[:, None] * v, axis=0)
        l_i = l_new; m_i = m_new

    out = (acc / l_i).to(Out_ptr.dtype.element_ty)
    off_out = seq_idx * stride_out_seq + head_idx * stride_out_head + tl.arange(0, BLOCK_D) * stride_out_dim
    tl.store(Out_ptr + off_out, out)

def paged_attention_v1(out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype, k_scale=1.0, v_scale=1.0, **kwargs):
    num_seqs, num_heads, head_size = query.shape
    is_fp8 = "fp8" in str(kv_cache_dtype).lower()
    
    s_k = key_cache.stride(); s_v = value_cache.stride()
    if len(s_k) == 5:
        stride_k_block, stride_k_head, stride_k_dim, stride_k_token, stride_k_x = s_k
        kv_x = key_cache.shape[-1]
    else:
        stride_k_block, stride_k_head, stride_k_dim, stride_k_token = s_k
        stride_k_x = 1; kv_x = 1
        
    grid = (num_seqs * num_heads,)
    _paged_attention_kernel[grid](
        out, query, key_cache, value_cache, block_tables, seq_lens,
        scale, num_seqs, num_heads, num_kv_heads, head_size, block_size, block_tables.shape[1],
        out.stride(0), out.stride(1), out.stride(2),
        query.stride(0), query.stride(1), query.stride(2),
        stride_k_block, stride_k_head, stride_k_dim, stride_k_token, stride_k_x,
        s_v[0], s_v[1], s_v[2], s_v[3],
        block_tables.stride(0), block_tables.stride(1), seq_lens.stride(0),
        k_scale, v_scale, IS_FP8=is_fp8, BLOCK_D=head_size, BLOCK_N=block_size, KV_X=kv_x
    )

def paged_attention_v2(*args, **kwargs):
    return paged_attention_v1(*args, **kwargs)
