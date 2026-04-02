# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def _paged_attention_kernel(
    Out_ptr, Q_ptr, K_ptr, V_ptr, 
    BlockTables_ptr,
    SeqLens_ptr,
    K_Ptrs_ptr, V_Ptrs_ptr,
    scale, num_seqs, num_heads, num_kv_heads, head_size: tl.constexpr,
    block_size: tl.constexpr, max_num_blocks_per_seq,
    stride_out_seq, stride_out_head, stride_out_dim,
    stride_q_seq, stride_q_head, stride_q_dim,
    stride_k_block, stride_k_token, stride_k_head, stride_k_dim,
    stride_v_block, stride_v_token, stride_v_head, stride_v_dim,
    stride_bt_seq, stride_bt_block, stride_sl_seq,
    k_scale_arg, v_scale_arg,
    K_Scale_Ptrs_ptr, V_Scale_Ptrs_ptr,
    stride_ks_block, stride_ks_token, stride_ks_head,
    stride_vs_block, stride_vs_token, stride_vs_head,
    IS_FP8: tl.constexpr,
    IS_INT4: tl.constexpr,
    IS_CACHED: tl.constexpr,
    HAS_ROW_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_idx = pid // num_heads; head_idx = pid % num_heads
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    seq_len = tl.load(SeqLens_ptr + seq_idx * stride_sl_seq)
    
    # Initialize accumulators
    m_i = -float('inf'); l_i = 0.0
    acc_low = tl.zeros([BLOCK_D // 2], dtype=tl.float32)
    acc_high = tl.zeros([BLOCK_D // 2], dtype=tl.float32)
    acc_full = tl.zeros([BLOCK_D], dtype=tl.float32)

    if IS_INT4:
        offs_d_half = tl.arange(0, BLOCK_D // 2)
        q_low = tl.load(Q_ptr + seq_idx * stride_q_seq + head_idx * stride_q_head + (offs_d_half * 2) * stride_q_dim).to(tl.float32)
        q_high = tl.load(Q_ptr + seq_idx * stride_q_seq + head_idx * stride_q_head + (offs_d_half * 2 + 1) * stride_q_dim).to(tl.float32)
    else:
        off_q = seq_idx * stride_q_seq + head_idx * stride_q_head + tl.arange(0, BLOCK_D) * stride_q_dim
        q = tl.load(Q_ptr + off_q).to(tl.float32)

    num_blocks = (seq_len + block_size - 1) // block_size
    offs_n = tl.arange(0, BLOCK_N)
    
    for i in range(0, num_blocks):
        if IS_CACHED:
            k_base_ptr = tl.load(K_Ptrs_ptr + seq_idx * max_num_blocks_per_seq + i).to(tl.pointer_type(tl.uint8 if IS_INT4 else tl.float16))
            v_base_ptr = tl.load(V_Ptrs_ptr + seq_idx * max_num_blocks_per_seq + i).to(tl.pointer_type(tl.uint8 if IS_INT4 else tl.float16))
            if HAS_ROW_SCALE:
                ks_base_ptr = tl.load(K_Scale_Ptrs_ptr + seq_idx * max_num_blocks_per_seq + i).to(tl.pointer_type(tl.float32))
                vs_base_ptr = tl.load(V_Scale_Ptrs_ptr + seq_idx * max_num_blocks_per_seq + i).to(tl.pointer_type(tl.float32))
        else:
            block_idx = tl.load(BlockTables_ptr + seq_idx * stride_bt_seq + i * stride_bt_block)
            k_base_ptr = K_ptr + block_idx * stride_k_block
            v_base_ptr = V_ptr + block_idx * stride_v_block
            if HAS_ROW_SCALE:
                ks_base_ptr = K_Scale_Ptrs_ptr + block_idx * stride_ks_block
                vs_base_ptr = V_Scale_Ptrs_ptr + block_idx * stride_vs_block

        global_token_idx = i * block_size + offs_n
        block_mask = global_token_idx < seq_len
        
        k_scale = k_scale_arg
        v_scale = v_scale_arg
        if HAS_ROW_SCALE:
            k_scale = tl.load(ks_base_ptr + offs_n * stride_ks_token + kv_head_idx * stride_ks_head, mask=block_mask, other=1.0)
            v_scale = tl.load(vs_base_ptr + offs_n * stride_vs_token + kv_head_idx * stride_vs_head, mask=block_mask, other=1.0)

        if IS_INT4:
            offs_d_half_2d = tl.arange(0, BLOCK_D // 2)[None, :]
            off_kv = offs_n[:, None] * stride_k_token + kv_head_idx * stride_k_head + offs_d_half_2d * stride_k_dim
            k_packed = tl.load(k_base_ptr.to(tl.pointer_type(tl.uint8)) + off_kv, mask=block_mask[:, None], other=0).to(tl.uint8)
            k_l = ((k_packed & 0x0F).to(tl.float32) - 8.0) * k_scale[:, None]
            k_h = (((k_packed >> 4) & 0x0F).to(tl.float32) - 8.0) * k_scale[:, None]
            qk = tl.sum(q_low[None, :] * k_l + q_high[None, :] * k_h, axis=1) * scale
        else:
            offs_d_2d = tl.arange(0, BLOCK_D)[None, :]
            off_kv = offs_n[:, None] * stride_k_token + kv_head_idx * stride_k_head + offs_d_2d * stride_k_dim
            k = tl.load(k_base_ptr + off_kv, mask=block_mask[:, None], other=0.0)
            if IS_FP8: k = (k.to(tl.float32) * k_scale[:, None])
            qk = tl.sum(q[None, :] * k, axis=1) * scale

        qk = tl.where(block_mask, qk, -float('inf'))
        m_curr = tl.max(qk, axis=0); m_new = tl.maximum(m_i, m_curr)
        
        # Softmax stabilization
        delta_m = (m_i - m_new).to(tl.float32)
        alpha = tl.exp(delta_m).to(tl.float32)
        
        p = tl.exp((qk - m_new).to(tl.float32)).to(tl.float32)
        p = tl.where(block_mask, p, 0.0).to(tl.float32)
        l_curr = tl.sum(p, axis=0).to(tl.float32)
        l_new = (alpha * l_i + l_curr).to(tl.float32)
        
        # Guard against zero sum to avoid NaN in division
        l_new = tl.maximum(l_new, 1e-20)
        
        if IS_INT4:
            v_packed = tl.load(v_base_ptr.to(tl.pointer_type(tl.uint8)) + off_kv, mask=block_mask[:, None], other=0).to(tl.uint8)
            v_l = ((v_packed & 0x0F).to(tl.float32) - 8.0) * v_scale[:, None]
            v_h = (((v_packed >> 4) & 0x0F).to(tl.float32) - 8.0) * v_scale[:, None]
            acc_low = acc_low * alpha + tl.sum(p[:, None] * v_l, axis=0)
            acc_high = acc_high * alpha + tl.sum(p[:, None] * v_h, axis=0)
        else:
            v = tl.load(v_base_ptr + off_kv, mask=block_mask[:, None], other=0.0)
            if IS_FP8: v = (v.to(tl.float32) * v_scale[:, None])
            acc_full = acc_full * alpha + tl.sum(p[:, None] * v, axis=0)
        
        l_i = l_new; m_i = m_new

    off_out_base = seq_idx * stride_out_seq + head_idx * stride_out_head
    inv_l_i = 1.0 / tl.maximum(l_i, 1e-20)
    if IS_INT4:
        # Standard Online Softmax normalization
        res_low = (acc_low * inv_l_i).to(Out_ptr.dtype.element_ty)
        res_high = (acc_high * inv_l_i).to(Out_ptr.dtype.element_ty)
        off_d_half = tl.arange(0, BLOCK_D // 2)
        tl.store(Out_ptr + off_out_base + (off_d_half * 2) * stride_out_dim, res_low)
        tl.store(Out_ptr + off_out_base + (off_d_half * 2 + 1) * stride_out_dim, res_high)
    else:
        out = (acc_full * inv_l_i).to(Out_ptr.dtype.element_ty)
        off_d = tl.arange(0, BLOCK_D)
        tl.store(Out_ptr + off_out_base + off_d * stride_out_dim, out)

def paged_attention_v1(out, query, key_cache, value_cache, num_heads, scale, block_tables, seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype, k_scale=1.0, v_scale=1.0, k_ptrs=None, v_ptrs=None, k_scale_ptrs=None, v_scale_ptrs=None, **kwargs):
    num_seqs, _num_heads_check, head_size = query.shape
    num_kv_heads = kwargs.get("num_kv_heads", num_heads)
    is_fp8 = "fp8" in str(kv_cache_dtype).lower()
    is_int4 = "int4" in str(kv_cache_dtype).lower()
    is_cached = k_ptrs is not None
    has_row_scale = k_scale_ptrs is not None
    
    s_k = key_cache.stride(); s_v = value_cache.stride()
    
    if has_row_scale:
        s_ks = k_scale_ptrs.stride(); s_vs = v_scale_ptrs.stride()
    else:
        s_ks = [0]*4; s_vs = [0]*4

    grid = (num_seqs * num_heads,)
    _paged_attention_kernel[grid](
        out, query, key_cache, value_cache, 
        block_tables if not is_cached else out, seq_lens,
        k_ptrs if is_cached else out, v_ptrs if is_cached else out,
        scale, num_seqs, num_heads, num_kv_heads, head_size, block_size, block_tables.shape[1] if block_tables is not None else 0,
        out.stride(0), out.stride(1), out.stride(2),
        query.stride(0), query.stride(1), query.stride(2),
        s_k[0], s_k[1], s_k[2], s_k[3],
        s_v[0], s_v[1], s_v[2], s_v[3],
        block_tables.stride(0) if block_tables is not None else 0,
        block_tables.stride(1) if block_tables is not None else 0,
        seq_lens.stride(0),
        k_scale if not has_row_scale else 1.0, v_scale if not has_row_scale else 1.0,
        k_scale_ptrs if has_row_scale else out, v_scale_ptrs if has_row_scale else out,
        s_ks[0], s_ks[1], s_ks[2],
        s_vs[0], s_vs[1], s_vs[2],
        IS_FP8=is_fp8, IS_INT4=is_int4, IS_CACHED=is_cached, HAS_ROW_SCALE=has_row_scale, BLOCK_D=head_size, BLOCK_N=block_size
    )

def paged_attention_v2(*args, **kwargs):
    return paged_attention_v1(*args, **kwargs)
