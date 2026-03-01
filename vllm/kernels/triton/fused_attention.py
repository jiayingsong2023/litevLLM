# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_paged_prefill_attn_kernel(
    Q, K, V,
    K_cache, V_cache,
    slot_mapping,
    Out,
    sm_scale, k_scale, v_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_kcb, stride_kcs, stride_kch, stride_kcd,
    stride_vcb, stride_vcs, stride_vch, stride_vcd,
    num_heads, num_kv_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_CACHE: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    cur_head_idx = tl.program_id(1)
    cur_kv_head_idx = cur_head_idx // (num_heads // num_kv_heads)
    
    off_d = tl.arange(0, HEAD_DIM)
    off_m = start_m + tl.arange(0, BLOCK_M)
    
    q = tl.load(Q + off_m[:, None] * stride_qb + cur_head_idx * stride_qh + off_d[None, :])
    
    # 融合写入 (带量化支持)
    for i in range(BLOCK_M):
        token_idx = start_m + i
        slot = tl.load(slot_mapping + token_idx)
        if slot >= 0:
            b_idx = slot // BLOCK_SIZE_CACHE
            s_idx = slot % BLOCK_SIZE_CACHE
            
            k_val = tl.load(K + token_idx * stride_kb + cur_kv_head_idx * stride_kh + off_d)
            v_val = tl.load(V + token_idx * stride_vb + cur_kv_head_idx * stride_vh + off_d)
            
            kc_ptr = K_cache + b_idx * stride_kcb + s_idx * stride_kcs + cur_kv_head_idx * stride_kch + off_d
            vc_ptr = V_cache + b_idx * stride_vcb + s_idx * stride_vcs + cur_kv_head_idx * stride_vch + off_d
            
            if IS_FP8:
                tl.store(kc_ptr, (k_val * k_scale).to(tl.float8e4m3fn))
                tl.store(vc_ptr, (v_val * v_scale).to(tl.float8e4m3fn))
            else:
                tl.store(kc_ptr, k_val.to(K_cache.dtype.element_ty))
                tl.store(vc_ptr, v_val.to(V_cache.dtype.element_ty))

    # 计算逻辑保持原有 Tiled FlashAttention 结构
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')

    for start_n in range(0, start_m + BLOCK_M, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        k_block = tl.load(K + off_n[:, None] * stride_kb + cur_kv_head_idx * stride_kh + off_d[None, :])
        v_block = tl.load(V + off_n[:, None] * stride_vb + cur_kv_head_idx * stride_vh + off_d[None, :])
        
        qk = tl.dot(q, tl.trans(k_block)) * sm_scale
        mask = off_m[:, None] >= off_n[None, :]
        qk = tl.where(mask, qk, -float('inf'))
        
        m_ij = tl.max(qk, axis=1)
        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        p = tl.exp(qk - m_next[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block)
        m_i = m_next

    acc = acc / l_i[:, None]
    tl.store(Out + off_m[:, None] * stride_ob + cur_head_idx * stride_oh + off_d[None, :], acc.to(Out.dtype.element_ty))

def fused_prefill_attention(
    q, k, v, k_cache, v_cache, slot_mapping, out, sm_scale,
    k_scale=1.0, v_scale=1.0, kv_cache_dtype="auto"
):
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k.shape[1]
    block_size_cache = k_cache.shape[1]
    is_fp8 = "fp8" in kv_cache_dtype
    
    BLOCK_M, BLOCK_N = 16, 16
    grid = (triton.cdiv(num_tokens, BLOCK_M), num_heads)
    
    _fused_paged_prefill_attn_kernel[grid](
        q, k, v, k_cache, v_cache, slot_mapping, out, sm_scale, k_scale, v_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        num_heads, num_kv_heads,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,
        BLOCK_SIZE_CACHE=block_size_cache, IS_FP8=is_fp8,
        num_warps=4, num_stages=2
    )
