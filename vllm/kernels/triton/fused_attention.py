# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

# Robust float8 type resolution for different Triton versions
def _get_fp8_dtype():
    # Standard names
    for name in ["float8e4m3fn", "float8_e4m3fn", "float8e4m3fnuz"]:
        if hasattr(tl, name):
            return getattr(tl, name)
    # Vendor specific or older names
    for name in ["float8e4nv", "float8e4b8", "float8e4b15"]:
        if hasattr(tl, name):
            return getattr(tl, name)
    return None

FP8_DTYPE = _get_fp8_dtype()

@triton.jit
def _fused_paged_prefill_attn_kernel(
    Q, K, V,
    K_cache, V_cache,
    slot_mapping,
    Out,
    sm_scale, k_scale_arg, v_scale_arg,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_kcb, stride_kcs, stride_kch, stride_kcd,
    stride_vcb, stride_vcs, stride_vch, stride_vcd,
    K_Scale_cache, V_Scale_cache,
    stride_ksb, stride_kss, stride_ksh,
    stride_vsb, stride_vss, stride_vsh,
    num_heads, num_kv_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_CACHE: tl.constexpr,
    IS_FP8: tl.constexpr,
    IS_INT4: tl.constexpr,
    COMPUTE_DYNAMIC_SCALE: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    cur_head_idx = tl.program_id(1)
    cur_kv_head_idx = cur_head_idx // (num_heads // num_kv_heads)
    
    off_d = tl.arange(0, HEAD_DIM)
    off_m = start_m + tl.arange(0, BLOCK_M)
    
    q = tl.load(Q + off_m[:, None] * stride_qb + cur_head_idx * stride_qh + off_d[None, :])
    
    # 融合写入 (带量化支持)
    # Only need to write KV once per KV-head. Since program_id(1) is num_heads, 
    # multiple heads might try to write the same KV. 
    # Use only first head in group to write.
    is_write_head = (cur_head_idx % (num_heads // num_kv_heads)) == 0
    
    if is_write_head:
        for i in range(BLOCK_M):
            token_idx = start_m + i
            slot = tl.load(slot_mapping + token_idx)
            if slot >= 0:
                b_idx = slot // BLOCK_SIZE_CACHE
                s_idx = slot % BLOCK_SIZE_CACHE
                
                k_val = tl.load(K + token_idx * stride_kb + cur_kv_head_idx * stride_kh + off_d).to(tl.float32)
                v_val = tl.load(V + token_idx * stride_vb + cur_kv_head_idx * stride_vh + off_d).to(tl.float32)

                k_scale = k_scale_arg
                v_scale = v_scale_arg

                if COMPUTE_DYNAMIC_SCALE:
                    k_scale = tl.max(tl.abs(k_val)) / 7.0
                    v_scale = tl.max(tl.abs(v_val)) / 7.0
                    k_scale = tl.where(k_scale == 0, 1.0, k_scale)
                    v_scale = tl.where(v_scale == 0, 1.0, v_scale)
                    # Store scales
                    ks_ptr = K_Scale_cache + b_idx * stride_ksb + s_idx * stride_kss + cur_kv_head_idx * stride_ksh
                    vs_ptr = V_Scale_cache + b_idx * stride_vsb + s_idx * stride_vss + cur_kv_head_idx * stride_vsh
                    tl.store(ks_ptr, k_scale)
                    tl.store(vs_ptr, v_scale)

                if IS_INT4:
                    off_d_half = tl.arange(0, HEAD_DIM // 2)
                    k_l = tl.load(K + token_idx * stride_kb + cur_kv_head_idx * stride_kh + (off_d_half * 2)).to(tl.float32)
                    k_h = tl.load(K + token_idx * stride_kb + cur_kv_head_idx * stride_kh + (off_d_half * 2 + 1)).to(tl.float32)
                    v_l = tl.load(V + token_idx * stride_vb + cur_kv_head_idx * stride_vh + (off_d_half * 2)).to(tl.float32)
                    v_h = tl.load(V + token_idx * stride_vb + cur_kv_head_idx * stride_vh + (off_d_half * 2 + 1)).to(tl.float32)
                    
                    k_l_q = (tl.clamp(tl.math.floor(k_l / k_scale + 0.5), -8.0, 7.0).to(tl.int32) + 8).to(tl.uint8)
                    k_h_q = (tl.clamp(tl.math.floor(k_h / k_scale + 0.5), -8.0, 7.0).to(tl.int32) + 8).to(tl.uint8)
                    v_l_q = (tl.clamp(tl.math.floor(v_l / v_scale + 0.5), -8.0, 7.0).to(tl.int32) + 8).to(tl.uint8)
                    v_h_q = (tl.clamp(tl.math.floor(v_h / v_scale + 0.5), -8.0, 7.0).to(tl.int32) + 8).to(tl.uint8)
                    
                    kc_ptr = K_cache + b_idx * stride_kcb + s_idx * stride_kcs + cur_kv_head_idx * stride_kch + off_d_half * stride_kcd
                    vc_ptr = V_cache + b_idx * stride_vcb + s_idx * stride_vcs + cur_kv_head_idx * stride_vch + off_d_half * stride_vcd
                    tl.store(kc_ptr, k_l_q | (k_h_q << 4))
                    tl.store(vc_ptr, v_l_q | (v_h_q << 4))
                elif IS_FP8:
                    kc_ptr = K_cache + b_idx * stride_kcb + s_idx * stride_kcs + cur_kv_head_idx * stride_kch + off_d
                    vc_ptr = V_cache + b_idx * stride_vcb + s_idx * stride_vcs + cur_kv_head_idx * stride_vch + off_d
                    tl.store(kc_ptr, (k_val * k_scale).to(FP8_DTYPE))
                    tl.store(vc_ptr, (v_val * v_scale).to(FP8_DTYPE))
                else:
                    kc_ptr = K_cache + b_idx * stride_kcb + s_idx * stride_kcs + cur_kv_head_idx * stride_kch + off_d
                    vc_ptr = V_cache + b_idx * stride_vcb + s_idx * stride_vcs + cur_kv_head_idx * stride_vch + off_d
                    tl.store(kc_ptr, k_val.to(K_cache.dtype.element_ty))
                    tl.store(vc_ptr, v_val.to(V_cache.dtype.element_ty))

    # 计算逻辑
    NEG_INF = -3.40282347e+38
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + NEG_INF

    for start_n in range(0, start_m + BLOCK_M, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        k_block = tl.load(K + off_n[:, None] * stride_kb + cur_kv_head_idx * stride_kh + off_d[None, :])
        v_block = tl.load(V + off_n[:, None] * stride_vb + cur_kv_head_idx * stride_vh + off_d[None, :])
        
        qk = tl.dot(q, tl.trans(k_block)) * sm_scale
        mask = off_m[:, None] >= off_n[None, :]
        qk = tl.where(mask, qk, NEG_INF)
        
        m_ij = tl.max(qk, axis=1)
        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp((m_i - m_next).to(tl.float32)).to(tl.float32)
        p = tl.exp((qk - m_next[:, None]).to(tl.float32)).to(tl.float32)
        p = tl.where(mask, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v_block.to(tl.float16))
        m_i = m_next

    acc = acc / (l_i[:, None] + 1e-6)
    tl.store(Out + off_m[:, None] * stride_ob + cur_head_idx * stride_oh + off_d[None, :], acc.to(Out.dtype.element_ty))

def fused_prefill_attention(
    q, k, v, k_cache, v_cache, slot_mapping, out, sm_scale,
    k_scale=1.0, v_scale=1.0, kv_cache_dtype="auto",
    k_scale_cache=None, v_scale_cache=None
):
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k.shape[1]
    block_size_cache = k_cache.shape[1]
    is_fp8 = "fp8" in str(kv_cache_dtype).lower()
    is_int4 = "int4" in str(kv_cache_dtype).lower()
    compute_dynamic = (k_scale_cache is not None and v_scale_cache is not None and is_int4)

    ks = k_scale_cache if compute_dynamic else out # Placeholder
    vs = v_scale_cache if compute_dynamic else out

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
        ks, vs,
        ks.stride(0) if compute_dynamic else 0,
        ks.stride(1) if compute_dynamic else 0,
        ks.stride(2) if compute_dynamic else 0,
        vs.stride(0) if compute_dynamic else 0,
        vs.stride(1) if compute_dynamic else 0,
        vs.stride(2) if compute_dynamic else 0,
        num_heads, num_kv_heads,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,
        BLOCK_SIZE_CACHE=block_size_cache, IS_FP8=is_fp8, IS_INT4=is_int4,
        COMPUTE_DYNAMIC_SCALE=compute_dynamic,
        FP8_DTYPE=FP8_DTYPE,
        num_warps=4, num_stages=2
    )
