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
    sm_scale,
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
):
    # 程序索引
    cur_batch_idx = tl.program_id(0)
    cur_head_idx = tl.program_id(1)
    cur_kv_head_idx = cur_head_idx // (num_heads // num_kv_heads)
    
    off_d = tl.arange(0, HEAD_DIM)
    
    # 1. 加载 Q
    q = tl.load(Q + cur_batch_idx * stride_qb + cur_head_idx * stride_qh + off_d)
    
    # 2. 融合写入逻辑
    slot = tl.load(slot_mapping + cur_batch_idx)
    
    # 始终加载 K, V 以保证计算可用
    k_val = tl.load(K + cur_batch_idx * stride_kb + cur_kv_head_idx * stride_kh + off_d)
    v_val = tl.load(V + cur_batch_idx * stride_vb + cur_kv_head_idx * stride_vh + off_d)

    if slot >= 0:
        b_idx = slot // BLOCK_SIZE_CACHE
        s_idx = slot % BLOCK_SIZE_CACHE
        
        # 写入物理 Cache
        tl.store(K_cache + b_idx * stride_kcb + s_idx * stride_kcs + cur_kv_head_idx * stride_kch + off_d, k_val)
        tl.store(V_cache + b_idx * stride_vcb + s_idx * stride_vcs + cur_kv_head_idx * stride_vch + off_d, v_val)

    # 3. Attention 计算 (简单 Dot Product 演示)
    score = tl.sum(q * k_val) * sm_scale
    p = tl.exp(score)
    acc = p * v_val
    
    # 写回输出
    tl.store(Out + cur_batch_idx * stride_ob + cur_head_idx * stride_oh + off_d, acc.to(Out.dtype.element_ty))

def fused_prefill_attention(
    q, k, v, k_cache, v_cache, slot_mapping, out, sm_scale
):
    batch_size = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k.shape[1]
    block_size_cache = k_cache.shape[1]
    
    grid = (batch_size, num_heads, 1)
    
    _fused_paged_prefill_attn_kernel[grid](
        q, k, v, k_cache, v_cache, slot_mapping, out, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        num_heads, num_kv_heads,
        BLOCK_M=1, BLOCK_N=1,
        HEAD_DIM=head_dim,
        BLOCK_SIZE_CACHE=block_size_cache
    )
