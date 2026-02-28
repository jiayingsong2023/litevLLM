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
    # 程序索引: 每个 Program 处理 BLOCK_M 个 Query
    start_m = tl.program_id(0) * BLOCK_M
    cur_head_idx = tl.program_id(1)
    cur_kv_head_idx = cur_head_idx // (num_heads // num_kv_heads)
    
    off_d = tl.arange(0, HEAD_DIM)
    off_m = start_m + tl.arange(0, BLOCK_M)
    
    # 1. 向量化加载 Q Block
    q = tl.load(Q + off_m[:, None] * stride_qb + cur_head_idx * stride_qh + off_d[None, :])
    
    # 2. 融合写入逻辑 (按块写入物理 Cache)
    # 在 Prefill 阶段，当前 Program 处理的 BLOCK_M 个 Token 的 KV 需要存入物理 Cache
    # 这部分逻辑通过 slot_mapping 寻址
    for i in range(BLOCK_M):
        token_idx = start_m + i
        slot = tl.load(slot_mapping + token_idx)
        if slot >= 0:
            b_idx = slot // BLOCK_SIZE_CACHE
            s_idx = slot % BLOCK_SIZE_CACHE
            
            k_val = tl.load(K + token_idx * stride_kb + cur_kv_head_idx * stride_kh + off_d)
            v_val = tl.load(V + token_idx * stride_vb + cur_kv_head_idx * stride_vh + off_d)
            
            tl.store(K_cache + b_idx * stride_kcb + s_idx * stride_kcs + cur_kv_head_idx * stride_kch + off_d, k_val)
            tl.store(V_cache + b_idx * stride_vcb + s_idx * stride_vcs + cur_kv_head_idx * stride_vch + off_d, v_val)

    # 3. 块级 Attention 计算 (FlashAttention 核心 Tiling)
    # 为了性能，此处使用刚加载到寄存器的 K, V 进行局部 Attention
    # 在生产级 Prefill 中，这里会遍历 start_m 之前的所有 KV 块
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')

    # 简化：仅与当前 BLOCK_M 内的 KV 进行计算 (Causal 遮蔽)
    for start_n in range(0, start_m + BLOCK_M, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        
        # 加载 K, V Block
        k_block = tl.load(K + off_n[:, None] * stride_kb + cur_kv_head_idx * stride_kh + off_d[None, :])
        v_block = tl.load(V + off_n[:, None] * stride_vb + cur_kv_head_idx * stride_vh + off_d[None, :])
        
        # 计算 Dot Product [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k_block)) * sm_scale
        
        # Causal Mask
        mask = off_m[:, None] >= off_n[None, :]
        qk = tl.where(mask, qk, -float('inf'))
        
        # Softmax 更新
        m_ij = tl.max(qk, axis=1)
        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        p = tl.exp(qk - m_next[:, None])
        
        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        
        # 更新累加值
        acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block)
        m_i = m_next

    # 归一化并写回
    acc = acc / l_i[:, None]
    tl.store(Out + off_m[:, None] * stride_ob + cur_head_idx * stride_oh + off_d[None, :], acc.to(Out.dtype.element_ty))

def fused_prefill_attention(
    q, k, v, k_cache, v_cache, slot_mapping, out, sm_scale
):
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k.shape[1]
    block_size_cache = k_cache.shape[1]
    
    # 块大小配置 (根据 GPU 资源微调)
    BLOCK_M = 16
    BLOCK_N = 16
    
    # Grid: [M_Blocks, Heads]
    grid = (triton.cdiv(num_tokens, BLOCK_M), num_heads)
    
    _fused_paged_prefill_attn_kernel[grid](
        q, k, v, k_cache, v_cache, slot_mapping, out, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        num_heads, num_kv_heads,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        HEAD_DIM=head_dim,
        BLOCK_SIZE_CACHE=block_size_cache,
        num_warps=4,
        num_stages=2
    )
