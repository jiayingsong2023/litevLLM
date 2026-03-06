# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def _index_aware_gemm_kernel(
    A, B, C,
    Index_Map,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M, N, K,
    weight,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n
    
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 修复 1: 在加载 Index_Map 时加入严格掩码
    # 如果超出 M，则加载 0（指向原始矩阵第 0 行，保证指针计算安全）
    m_mask = off_m < M
    real_m_indices = tl.load(Index_Map + off_m, mask=m_mask, other=0)
    
    # 修复 2: 确保 a_ptrs 计算在 mask 保护下
    a_ptrs = A + (real_m_indices[:, None] * stride_am + off_k[None, :] * stride_ak)
    b_ptrs = B + (off_k[:, None] * stride_bk + off_n[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = off_k[None, :] < K - k * BLOCK_SIZE_K
        # 同时对 M 和 K 进行遮蔽
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask.T & (off_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    res = (accumulator * weight).to(C.dtype.element_ty)
    
    # 修复 3: 写回输出时确保 real_m_indices 是有效的
    c_ptrs = C + (real_m_indices[:, None] * stride_cm + off_n[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, res, mask=m_mask[:, None] & (off_n[None, :] < N))

@triton.jit
def _fused_moe_kernel(
    X,
    W1, W1_Up, W2,
    TopK_Ids, TopK_Weights,
    Out,
    stride_xm, stride_xk,
    stride_we, stride_wk, stride_wn,
    stride_w2e, stride_w2n, stride_w2k,
    stride_om, stride_on,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    TOPK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Process output hidden dimension N (hidden_size) in chunks of BLOCK_SIZE_N
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = off_m < M
    n_mask = off_n < N
    
    # Final output accumulator (M, N)
    out_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # For each token in this block, we compute its top-k experts and add them up
    # We iterate over top-k to avoid massive shared memory allocation for 64 experts
    for k_idx in range(TOPK):
        # expert_ids: [BLOCK_SIZE_M]
        # weights: [BLOCK_SIZE_M]
        expert_ids = tl.load(TopK_Ids + off_m * TOPK + k_idx, mask=m_mask, other=0)
        expert_weights = tl.load(TopK_Weights + off_m * TOPK + k_idx, mask=m_mask, other=0.0)
        
        # Now we need to perform X @ W1[expert_id].T -> intermediate (1408)
        # and then intermediate @ W2[expert_id].T -> hidden (2048)
        # This is a complex fusion. For BS=32, we can just use the standard grouped path
        # as a baseline and then move to a more aggressive tiled approach.
        
        # Simplified: Use existing optimized GEMMs and fuse the weights
        pass

def fused_moe_deepseek_triton(x, w1, w1_up, w2, topk_ids, topk_weights):
    # This dispatcher will call a high-performance tiled kernel.
    # For now, it serves as a placeholder for the 900+ TPS optimization.
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
    # If the specialized Triton kernel is not ready, we use the tiered dispatcher
    # which we optimized previously for 119 tok/s, then move to the target milestone.
    pass

def index_aware_linear(a, b, index_map, out, weight=1.0):
    # 如果该专家没有分配到 token，直接返回
    M = index_map.shape[0]
    if M == 0:
        return
        
    K = a.shape[1]
    N = b.shape[0]
    
    # 动态调整 BLOCK_SIZE_M 以适应极小的 M (例如 M=1 时不需要 BLOCK=32)
    # 但为了 Tensor Core 效率，通常保持 16 或 32 的倍数
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    _index_aware_gemm_kernel[grid](
        a, b, out, index_map,
        a.stride(0), a.stride(1),
        b.stride(1), b.stride(0),
        out.stride(0), out.stride(1),
        M, N, K,
        weight,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8
    )
