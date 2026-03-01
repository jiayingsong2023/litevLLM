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
    weight, # 当前专家的路由权重
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
    
    # 索引感知加载
    real_m_indices = tl.load(Index_Map + off_m, mask=off_m < M, other=0)
    
    a_ptrs = A + (real_m_indices[:, None] * stride_am + off_k[None, :] * stride_ak)
    b_ptrs = B + (off_k[:, None] * stride_bk + off_n[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(off_m[:, None] < M) & (off_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(off_k[:, None] < K - k * BLOCK_SIZE_K) & (off_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    # 应用权重并进行原子累加 (Atomic Add) 到全局输出 C
    # 这样多个专家可以并发地向同一个 C 写入结果
    res = (accumulator * weight).to(C.dtype.element_ty)
    c_ptrs = C + (real_m_indices[:, None] * stride_cm + off_n[None, :] * stride_cn)
    
    # 注意：原子累加确保了并发安全性
    tl.atomic_add(c_ptrs, res, mask=(off_m[:, None] < M) & (off_n[None, :] < N))

def index_aware_linear(a, b, index_map, out, weight=1.0):
    M = index_map.shape[0]
    K = a.shape[1]
    N = b.shape[0] # 权重 [N, K]
    
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
