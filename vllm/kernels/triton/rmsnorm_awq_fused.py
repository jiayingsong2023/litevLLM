# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_awq_fused_kernel(
    a_ptr, b_ptr, s_ptr, z_ptr, nw_ptr, c_ptr,
    M, N, K, group_size, eps,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sn, stride_sk,
    stride_zn, stride_zk,
    stride_norm,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    
    # 1. Compute RMSNorm scaling
    row_a_ptr = a_ptr + offs_am[:, None] * stride_am
    var = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        a_row = tl.load(row_a_ptr + k_offs[None, :] * stride_ak, mask=mask[None, :], other=0.0).to(tl.float32)
        var += tl.sum(a_row * a_row, axis=1)
    rrms = tl.math.rsqrt(var / K + eps)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pre-compute pointers for B (Weights)
    # B is [N, K // 8]
    b_base_ptr = b_ptr + offs_bn[:, None] * stride_bn
    s_base_ptr = s_ptr + offs_bn[:, None] * stride_sn
    z_base_ptr = z_ptr + offs_bn[:, None] * stride_zn

    for k in range(0, K, BLOCK_K):
        current_k_offs = k + offs_k
        mask_k = current_k_offs < K
        
        # Load A and apply norm
        a = tl.load(row_a_ptr + current_k_offs[None, :] * stride_ak, mask=mask_k[None, :], other=0.0).to(tl.float32)
        nw = tl.load(nw_ptr + current_k_offs, mask=mask_k, other=1.0).to(tl.float32)
        a_normed = (a * rrms[:, None]) * nw[None, :]

        # Load Weights [BLOCK_N, BLOCK_K // 8]
        # Important: Match awq_dequantize_triton exactly
        b_packed = tl.load(b_base_ptr + (current_k_offs[None, :] // 8) * stride_bk,
                           mask=(offs_bn[:, None] < N) & (mask_k[None, :]), other=0)
        
        # AWQ Packing logic: typically (k % 8) * 4
        shift = (current_k_offs[None, :] % 8) * 4
        b_unpacked = (b_packed >> shift) & 0xF
        
        # Load Scales/Zeros
        g_idx = current_k_offs // group_size
        scales = tl.load(s_base_ptr + g_idx[None, :] * stride_sk, 
                         mask=(offs_bn[:, None] < N) & (mask_k[None, :]), other=1.0)
        z_packed = tl.load(z_base_ptr + (g_idx[None, :] // 8) * stride_zk,
                           mask=(offs_bn[:, None] < N) & (mask_k[None, :]), other=0)
        
        zeros = (z_packed >> ((g_idx[None, :] % 8) * 4)) & 0xF

        b = (b_unpacked.to(tl.float32) - zeros.to(tl.float32)) * scales.to(tl.float32)
        
        # Dot product: A[M, K] @ B.T[K, N]
        accumulator += tl.dot(a_normed.to(tl.float16), tl.trans(b.to(tl.float16)))

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

def rmsnorm_awq_fused_linear(x, qweight, scales, qzeros, group_size, norm_w, eps=1e-6):
    M, K = x.shape
    N = qweight.shape[0]
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    
    rmsnorm_awq_fused_kernel[grid](
        a_ptr=x, b_ptr=qweight, s_ptr=scales, z_ptr=qzeros, nw_ptr=norm_w, c_ptr=c,
        M=M, N=N, K=K, group_size=group_size, eps=eps,
        stride_am=x.stride(0), stride_ak=x.stride(1),
        stride_bn=qweight.stride(0), stride_bk=qweight.stride(1),
        stride_sn=scales.stride(0), stride_sk=scales.stride(1),
        stride_zn=qzeros.stride(0), stride_zk=qzeros.stride(1),
        stride_norm=norm_w.stride(0),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )
    return c
