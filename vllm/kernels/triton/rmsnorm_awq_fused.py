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
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a_val = tl.load(row_a_ptr + k_offs[None, :] * stride_ak, mask=k_offs[None, :] < K, other=0.0).to(tl.float32)
        var += tl.sum(a_val * a_val, axis=1)
    rrms = tl.math.rsqrt(var / K + eps)
    
    # 2. GEMM Loop
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        current_k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)
        # Load A and apply norm
        a = tl.load(row_a_ptr + current_k_offs[None, :] * stride_ak, mask=current_k_offs[None, :] < K, other=0.0)
        nw = tl.load(nw_ptr + current_k_offs, mask=current_k_offs < K, other=1.0)
        a_normed = (a.to(tl.float32) * rrms[:, None]) * nw[None, :].to(tl.float32)

        # Load and dequant B [N, K // 8]
        b_packed = tl.load(b_ptr + (offs_bn[:, None] * stride_bn + (current_k_offs[None, :] // 8) * stride_bk),
                           mask=(offs_bn[:, None] < N) & (current_k_offs[None, :] < K), other=0)
        shift = (current_k_offs[None, :] % 8) * 4
        b_unpacked = (b_packed >> shift) & 0xF
        
        g_idx = current_k_offs // group_size
        scales = tl.load(s_ptr + (offs_bn[:, None] * stride_sn + g_idx[None, :] * stride_sk), 
                         mask=(offs_bn[:, None] < N) & (g_idx[None, :] < tl.cdiv(K, group_size)), other=1.0)
        z_packed = tl.load(z_ptr + (offs_bn[:, None] * stride_zn + (g_idx[None, :] // 8) * stride_zk),
                           mask=(offs_bn[:, None] < N) & (g_idx[None, :] < tl.cdiv(K, group_size)), other=0)
        zeros = (z_packed >> ((g_idx[None, :] % 8) * 4)) & 0xF

        b = (b_unpacked.to(tl.float32) - zeros.to(tl.float32)) * scales.to(tl.float32)
        accumulator += tl.dot(a_normed.to(tl.float16), tl.trans(b.to(tl.float16)))

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

def rmsnorm_awq_fused_linear(x, qweight, scales, qzeros, group_size, norm_w, eps=1e-6):
    M, K = x.shape
    N = qweight.shape[0]
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    
    rmsnorm_awq_fused_kernel[grid](
        x, qweight, scales, qzeros, norm_w, c,
        M, N, K, group_size, eps,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        norm_w.stride(0),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
    )
    return c
