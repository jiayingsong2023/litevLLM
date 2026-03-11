# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def awq_fused_gemm_kernel(
    a_ptr, b_ptr, s_ptr, z_ptr, c_ptr,
    M, N, K, group_size,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sn, stride_sk,
    stride_zn, stride_zk,
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
    offs_k = tl.arange(0, BLOCK_K)

    # A: [M, K]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B: [N, K // 8]
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + (offs_k[None, :] // 8) * stride_bk)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        
        # Load A [BLOCK_M, BLOCK_K]
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)

        # Load B Packed [BLOCK_N, BLOCK_K // 8]
        b_packed = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & (offs_k[None, :] < k_remaining), other=0)

        # Dequantize B
        shift = (offs_k[None, :] % 8) * 4
        b_unpacked = (b_packed >> shift) & 0xF

        # Load Scales/Zeros [BLOCK_N, BLOCK_K]
        current_k = k * BLOCK_K + offs_k
        group_idx = current_k // group_size
        
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx[None, :] * stride_sk)
        z_ptrs = z_ptr + (offs_bn[:, None] * stride_zn + (group_idx[None, :] // 8) * stride_zk)
        
        scales = tl.load(s_ptrs, mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)), other=1.0)
        z_packed = tl.load(z_ptrs, mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)), other=0)
        
        zeros = (z_packed >> ((group_idx[None, :] % 8) * 4)) & 0xF

        b = (b_unpacked.to(tl.float32) - zeros.to(tl.float32)) * scales.to(tl.float32)

        accumulator += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def awq_fused_gemm(a, qweight, scales, qzeros, group_size, out=None):
    M, K = a.shape
    N = qweight.shape[0]
    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    awq_fused_gemm_kernel[grid](
        a_ptr=a, b_ptr=qweight, s_ptr=scales, z_ptr=qzeros, c_ptr=c,
        M=M, N=N, K=K, group_size=group_size,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bn=qweight.stride(0), stride_bk=qweight.stride(1),
        stride_sn=scales.stride(0), stride_sk=scales.stride(1),
        stride_zn=qzeros.stride(0), stride_zk=qzeros.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )
    return c
