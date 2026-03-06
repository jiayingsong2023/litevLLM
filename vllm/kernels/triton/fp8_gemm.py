# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def _fp8_block_gemm_kernel(
    A, B, C,
    Scale_A, Scale_B, # Block-wise scales
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_sam, stride_sak,
    stride_sbk, stride_sbn,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A + (off_m[:, None] * stride_am + off_k[None, :] * stride_ak)
    b_ptrs = B + (off_k[:, None] * stride_bk + off_n[None, :] * stride_bn)
    
    # Scale pointers (block-wise)
    # DeepSeek scales are usually [M//128, K//128] and [N//128, K//128]
    # For simplicity, we assume scales are aligned with BLOCK_SIZE
    sa_ptrs = Scale_A + (pid_m * stride_sam)
    sb_ptrs = Scale_B + (pid_n * stride_sbn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = (k * BLOCK_SIZE_K + off_k) < K
        a = tl.load(a_ptrs, mask=(off_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (off_n[None, :] < N), other=0.0)
        
        # Load scales for this K-block
        # In a real block-wise kernel, scales would change with k
        # For this version, we assume per-row/per-col scaling or simple block scaling
        s_a = tl.load(sa_ptrs + k * stride_sak)
        s_b = tl.load(sb_ptrs + k * stride_sbk)
        
        # Explicit cast to float16 for stability if native fp8 dot is flaky
        accumulator += tl.dot(a.to(tl.float16), b.to(tl.float16)) * (s_a * s_b)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    res = accumulator.to(C.dtype.element_ty)
    c_ptrs = C + (off_m[:, None] * stride_cm + off_n[None, :] * stride_cn)
    tl.store(c_ptrs, res, mask=(off_m[:, None] < M) & (off_n[None, :] < N))

def fp8_block_gemm(a, b, scale_a, scale_b, out=None):
    M, K = a.shape
    K_b, N = b.shape
    
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.float16)
        
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _fp8_block_gemm_kernel[grid](
        a, b, out,
        scale_a, scale_b,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        scale_a.stride(0), scale_a.stride(1) if scale_a.dim() > 1 else 0,
        scale_b.stride(0), scale_b.stride(1) if scale_b.dim() > 1 else 0,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=8
    )
    return out
