# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def _gguf_q4_0_native_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Native Triton GGUF Q4_0 GEMM.
    B is [N, (K/32)*18] uint8.
    Each GGML block handles 32 weights.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    
    offs_k = tl.arange(0, BLOCK_K)
    
    # A is standard FP16 [M, K]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    # B is packed Q4_0 [N, (K/32)*18]
    # We load in groups of GGML blocks.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        
        # 1. Load A [BLOCK_M, BLOCK_K]
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)

        # 2. Load and Dequantize B [BLOCK_N, BLOCK_K]
        # Physical index in GGUF: (k_start // 32) * 18
        num_ggml_blocks = BLOCK_K // 32
        b_block_ptr = b_ptr + offs_bn[:, None] * stride_bn + (k * num_ggml_blocks * 18)
        
        # We'll dequantize into a local shared buffer (register-based)
        b_dequant = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float16)
        
        for g in range(num_ggml_blocks):
            g_ptr = b_block_ptr + g * 18
            # Load Delta (fp16)
            delta = tl.load(tl.cast(g_ptr, tl.pointer_type(tl.float16))).to(tl.float32)
            
            # Load 16 bytes of weights
            qs = tl.load(g_ptr + 2 + tl.arange(0, 16)) # [16] uint8
            
            # Unpack to registers
            v_low = (qs & 0xF).to(tl.float32) - 8.0
            v_high = (qs >> 4).to(tl.float32) - 8.0
            
            # Map to BLOCK_K dim
            g_off = g * 32
            # Conceptually filling b_dequant
            # In optimized Triton, we do dot on these register slices directly
            # For brevity in prototype, we show the fused dot logic:
            b_slice_low = (v_low * delta).to(tl.float16)
            b_slice_high = (v_high * delta).to(tl.float16)
            
            # Accumulate slices
            a_slice_low = a[:, g_off : g_off + 16]
            a_slice_high = a[:, g_off + 16 : g_off + 32]
            
            accumulator += tl.dot(a_slice_low, tl.trans(tl.broadcast_to(b_slice_low[None, :], (BLOCK_N, 16))))
            accumulator += tl.dot(a_slice_high, tl.trans(tl.broadcast_to(b_slice_high[None, :], (BLOCK_N, 16))))

        a_ptrs += BLOCK_K * stride_ak

    # Store result
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def gguf_fused_gemm(a, qweight, out=None):
    M, K = a.shape
    N = qweight.shape[0]
    if out is None: c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else: c = out
    
    # GGUF constraint: BLOCK_K must be 32 (size of one GGML block) or multiple
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    
    _gguf_q4_0_native_gemm_kernel[grid](
        a, qweight, c,
        M, N, K,
        a.stride(0), a.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )
    return c
