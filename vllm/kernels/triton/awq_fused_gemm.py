# SPDX-License-Identifier: Apache-2.0
import os
import torch
import triton
import triton.language as tl


def _env_awq_fused_gemm_dot_bf16() -> bool:
    """Prefer bf16 tl.dot when activations are bf16 (avoids fp16 upcast on ROCm/CUDA)."""
    return os.environ.get("FASTINFERENCE_AWQ_FUSED_GEMM_DOT_BF16", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _select_fused_gemm_blocks(
    m: int, n: int, _k: int
) -> tuple[int, int, int, int, int]:
    """
    Heuristic tile sizes for AWQ fused GEMM.

    Decode often uses M in {1..32}; fixed BLOCK_M=64 wastes most warps on the M axis.
    Prefill uses larger M where BLOCK_M=64 is appropriate.
    """
    raw = os.environ.get("FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_M", "").strip()
    if raw:
        try:
            block_m = max(1, int(raw))
        except ValueError:
            block_m = 64
        raw_n = os.environ.get("FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_N", "").strip()
        raw_k = os.environ.get("FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_K", "").strip()
        block_n = int(raw_n) if raw_n else 64
        block_k = int(raw_k) if raw_k else 32
        block_n = max(16, block_n)
        block_k = max(8, block_k)
        raw_w = os.environ.get("FASTINFERENCE_AWQ_FUSED_GEMM_NUM_WARPS", "").strip()
        num_warps = int(raw_w) if raw_w else 4
        num_warps = max(1, min(8, num_warps))
        raw_s = os.environ.get("FASTINFERENCE_AWQ_FUSED_GEMM_NUM_STAGES", "").strip()
        num_stages = int(raw_s) if raw_s else 2
        num_stages = max(1, min(4, num_stages))
        return block_m, block_n, block_k, num_warps, num_stages

    block_k = 32
    if m <= 4:
        block_m = 16
    elif m <= 16:
        block_m = 32
    else:
        block_m = 64

    if m <= 16 and n >= 4096:
        block_n = 128
    else:
        block_n = 64

    if m >= 64 and n >= 4096:
        num_warps = 8
    elif block_n >= 128:
        num_warps = 8
    else:
        num_warps = 4
    num_stages = 2
    return block_m, block_n, block_k, num_warps, num_stages


@triton.jit
def _awq_native_tiled_gemm_kernel(
    a_ptr, b_ptr, s_ptr, z_ptr, c_ptr,
    M, N, K, group_size,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sn, stride_sk,
    stride_zn, stride_zk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
):
    """
    Native Triton Tiling Implementation for AMD gfx1151.
    Performs K-axis splitting inside a single kernel launch to avoid TLB storm.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    
    # Static K-axis offsets
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + (offs_k[None, :] // 8) * stride_bk)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Dimensionality Reduction: The main loop IS the tiling logic
    # By controlling BLOCK_K and the number of iterations, we keep MC stable
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k[None, :] < k_remaining
        
        # 1. Load A
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)

        # 2. Load and Dequantize B
        b_packed = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & mask_k, other=0)
        b_unpacked = (b_packed >> ((offs_k[None, :] % 8) * 4)) & 0xF

        # 3. Load Scales/Zeros
        current_k = k * BLOCK_K + offs_k
        group_idx = current_k // group_size
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx[None, :] * stride_sk)
        z_ptrs = z_ptr + (offs_bn[:, None] * stride_zn + (group_idx[None, :] // 8) * stride_zk)
        
        scales = tl.load(s_ptrs, mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)), other=1.0)
        z_packed = tl.load(z_ptrs, mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)), other=0)
        zeros = (z_packed >> ((group_idx[None, :] % 8) * 4)) & 0xF

        b = (b_unpacked.to(tl.float32) - zeros.to(tl.float32)) * scales.to(tl.float32)

        # 4. Multiply-Accumulate (bf16 dot avoids extra cast when activations are bf16)
        if USE_BF16_DOT:
            accumulator += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
        else:
            accumulator += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    # Store Result
    if USE_BF16_DOT:
        c = accumulator.to(tl.bfloat16)
    else:
        c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.jit
def _packed_int4_symmetric_tiled_gemm_kernel(
    a_ptr, b_ptr, s_ptr, c_ptr,
    M, N, K, group_size,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sn, stride_sk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + (offs_k[None, :] // 8) * stride_bk)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k[None, :] < k_remaining

        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        b_packed = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & mask_k, other=0)
        b_unpacked = (b_packed >> ((offs_k[None, :] % 8) * 4)) & 0xF

        current_k = k * BLOCK_K + offs_k
        group_idx = current_k // group_size
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx[None, :] * stride_sk)
        scales = tl.load(
            s_ptrs,
            mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)),
            other=1.0,
        )

        # Symmetric int4 unpack: (q - 8) * scale
        b = (b_unpacked.to(tl.float32) - 8.0) * scales.to(tl.float32)
        if USE_BF16_DOT:
            accumulator += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
        else:
            accumulator += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    if USE_BF16_DOT:
        c = accumulator.to(tl.bfloat16)
    else:
        c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def awq_fused_gemm(a, qweight, scales, qzeros, group_size, out=None):
    M, K = a.shape
    N = qweight.shape[0]
    if out is None: c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else: c = out

    block_m, block_n, block_k, num_warps, num_stages = _select_fused_gemm_blocks(M, N, K)
    use_bf16_dot = a.dtype == torch.bfloat16 and _env_awq_fused_gemm_dot_bf16()
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)

    _awq_native_tiled_gemm_kernel[grid](
        a, qweight, scales, qzeros, c,
        M, N, K, group_size,
        a.stride(0), a.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1) if qzeros is not None else 0,
        c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        USE_BF16_DOT=use_bf16_dot,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c


def packed_int4_symmetric_fused_gemm(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    m, k = a.shape
    n = qweight.shape[0]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype) if out is None else out
    block_m, block_n, block_k, num_warps, num_stages = _select_fused_gemm_blocks(m, n, k)
    use_bf16_dot = a.dtype == torch.bfloat16 and _env_awq_fused_gemm_dot_bf16()
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _packed_int4_symmetric_tiled_gemm_kernel[grid](
        a, qweight, scales, c,
        m, n, k, group_size,
        a.stride(0), a.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        USE_BF16_DOT=use_bf16_dot,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c


def awq_fused_gemm_safe(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, bool, str]:
    if a.dim() != 2:
        return a, False, "input_not_2d"
    if qweight.dim() != 2:
        return a, False, "qweight_not_2d"
    if scales.dim() != 2:
        return a, False, "scales_not_2d"
    if qzeros is None or qzeros.dim() != 2:
        return a, False, "qzeros_bad"
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(qweight.shape[0])
    if k != int(qweight.shape[1] * 8):
        return a, False, "k_mismatch"
    if group_size <= 0 or k % int(group_size) != 0:
        return a, False, "group_mismatch"
    if a.device != qweight.device or a.device != scales.device or a.device != qzeros.device:
        return a, False, "device_mismatch"
    if a.dtype not in (torch.float16, torch.bfloat16):
        return a, False, f"unsupported_dtype_{str(a.dtype)}"
    try:
        c = awq_fused_gemm(a, qweight, scales, qzeros, group_size=group_size, out=out)
        if c.shape != (m, n):
            return c, False, "bad_output_shape"
        return c, True, "ok"
    except Exception:
        return a, False, "kernel_exception"


def packed_int4_symmetric_fused_gemm_safe(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, bool, str]:
    if a.dim() != 2:
        return a, False, "input_not_2d"
    if qweight.dim() != 2:
        return a, False, "qweight_not_2d"
    if scales.dim() != 2:
        return a, False, "scales_not_2d"
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(qweight.shape[0])
    if k != int(qweight.shape[1] * 8):
        return a, False, "k_mismatch"
    if group_size <= 0 or k % int(group_size) != 0:
        return a, False, "group_mismatch"
    if a.device != qweight.device or a.device != scales.device:
        return a, False, "device_mismatch"
    if a.dtype not in (torch.float16, torch.bfloat16):
        return a, False, f"unsupported_dtype_{str(a.dtype)}"
    try:
        c = packed_int4_symmetric_fused_gemm(a, qweight, scales, group_size=group_size, out=out)
        if c.shape != (m, n):
            return c, False, "bad_output_shape"
        return c, True, "ok"
    except Exception:
        return a, False, "kernel_exception"
