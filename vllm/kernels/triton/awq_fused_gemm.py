# SPDX-License-Identifier: Apache-2.0
import os
import torch
import triton
import triton.language as tl


def _resolve_use_bf16_dot(a: torch.Tensor, m: int, n: int) -> bool:
    """
    Whether to use bf16 operands in tl.dot (vs fp16).

    On ROCm, bf16 dot can be slower than fp16 dot for narrow decode (small M, moderate N);
    microbench: use fp16 dot there but still write bf16 output (USE_BF16_OUTPUT).

    FASTINFERENCE_AWQ_FUSED_GEMM_DOT_BF16:
      - unset or "auto": heuristic (ROCm narrow decode -> fp16 dot; else bf16 when a is bf16)
      - "1"/"true"/"on": always bf16 dot when a is bf16
      - "0"/"false"/"off": always fp16 dot when a is bf16
    """
    if a.dtype != torch.bfloat16:
        return False
    raw = os.environ.get("FASTINFERENCE_AWQ_FUSED_GEMM_DOT_BF16", "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    # auto
    if getattr(torch.version, "hip", None) is not None:
        if m <= 4 and n <= 8192:
            return False
    return True


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

    block_k = 64
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


def _env_fused_gemm_autotune() -> bool:
    """Bench candidate tile shapes at first launch per (M,N,K,dot) key; disable to use heuristics only."""
    return os.environ.get("FASTINFERENCE_AWQ_FUSED_AUTOTUNE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# Packed int4 + AWQ native: tuned configs for ROCm (gfx1151-class) and CUDA; autotune picks best per key.
_PACKED_FUSED_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
]


@triton.jit
def _awq_native_tiled_gemm_split_k(
    a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, bias_ptr,
    M, N, K, group_size,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sn, stride_sk,
    stride_zn, stride_zk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Split-K AWQ GEMM Kernel for small M.
    Parallelizes reduction over K dimension to increase GPU utilization.
    """
    pid = tl.program_id(0)
    pid_sk = tl.program_id(1)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    
    # Split K dimension
    iters_per_sk = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT_K)
    start_k = pid_sk * iters_per_sk * BLOCK_K
    end_k = min(K, (pid_sk + 1) * iters_per_sk * BLOCK_K)
    
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + (start_k + offs_k[None, :]) * stride_ak)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + ((start_k + offs_k[None, :]) // 8) * stride_bk)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, iters_per_sk):
        curr_k_base = start_k + k * BLOCK_K
        k_remaining = K - curr_k_base
        mask_k = (offs_k[None, :] < BLOCK_K) & (curr_k_base + offs_k[None, :] < K)
        mask_k = mask_k & (curr_k_base < end_k)
        
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        b_packed = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & mask_k, other=0)
        b_unpacked = (b_packed >> ((offs_k[None, :] % 8) * 4)) & 0xF

        current_k = curr_k_base + offs_k
        group_idx = current_k // group_size
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx[None, :] * stride_sk)
        z_ptrs = z_ptr + (offs_bn[:, None] * stride_zn + (group_idx[None, :] // 8) * stride_zk)
        
        scales = tl.load(s_ptrs, mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)), other=1.0)
        z_packed = tl.load(z_ptrs, mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)), other=0)
        zeros = (z_packed >> ((group_idx[None, :] % 8) * 4)) & 0xF

        b = (b_unpacked.to(tl.float32) - zeros.to(tl.float32)) * scales.to(tl.float32)

        if USE_BF16_DOT:
            accumulator += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
        else:
            accumulator += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    if USE_BF16_OUTPUT:
        result = accumulator.to(tl.bfloat16)
    else:
        result = accumulator.to(tl.float16)
        
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Use atomic add if SPLIT_K > 1
    if SPLIT_K > 1:
        tl.atomic_add(c_ptrs, result, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
    else:
        if HAS_BIAS:
            bias_vals = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
            accumulator = accumulator + bias_vals.to(tl.float32)[None, :]
            if USE_BF16_OUTPUT: result = accumulator.to(tl.bfloat16)
            else: result = accumulator.to(tl.float16)
        tl.store(c_ptrs, result, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.jit
def _awq_native_tiled_gemm_heuristic(
    a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, bias_ptr,
    M, N, K, group_size,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sn, stride_sk,
    stride_zn, stride_zk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
        accumulator = accumulator + bias_vals.to(tl.float32)[None, :]
    # Store Result (output dtype may be bf16 even when tl.dot used fp16 operands)
    if USE_BF16_OUTPUT:
        c = accumulator.to(tl.bfloat16)
    else:
        c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.jit
def _packed_int4_symmetric_tiled_gemm_split_k(
    a_ptr, b_ptr, s_ptr, c_ptr, bias_ptr,
    M, N, K, group_size,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sn, stride_sk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_sk = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    
    iters_per_sk = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT_K)
    start_k = pid_sk * iters_per_sk * BLOCK_K
    end_k = min(K, (pid_sk + 1) * iters_per_sk * BLOCK_K)
    
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + (start_k + offs_k[None, :]) * stride_ak)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + ((start_k + offs_k[None, :]) // 8) * stride_bk)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, iters_per_sk):
        curr_k_base = start_k + k * BLOCK_K
        k_remaining = K - curr_k_base
        mask_k = (offs_k[None, :] < BLOCK_K) & (curr_k_base + offs_k[None, :] < K)
        mask_k = mask_k & (curr_k_base < end_k)

        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        b_packed = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & mask_k, other=0)
        b_unpacked = (b_packed >> ((offs_k[None, :] % 8) * 4)) & 0xF

        current_k = curr_k_base + offs_k
        group_idx = current_k // group_size
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx[None, :] * stride_sk)
        scales = tl.load(
            s_ptrs,
            mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)),
            other=1.0,
        )

        b = (b_unpacked.to(tl.float32) - 8.0) * scales.to(tl.float32)
        if USE_BF16_DOT:
            accumulator += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
        else:
            accumulator += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    if USE_BF16_OUTPUT:
        result = accumulator.to(tl.bfloat16)
    else:
        result = accumulator.to(tl.float16)
        
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    if SPLIT_K > 1:
        tl.atomic_add(c_ptrs, result, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
    else:
        if HAS_BIAS:
            bias_vals = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
            accumulator = accumulator + bias_vals.to(tl.float32)[None, :]
            if USE_BF16_OUTPUT: result = accumulator.to(tl.bfloat16)
            else: result = accumulator.to(tl.float16)
        tl.store(c_ptrs, result, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.jit
def _packed_int4_symmetric_tiled_gemm_heuristic(
    a_ptr, b_ptr, s_ptr, c_ptr, bias_ptr,
    M, N, K, group_size,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sn, stride_sk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Optimized Symmetric INT4 GEMM. 
    Reduces Scale load frequency and uses vectorized unpacking.
    """
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

    # Heuristic: if BLOCK_K is a multiple of group_size (or vice versa), we can optimize scale loads.
    # Most common: BLOCK_K=64, group_size=128. Scale stays same for 2 iterations.
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k[None, :] < k_remaining

        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        b_packed = tl.load(b_ptrs, mask=(offs_bn[:, None] < N) & mask_k, other=0)
        
        # Fast 4-bit unpacking using bitwise shifts
        b_unpacked = (b_packed >> ((offs_k[None, :] % 8) * 4)) & 0xF

        # Scale loading: check if we are at a group boundary
        # For simplicity and performance in Triton, we load once per K-block
        # but ensure group_idx is calculated correctly for the first element of offs_k.
        current_k_base = k * BLOCK_K
        group_idx = current_k_base // group_size
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx * stride_sk)
        scales = tl.load(s_ptrs, mask=offs_bn[:, None] < N, other=1.0)

        # (q - 8) * scale
        b = (b_unpacked.to(tl.float32) - 8.0) * scales.to(tl.float32)
        
        if USE_BF16_DOT:
            accumulator += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
        else:
            accumulator += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
        accumulator = accumulator + bias_vals.to(tl.float32)[None, :]
    
    if USE_BF16_OUTPUT:
        c = accumulator.to(tl.bfloat16)
    else:
        c = accumulator.to(tl.float16)
        
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.autotune(
    configs=_PACKED_FUSED_AUTOTUNE_CONFIGS,
    key=["M", "N", "K", "BF16_DOT", "OUT_BF16", "HAS_BIAS"],
    warmup=8,
    rep=20,
    cache_results=True,
)
@triton.jit
def _packed_int4_symmetric_tiled_gemm_autotuned(
    a_ptr, b_ptr, s_ptr, c_ptr, bias_ptr,
    M, N, K, group_size,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sn, stride_sk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BF16_DOT: tl.constexpr,
    OUT_BF16: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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
        b_unpacked = (b_packed >> ((offs_k[None, :] % 8) * 4)) & 0x0F

        current_k = k * BLOCK_K + offs_k
        group_idx = current_k // group_size
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx[None, :] * stride_sk)
        scales = tl.load(
            s_ptrs,
            mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)),
            other=1.0,
        )

        b = (b_unpacked.to(tl.float32) - 8.0) * scales.to(tl.float32)
        if BF16_DOT:
            accumulator += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
        else:
            accumulator += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
        accumulator = accumulator + bias_vals.to(tl.float32)[None, :]
    if OUT_BF16:
        c = accumulator.to(tl.bfloat16)
    else:
        c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.autotune(
    configs=_PACKED_FUSED_AUTOTUNE_CONFIGS,
    key=["M", "N", "K", "BF16_DOT", "OUT_BF16", "HAS_BIAS"],
    warmup=8,
    rep=20,
    cache_results=True,
)
@triton.jit
def _awq_native_tiled_gemm_autotuned(
    a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, bias_ptr,
    M, N, K, group_size,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sn, stride_sk,
    stride_zn, stride_zk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BF16_DOT: tl.constexpr,
    OUT_BF16: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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
        b_unpacked = (b_packed >> ((offs_k[None, :] % 8) * 4)) & 0x0F

        current_k = k * BLOCK_K + offs_k
        group_idx = current_k // group_size
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx[None, :] * stride_sk)
        z_ptrs = z_ptr + (offs_bn[:, None] * stride_zn + (group_idx[None, :] // 8) * stride_zk)

        scales = tl.load(s_ptrs, mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)), other=1.0)
        z_packed = tl.load(z_ptrs, mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)), other=0)
        zeros = (z_packed >> ((group_idx[None, :] % 8) * 4)) & 0x0F

        b = (b_unpacked.to(tl.float32) - zeros.to(tl.float32)) * scales.to(tl.float32)

        if BF16_DOT:
            accumulator += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
        else:
            accumulator += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
        accumulator = accumulator + bias_vals.to(tl.float32)[None, :]
    if OUT_BF16:
        c = accumulator.to(tl.bfloat16)
    else:
        c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def awq_fused_gemm(a, qweight, scales, qzeros, group_size, out=None, bias=None):
    M, K = a.shape
    N = qweight.shape[0]
    
    has_bias = bias is not None
    if has_bias:
        if bias.dim() != 1 or int(bias.shape[0]) != N:
            raise ValueError(
                f"awq_fused_gemm: bias must be 1D of size N={N}, got {tuple(bias.shape)}"
            )
        bias = bias.contiguous().reshape(N)
        if bias.device != a.device:
            raise ValueError("awq_fused_gemm: bias must be on the same device as activations")
        if bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"awq_fused_gemm: unsupported bias dtype {bias.dtype}")
    bias_ptr_arg = bias if has_bias else a

    use_bf16_dot = _resolve_use_bf16_dot(a, M, N)
    use_bf16_output = a.dtype == torch.bfloat16
    bf16_dot = 1 if use_bf16_dot else 0
    out_bf16 = 1 if use_bf16_output else 0

    # Decision for Split-K: Only for extremely narrow M and deep K
    split_k = 1
    if M == 1 and K >= 8192:
        split_k = 4

    if out is None:
        if split_k > 1:
            # Atomic add requires zeros
            c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
        else:
            c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out
        if split_k > 1:
            c.zero_()

    if _env_fused_gemm_autotune() and split_k == 1:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
        try:
            _awq_native_tiled_gemm_autotuned[grid](
                a,
                qweight,
                scales,
                qzeros,
                c,
                bias_ptr_arg,
                M,
                N,
                K,
                group_size,
                a.stride(0),
                a.stride(1),
                qweight.stride(0),
                qweight.stride(1),
                scales.stride(0),
                scales.stride(1),
                qzeros.stride(0),
                qzeros.stride(1) if qzeros is not None else 0,
                c.stride(0),
                c.stride(1),
                BF16_DOT=bf16_dot,
                OUT_BF16=out_bf16,
                HAS_BIAS=has_bias,
            )
            return c
        except Exception:
            pass

    block_m, block_n, block_k, num_warps, num_stages = _select_fused_gemm_blocks(M, N, K)
    
    if split_k > 1:
        grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), split_k)
        _awq_native_tiled_gemm_split_k[grid](
            a,
            qweight,
            scales,
            qzeros,
            c,
            bias_ptr_arg,
            M,
            N,
            K,
            group_size,
            a.stride(0),
            a.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            scales.stride(0),
            scales.stride(1),
            qzeros.stride(0),
            qzeros.stride(1) if qzeros is not None else 0,
            c.stride(0),
            c.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SPLIT_K=split_k,
            USE_BF16_DOT=use_bf16_dot,
            USE_BF16_OUTPUT=use_bf16_output,
            HAS_BIAS=has_bias,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
        _awq_native_tiled_gemm_heuristic[grid](
            a,
            qweight,
            scales,
            qzeros,
            c,
            bias_ptr_arg,
            M,
            N,
            K,
            group_size,
            a.stride(0),
            a.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            scales.stride(0),
            scales.stride(1),
            qzeros.stride(0),
            qzeros.stride(1) if qzeros is not None else 0,
            c.stride(0),
            c.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            USE_BF16_DOT=use_bf16_dot,
            USE_BF16_OUTPUT=use_bf16_output,
            HAS_BIAS=has_bias,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return c


def packed_int4_symmetric_fused_gemm(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    m, k = a.shape
    n = qweight.shape[0]
    
    has_bias = bias is not None
    if has_bias:
        if bias.dim() != 1 or int(bias.shape[0]) != n:
            raise ValueError(
                f"packed_int4_symmetric_fused_gemm: bias must be 1D of size N={n}, got {tuple(bias.shape)}"
            )
        bias = bias.contiguous().reshape(n)
        if bias.device != a.device:
            raise ValueError(
                "packed_int4_symmetric_fused_gemm: bias must be on the same device as activations"
            )
        if bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"packed_int4_symmetric_fused_gemm: unsupported bias dtype {bias.dtype}")
    bias_ptr_arg = bias if has_bias else a

    use_bf16_dot = _resolve_use_bf16_dot(a, m, n)
    use_bf16_output = a.dtype == torch.bfloat16
    bf16_dot = 1 if use_bf16_dot else 0
    out_bf16 = 1 if use_bf16_output else 0

    split_k = 1
    if m == 1 and k >= 8192:
        split_k = 4

    if out is None:
        if split_k > 1:
            c = torch.zeros((m, n), device=a.device, dtype=a.dtype)
        else:
            c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    else:
        c = out
        if split_k > 1:
            c.zero_()

    if _env_fused_gemm_autotune() and split_k == 1:
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
        try:
            _packed_int4_symmetric_tiled_gemm_autotuned[grid](
                a,
                qweight,
                scales,
                c,
                bias_ptr_arg,
                m,
                n,
                k,
                group_size,
                a.stride(0),
                a.stride(1),
                qweight.stride(0),
                qweight.stride(1),
                scales.stride(0),
                scales.stride(1),
                c.stride(0),
                c.stride(1),
                BF16_DOT=bf16_dot,
                OUT_BF16=out_bf16,
                HAS_BIAS=has_bias,
            )
            return c
        except Exception:
            pass

    block_m, block_n, block_k, num_warps, num_stages = _select_fused_gemm_blocks(m, n, k)
    
    if split_k > 1:
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), split_k)
        _packed_int4_symmetric_tiled_gemm_split_k[grid](
            a,
            qweight,
            scales,
            c,
            bias_ptr_arg,
            m,
            n,
            k,
            group_size,
            a.stride(0),
            a.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            scales.stride(0),
            scales.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SPLIT_K=split_k,
            USE_BF16_DOT=use_bf16_dot,
            USE_BF16_OUTPUT=use_bf16_output,
            HAS_BIAS=has_bias,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        _packed_int4_symmetric_tiled_gemm_heuristic[grid](
            a,
            qweight,
            scales,
            c,
            bias_ptr_arg,
            m,
            n,
            k,
            group_size,
            a.stride(0),
            a.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            scales.stride(0),
            scales.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            USE_BF16_DOT=use_bf16_dot,
            USE_BF16_OUTPUT=use_bf16_output,
            HAS_BIAS=has_bias,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return c


def awq_fused_gemm_safe(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
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
    if bias is not None:
        if bias.dim() != 1 or int(bias.numel()) != n:
            return a, False, "bias_bad_shape"
        if bias.device != a.device:
            return a, False, "bias_device_mismatch"
        if bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return a, False, f"unsupported_bias_dtype_{str(bias.dtype)}"
    a = a.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    qzeros = qzeros.contiguous()
    bias_arg = bias.contiguous() if bias is not None else None
    try:
        c = awq_fused_gemm(
            a,
            qweight,
            scales,
            qzeros,
            group_size=group_size,
            out=out,
            bias=bias_arg,
        )
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
    bias: torch.Tensor | None = None,
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
    if bias is not None:
        if bias.dim() != 1 or int(bias.numel()) != n:
            return a, False, "bias_bad_shape"
        if bias.device != a.device:
            return a, False, "bias_device_mismatch"
        if bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return a, False, f"unsupported_bias_dtype_{str(bias.dtype)}"
    a = a.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    bias_arg = bias.contiguous() if bias is not None else None
    try:
        c = packed_int4_symmetric_fused_gemm(
            a,
            qweight,
            scales,
            group_size=group_size,
            out=out,
            bias=bias_arg,
        )
        if c.shape != (m, n):
            return c, False, "bad_output_shape"
        return c, True, "ok"
    except Exception:
        return a, False, "kernel_exception"
