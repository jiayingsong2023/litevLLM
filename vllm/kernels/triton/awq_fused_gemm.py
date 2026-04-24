# SPDX-License-Identifier: Apache-2.0
import os
import json
from pathlib import Path
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

    k = int(_k)
    # Gemma4-31B MLP down_proj decode hot shape:
    #   n=5376, k=21504, m in {1,2}
    # This shape is latency-sensitive and generally prefers lower warp/stage pressure
    # compared with the broader decode defaults.
    mlp_down_decode_small_m = n == 5376 and k == 21504 and m <= 2
    block_k = 64
    deep_k_narrow_out = k >= 16384 and n <= 6144
    if m == 1:
        # M=1 decode: route to the GEMV specialization inside the kernel. A
        # BLOCK_M of 16 would waste 15/16 MFMA lanes on replicated rows; the
        # GEMV branch needs BLOCK_M==1 to be picked up (constexpr DCE of the
        # MMA path). Wider N tiles keep the reduction dominated by memory.
        block_m = 1
        if mlp_down_decode_small_m:
            block_n = 128
        elif n <= 8192 or k >= 16384:
            block_n = 256
        else:
            block_n = 128
    elif m <= 4:
        block_m = 16
        # Decode-heavy Gemma4 shapes prefer wider N tiles, but very wide output
        # projections (e.g. global q / gate-up) still favor 128.
        if mlp_down_decode_small_m:
            block_n = 128
        elif deep_k_narrow_out and m >= 4:
            block_n = 128
        elif n <= 8192 or (m == 1 and k >= 16384):
            block_n = 256
        else:
            block_n = 128
    elif m <= 32:
        block_m = 32
        if deep_k_narrow_out:
            block_n = 128
        else:
            block_n = 256 if n >= 4096 else 128
    elif m <= 192:
        # Gemma4 prefill-like batches consistently benefit from shorter M tiles
        # with a wider N tile on packed-int4 fused GEMM.
        if deep_k_narrow_out:
            block_m = 64
            block_n = 64
        else:
            block_m = 32
            block_n = 256 if n >= 4096 else 128
    else:
        block_m = 64
        # M=256 prefill-like shapes on Gemma4-31B favor square-ish N tiles.
        block_n = 64

    if mlp_down_decode_small_m:
        # m in {1,2} decode path: reduce occupancy pressure to improve single-request
        # token latency and stabilize C1 decode TPS.
        num_warps = 4
    elif block_n >= 128 or (m >= 64 and n >= 4096):
        num_warps = 8
    else:
        num_warps = 4
    num_stages = 1 if mlp_down_decode_small_m else 2
    return block_m, block_n, block_k, num_warps, num_stages


_PERSISTENT_PROFILE_CACHE: dict[str, list[dict[str, int]]] | None = None


def _persistent_profile_path() -> Path:
    raw = os.environ.get("FASTINFERENCE_AWQ_FUSED_PROFILE_JSON", "").strip()
    if not raw:
        return Path()
    return Path(raw).expanduser()


def _load_persistent_profile() -> dict[str, list[dict[str, int]]]:
    global _PERSISTENT_PROFILE_CACHE
    if _PERSISTENT_PROFILE_CACHE is not None:
        return _PERSISTENT_PROFILE_CACHE
    path = _persistent_profile_path()
    if not str(path) or (not path.is_file()):
        _PERSISTENT_PROFILE_CACHE = {}
        return _PERSISTENT_PROFILE_CACHE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _PERSISTENT_PROFILE_CACHE = {}
        return _PERSISTENT_PROFILE_CACHE
    out: dict[str, list[dict[str, int]]] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "version":
                continue
            if isinstance(value, list):
                rows: list[dict[str, int]] = []
                for row in value:
                    if isinstance(row, dict):
                        rows.append({str(k): int(v) for k, v in row.items() if isinstance(v, (int, float))})
                out[str(key)] = rows
    _PERSISTENT_PROFILE_CACHE = out
    return _PERSISTENT_PROFILE_CACHE


def _match_profile_entry(
    row: dict[str, int],
    *,
    m: int,
    n: int,
    k: int,
    group_size: int,
) -> bool:
    m_min = int(row.get("m_min", 1))
    m_max = int(row.get("m_max", 1 << 30))
    if m < m_min or m > m_max:
        return False
    if "n" in row and int(row["n"]) != n:
        return False
    if "k" in row and int(row["k"]) != k:
        return False
    if "group_size" in row and int(row["group_size"]) != group_size:
        return False
    return True


def _lookup_persistent_blocks(
    kind: str,
    *,
    m: int,
    n: int,
    k: int,
    group_size: int,
) -> tuple[int, int, int, int, int] | None:
    rows = _load_persistent_profile().get(kind, [])
    for row in rows:
        if not _match_profile_entry(row, m=m, n=n, k=k, group_size=group_size):
            continue
        keys = ("block_m", "block_n", "block_k", "num_warps", "num_stages")
        if all(key in row for key in keys):
            return (
                int(row["block_m"]),
                int(row["block_n"]),
                int(row["block_k"]),
                int(row["num_warps"]),
                int(row["num_stages"]),
            )
    return None


def _env_fused_gemm_autotune(m: int, n: int, k: int) -> bool:
    """
    Whether to launch Triton autotune for fused GEMM.

    FASTINFERENCE_AWQ_FUSED_AUTOTUNE:
      - "1"/"true"/"on": always autotune (except M==1; see below)
      - "0"/"false"/"off": never autotune
      - unset or "auto": shape-aware default

    The "auto" mode skips autotune on Gemma4-like production shapes where our
    curated heuristics are consistently faster and avoid warmup overhead.

    M==1 is **never** autotuned: ``_PACKED_FUSED_AUTOTUNE_CONFIGS`` only lists
    ``BLOCK_M >= 16``, so autotune would always pick an MMA tile that skips the
    ``BLOCK_M == 1`` GEMV fast path inside the heuristic/split-k kernels. Even
    when operators force ``FASTINFERENCE_AWQ_FUSED_AUTOTUNE=1``, single-row
    decode must stay on the deterministic heuristic launcher.
    """
    mm, nn, kk = int(m), int(n), int(k)
    if mm == 1:
        return False
    raw = os.environ.get("FASTINFERENCE_AWQ_FUSED_AUTOTUNE", "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    # auto
    # Gemma4-31B decode dominant projections (m in {1,2,4}) benefit a lot from
    # autotuned tiles after the scale-index correctness fix; heuristic kernels
    # are kept for medium-M wide-output shapes to avoid unnecessary tuning cost.
    if mm <= 8 and nn >= 4096 and kk >= 4096:
        return True
    if mm <= 192 and nn >= 4096 and kk >= 4096:
        return False
    return True


def _select_split_k(m: int, n: int, k: int) -> int:
    """
    Split-K policy for fused int4 GEMM.

    FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K:
      - integer >= 1: force value
      - unset/"auto": shape-aware default
    """
    raw = os.environ.get("FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K", "auto").strip().lower()
    if raw not in ("", "auto"):
        try:
            return max(1, int(raw))
        except ValueError:
            return 1
    mm, nn, kk = int(m), int(n), int(k)
    # Decode m=1 with deep-K benefits from K-splitting when there is not
    # enough work along N to saturate CU/SM occupancy.
    #
    # Shape families this heuristic covers (decode M=1):
    #   * Gemma4-31B / Qwen3.5-like gate_up fused pair: K~hidden (<=8k),
    #     N~2*intermediate (>=16k)   -> split_k=1 (lots of N-parallelism).
    #   * Gemma4-31B down_proj:     K~21504, N~hidden (~5k-6k) -> split_k=4.
    #     Narrow-N means few (pid_m, pid_n) tiles, so atomics over SPLIT_K
    #     are amortized by the extra CU fill.
    #   * Qwen3.5-9B style wide-N deep-K (K~16k, N>=8k)       -> split_k=4.
    #
    # The narrow-N lane uses a moderately higher K threshold (>= 20k) so
    # shallower-K decodes on Qwen3.5 (K~18944, N~3584) stay on split_k=1
    # where atomic overhead would otherwise dominate.
    if mm == 1 and kk >= 16384 and nn >= 8192:
        return 4
    if mm == 1 and kk >= 20480 and nn >= 4096:
        return 4
    return 1


# Packed int4 + AWQ native: tuned configs for ROCm (gfx1151-class) and CUDA; autotune picks best per key.
_PACKED_FUSED_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    # Gemma4-31B packed-int4 winners on ROCm/gfx11-class workloads.
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
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

        # Same M=1 GEMV rationale as ``_packed_int4_symmetric_tiled_gemm_split_k``:
        # MFMA tiles with BLOCK_M>1 waste lanes when M==1 at runtime.
        if BLOCK_M == 1:
            prod = a.to(tl.float32) * b
            accumulator += tl.sum(prod, axis=1)[None, :]
        else:
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

        # 4. Multiply-Accumulate: M=1 GEMV path matches packed-int4 kernels.
        if BLOCK_M == 1:
            prod = a.to(tl.float32) * b
            accumulator += tl.sum(prod, axis=1)[None, :]
        else:
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
        # ---- M=1 GEMV specialization -----------------------------------
        # Decode batches with M=1 produce a [1, BLOCK_K] activation tile. The
        # generic MMA path forces BLOCK_M>=16 (MFMA lane requirement), which
        # wastes 15/16 SIMD lanes on replicated inputs. When BLOCK_M==1 we
        # broadcast multiply-add along the N axis in fp32 to keep every lane
        # productive without sacrificing accuracy.
        #
        #   a : [1,        BLOCK_K]  (activation row)
        #   b : [BLOCK_N,  BLOCK_K]  (dequantized weights, fp32)
        # a * b broadcasts to [BLOCK_N, BLOCK_K]; tl.sum over K yields [BLOCK_N].
        # ----------------------------------------------------------------
        if BLOCK_M == 1:
            prod = a.to(tl.float32) * b
            accumulator += tl.sum(prod, axis=1)[None, :]
        else:
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
    Uses per-element group index for scales so results remain correct when
    BLOCK_K spans multiple quant groups (e.g. group_size=32, BLOCK_K=64).
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

        current_k = k * BLOCK_K + offs_k
        group_idx = current_k // group_size
        s_ptrs = s_ptr + (offs_bn[:, None] * stride_sn + group_idx[None, :] * stride_sk)
        scales = tl.load(
            s_ptrs,
            mask=(offs_bn[:, None] < N) & (group_idx[None, :] < tl.cdiv(K, group_size)),
            other=1.0,
        )

        # (q - 8) * scale
        b = (b_unpacked.to(tl.float32) - 8.0) * scales.to(tl.float32)

        # See split-k kernel above for the M=1 GEMV rationale: broadcast-FMA
        # along N with fp32 accumulation keeps SIMD lanes fully utilized when
        # MFMA's 16-row tile would otherwise waste 15/16 lanes.
        if BLOCK_M == 1:
            prod = a.to(tl.float32) * b
            accumulator += tl.sum(prod, axis=1)[None, :]
        else:
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
        # See split-k kernel for M=1 GEMV rationale.
        if BLOCK_M == 1:
            prod = a.to(tl.float32) * b
            accumulator += tl.sum(prod, axis=1)[None, :]
        else:
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
    split_k = _select_split_k(M, N, K)

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

    if _env_fused_gemm_autotune(M, N, K) and split_k == 1 and M > 1:
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

    split_k = _select_split_k(m, n, k)

    if out is None:
        if split_k > 1:
            c = torch.zeros((m, n), device=a.device, dtype=a.dtype)
        else:
            c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    else:
        c = out
        if split_k > 1:
            c.zero_()

    profile_blocks = _lookup_persistent_blocks(
        "packed_int4_symmetric",
        m=m,
        n=n,
        k=k,
        group_size=int(group_size),
    )

    # Persistent JSON profiles are tuned for prefill / medium-M; M=1 decode
    # must use ``_select_fused_gemm_blocks`` so BLOCK_M==1 and the GEMV branch
    # inside the heuristic kernel are guaranteed.
    if profile_blocks is not None and split_k == 1 and m > 1:
        block_m, block_n, block_k, num_warps, num_stages = profile_blocks
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

    if _env_fused_gemm_autotune(m, n, k) and split_k == 1 and m > 1:
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
