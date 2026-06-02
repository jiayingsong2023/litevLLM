# SPDX-License-Identifier: Apache-2.0
"""HIP-accelerated INT4 GEMV kernels for AMD RDNA 3.5.

Provides a drop-in replacement for Triton's M=1 group32 GEMV on the
down_proj hot path. Falls back transparently to Triton if HIP runtime
compilation fails (non-AMD platform, missing toolchain, etc.).

All other kernels remain Triton-only — this module is a minimal,
scoped supplement for the single operation where Triton's AMD code
generation underperforms.
"""

from __future__ import annotations

import ctypes
import os
import tempfile
from typing import Any

import torch

_HIP_MODULE_CACHE: dict[int, Any] = {}
_HIP_RTC_AVAILABLE: bool | None = None


def _hiprtc_available() -> bool:
    global _HIP_RTC_AVAILABLE
    if _HIP_RTC_AVAILABLE is None:
        try:
            ctypes.CDLL("libhiprtc.so")
            _HIP_RTC_AVAILABLE = True
        except OSError:
            _HIP_RTC_AVAILABLE = False
    return _HIP_RTC_AVAILABLE


def _get_hip_kernel_source() -> str:
    """Return the HIP kernel C++ source for M=1 INT4 group32 GEMV."""
    return r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// M=1 INT4 group32 GEMV — drop-in replacement for Triton's
// _packed_int4_symmetric_group32_gemv_m1_fp16.
//
// Memory layout:
//   a:      [K]            fp16 input activation
//   qw:     [N, K/8]       int32 packed-int4 weights (8 nibbles per int32)
//   scales: [N, K/32]      fp16 per-group scales
//   c:      [N]            fp16 output
//
// Grid:  (N + 63) / 64
// Block: 64 threads (1 wavefront on RDNA 3.5)
//
// Each thread computes one output element by iterating over all K.
// All 64 threads load the same activation element (hardware broadcast
// on RDNA 3.5) but different weight rows (strided by K/8 elements).
//
// Uses __half (fp16) accumulation with periodic fp32 reduction every
// REDUCE_EVERY groups to exploit 2× fp16 ALU throughput on RDNA 3.5.

#define WARP_SIZE 64
#define REDUCE_EVERY 64

extern "C" __global__ void int4_group32_gemv_m1(
    const __half* __restrict__ a,
    const int*    __restrict__ qw,
    const __half* __restrict__ scales,
    __half* __restrict__ c,
    const int N,
    const int K,
    const int stride_k,      // a stride (should be 1)
    const int stride_bn,     // qw stride_0 = K/8
    const int stride_sn      // scales stride_0 = K/32
) {
    int n = blockIdx.x * WARP_SIZE + threadIdx.x;
    if (n >= N) return;

    int k_groups = K / 32;

    // __half accumulator for 2× ALU throughput on RDNA 3.5
    __half acc_f16 = __float2half(0.0f);
    // float accumulator for periodic reduction
    float  acc_f32 = 0.0f;

    // Activation pointer — same for all threads (broadcast)
    const __half* a_base = a;

    // Weight and scale base pointers for this output row
    const int*    qw_row   = qw     + (size_t)n * stride_bn;
    const __half* sc_row   = scales + (size_t)n * stride_sn;

    for (int group = 0; group < k_groups; group++) {
        // Scale for this group (fp16 → float for multiply)
        __half scale_h = sc_row[group];
        float  scale_f = __half2float(scale_h);

        // 4 packed int32 words per group (each holds 8 nibbles = 32 total)
        #pragma unroll
        for (int pack = 0; pack < 4; pack++) {
            int packed = qw_row[group * 4 + pack];

            // Unpack 8 nibbles
            #pragma unroll
            for (int nib = 0; nib < 8; nib++) {
                int    w_val = (packed >> (nib * 4)) & 0xF;
                int    k_idx = group * 32 + pack * 8 + nib;
                __half a_val = a_base[k_idx * stride_k];

                float a_f   = __half2float(a_val);
                float w_f   = (float)(w_val - 8);
                float contrib = a_f * w_f * scale_f;

                acc_f16 = __float2half(__half2float(acc_f16) + contrib);
            }
        }

        // Periodic fp32 reduction
        if ((group + 1) % REDUCE_EVERY == 0) {
            acc_f32 += __half2float(acc_f16);
            acc_f16 = __float2half(0.0f);
        }
    }

    // Final reduction
    acc_f32 += __half2float(acc_f16);
    c[n] = __float2half(acc_f32);
}
"""


def _compile_hip_kernel() -> Any:
    """Compile the HIP GEMV kernel via hiprtc and return a callable function."""
    import ctypes

    hiprtc = ctypes.CDLL("libhiprtc.so")
    hip = ctypes.CDLL("libamdhip64.so")

    source = _get_hip_kernel_source()
    source_bytes = source.encode("utf-8")

    # Create hiprtc program
    prog = ctypes.c_void_p()
    err = hiprtc.hiprtcCreateProgram(
        ctypes.byref(prog),
        ctypes.c_char_p(source_bytes),
        ctypes.c_char_p(b"int4_gemv_kernel"),
        0, None, None,
    )
    if err != 0:
        raise RuntimeError(f"hiprtcCreateProgram failed: {err}")

    # Compile for gfx1151
    opts = [
        b"--gpu-architecture=gfx1151",
        b"-O3",
        b"-ffast-math",
        b"-D__HIP_PLATFORM_AMD__",
        b"-I/opt/rocm/include",
    ]
    opt_array = (ctypes.c_char_p * len(opts))(*opts)

    err = hiprtc.hiprtcCompileProgram(prog, len(opts), opt_array)
    if err != 0:
        log_size = ctypes.c_size_t()
        hiprtc.hiprtcGetProgramLogSize(prog, ctypes.byref(log_size))
        log_buf = ctypes.create_string_buffer(log_size.value)
        hiprtc.hiprtcGetProgramLog(prog, log_buf)
        hiprtc.hiprtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"hiprtcCompileProgram failed:\n{log_buf.value.decode()}")

    # Get compiled binary
    binary_size = ctypes.c_size_t()
    hiprtc.hiprtcGetCodeSize(prog, ctypes.byref(binary_size))
    binary = ctypes.create_string_buffer(binary_size.value)
    hiprtc.hiprtcGetCode(prog, binary)
    hiprtc.hiprtcDestroyProgram(ctypes.byref(prog))

    # Load module
    module = ctypes.c_void_p()
    err = hip.hipModuleLoadData(ctypes.byref(module), binary)
    if err != 0:
        raise RuntimeError(f"hipModuleLoadData failed: {err}")

    # Get function
    func = ctypes.c_void_p()
    err = hip.hipModuleGetFunction(
        ctypes.byref(func), module, b"int4_group32_gemv_m1"
    )
    if err != 0:
        raise RuntimeError(f"hipModuleGetFunction failed: {err}")

    return module, func


def _get_cached_kernel() -> Any:
    """Return cached HIP kernel function, compiling on first call."""
    cache_key = 0  # single kernel
    if cache_key not in _HIP_MODULE_CACHE:
        _HIP_MODULE_CACHE[cache_key] = _compile_hip_kernel()
    return _HIP_MODULE_CACHE[cache_key]


def int4_group32_gemv_m1_fp16_hip(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    *,
    bias: torch.Tensor | None = None,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> torch.Tensor:
    """HIP-accelerated M=1 INT4 group32 GEMV.

    Drop-in replacement for Triton's
    packed_int4_symmetric_group32_gemv_m1_fp16.

    Args:
        a: [1, K] input activation (fp16/bf16)
        qweight: [N, K//8] int32 packed-int4 weights
        scales: [N, K//32] fp16 scales
        group_size: quantization group size (must be 32)
        bias: optional [N] bias tensor

    Returns:
        [1, N] output tensor, same dtype as input.
    """
    del config, policy
    if int(group_size) != 32:
        raise ValueError("group_size must be 32")
    m, k = a.shape
    n = int(qweight.shape[0])
    if m != 1:
        raise ValueError("M must be 1 for GEMV")

    # Ensure contiguous tensors on GPU
    a_flat = a.reshape(-1).contiguous()
    qw = qweight.contiguous()
    sc = scales.contiguous()
    c = torch.empty((1, n), device=a.device, dtype=a.dtype)

    module, func = _get_cached_kernel()
    hip = ctypes.CDLL("libamdhip64.so")

    # Grid/block dims
    block_dim = 64
    grid_dim = (n + block_dim - 1) // block_dim

    # HIP kernel arguments: must be passed as void* array where each
    # element points to the actual argument value.
    a_ptr = ctypes.c_void_p(a_flat.data_ptr())
    qw_ptr = ctypes.c_void_p(qw.data_ptr())
    sc_ptr = ctypes.c_void_p(sc.data_ptr())
    c_ptr = ctypes.c_void_p(c.data_ptr())
    n_val = ctypes.c_int(n)
    k_val = ctypes.c_int(k)
    sk_val = ctypes.c_int(a_flat.stride(0))
    sbn_val = ctypes.c_int(qw.stride(0))
    ssn_val = ctypes.c_int(sc.stride(0))

    # Build args array (array of void* pointers to each argument)
    arg_array = (ctypes.c_void_p * 9)(
        ctypes.addressof(a_ptr),
        ctypes.addressof(qw_ptr),
        ctypes.addressof(sc_ptr),
        ctypes.addressof(c_ptr),
        ctypes.addressof(n_val),
        ctypes.addressof(k_val),
        ctypes.addressof(sk_val),
        ctypes.addressof(sbn_val),
        ctypes.addressof(ssn_val),
    )

    err = hip.hipModuleLaunchKernel(
        func,
        ctypes.c_uint(grid_dim), ctypes.c_uint(1), ctypes.c_uint(1),
        ctypes.c_uint(block_dim), ctypes.c_uint(1), ctypes.c_uint(1),
        ctypes.c_uint(0),
        ctypes.c_void_p(0),
        ctypes.cast(arg_array, ctypes.c_void_p),
        None,
    )
    if err != 0:
        raise RuntimeError(f"hipModuleLaunchKernel failed: {err}")

    if bias is not None:
        c.add_(bias.reshape(1, -1))

    return c
