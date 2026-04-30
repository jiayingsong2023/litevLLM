# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Tuple

import torch
from vllm.triton_utils import triton, tl


# Robust float8 type resolution for different Triton versions
def _get_fp8_dtype():
    # Standard names
    for name in ["float8e4m3fn", "float8_e4m3fn", "float8e4m3fnuz"]:
        if hasattr(tl, name):
            return getattr(tl, name)
    # Vendor specific or older names
    for name in ["float8e4nv", "float8e4b8", "float8e4b15"]:
        if hasattr(tl, name):
            return getattr(tl, name)
    return None


FP8_DTYPE = _get_fp8_dtype()


# ---- Step 5: per-(device, dtype, value) scalar scale tensor cache ----
#
# reshape_and_cache gets called once per decoder layer per step. When the
# caller passes Python scalar k_scale / v_scale (the non-dynamic path,
# which covers bf16/f16 KV writes and fp8 with a fixed calibration scale),
# naive ``torch.tensor([k_scale], device=...)`` constructs a fresh
# 1-element CUDA tensor each call, which on ROCm surfaces as a
# ``hipMemcpyWithStream`` H->D copy that blocks behind the current Triton
# stream. Profiling Gemma4-31B decode shows this was ~120 syncs / step
# and ~79% of wall time.
#
# We deduplicate the tensor construction using a tiny module-level cache.
# Keys include the device, dtype and the scalar *value* so fp8 callers
# with non-unit scales stay bitwise-correct without each layer paying the
# allocation cost after the first step.
_SCALE_TENSOR_CACHE: Dict[Tuple[torch.device, torch.dtype, float], torch.Tensor] = {}


def _get_scalar_scale_tensor(
    value: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a cached 1-element tensor on ``device`` holding ``value``.

    The returned tensor is owned by the cache and MUST be treated as
    read-only by the caller; reshape_and_cache only reads it via
    ``tl.load``.
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    # Normalize to a plain Python float so the dict key is hashable and
    # value-stable (NaN is not expected for scale; we do not special-case
    # it because an upstream NaN scale would itself be a correctness bug).
    key = (device, dtype, float(value))
    cached = _SCALE_TENSOR_CACHE.get(key)
    if cached is None:
        cached = torch.tensor([float(value)], device=device, dtype=dtype)
        _SCALE_TENSOR_CACHE[key] = cached
    return cached


def _clear_scale_tensor_cache() -> None:
    """Test-only helper: drop the per-device scalar scale cache.

    Production code never needs to call this; it exists so test fixtures
    can assert that the cache is being populated lazily and reused across
    calls instead of reallocated.
    """
    _SCALE_TENSOR_CACHE.clear()


@triton.jit
def _reshape_and_cache_kernel(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_kcb,
    stride_kcs,
    stride_kch,
    stride_kcd,
    stride_vcb,
    stride_vcs,
    stride_vch,
    stride_vcd,
    K_Scale_cache,
    V_Scale_cache,
    stride_ksb,
    stride_kss,
    stride_ksh,
    stride_vsb,
    stride_vss,
    stride_vsh,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    IS_INT4: tl.constexpr,
    COMPUTE_DYNAMIC_SCALE: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping + token_idx)
    if slot_idx < 0:
        return

    block_idx = slot_idx // BLOCK_SIZE
    block_offset = slot_idx % BLOCK_SIZE

    # Input K,V have num_kv_heads in dim 1
    base_k_ptr = key + token_idx * stride_kt + head_idx * stride_kh
    base_v_ptr = value + token_idx * stride_vt + head_idx * stride_vh

    off_d = tl.arange(0, HEAD_DIM)
    k = tl.load(base_k_ptr + off_d * stride_kd).to(tl.float32)
    v = tl.load(base_v_ptr + off_d * stride_vd).to(tl.float32)

    k_scale = 1.0
    v_scale = 1.0

    if COMPUTE_DYNAMIC_SCALE:
        # Dynamic Row-wise Scale with Clipping for MoE Outliers
        # We use a divisor of 6.0 to give more bits to non-outliers.
        # This effectively treats values > 6.0*scale as outliers to be clipped.
        k_max = tl.max(tl.abs(k))
        v_max = tl.max(tl.abs(v))
        k_scale = tl.maximum(k_max / 7.0, 1e-4)
        v_scale = tl.maximum(v_max / 7.0, 1e-4)

        # Experimental: If max is too high, clip the scale to protect small values
        # For Qwen3.5 35B, outliers can be huge.
        # k_scale = tl.minimum(k_scale, 0.5)
        # v_scale = tl.minimum(v_scale, 0.5)

        # Store to scale cache: (block, offset, head, 1)
        ks_ptr = (
            K_Scale_cache
            + block_idx * stride_ksb
            + block_offset * stride_kss
            + head_idx * stride_ksh
        )
        vs_ptr = (
            V_Scale_cache
            + block_idx * stride_vsb
            + block_offset * stride_vss
            + head_idx * stride_vsh
        )
        tl.store(ks_ptr, k_scale)
        tl.store(vs_ptr, v_scale)
    else:
        # Load from single scale pointer (legacy / scalar)
        k_scale = tl.load(K_Scale_cache)
        v_scale = tl.load(V_Scale_cache)

    if IS_INT4:
        off_d_half = tl.arange(0, HEAD_DIM // 2)
        # Contiguous layout: first half of channels in low nibbles, second half in high nibbles
        k_low = tl.load(base_k_ptr + off_d_half * stride_kd).to(tl.float32)
        k_high = tl.load(base_k_ptr + (off_d_half + HEAD_DIM // 2) * stride_kd).to(
            tl.float32
        )
        v_low = tl.load(base_v_ptr + off_d_half * stride_vd).to(tl.float32)
        v_high = tl.load(base_v_ptr + (off_d_half + HEAD_DIM // 2) * stride_vd).to(
            tl.float32
        )

        k_l_q = (
            tl.clamp(tl.math.floor(k_low / k_scale + 0.5), -7.0, 7.0)
            .to(tl.int8)
            .to(tl.uint8)
            & 0x0F
        )
        k_h_q = (
            tl.clamp(tl.math.floor(k_high / k_scale + 0.5), -7.0, 7.0)
            .to(tl.int8)
            .to(tl.uint8)
            & 0x0F
        )
        v_l_q = (
            tl.clamp(tl.math.floor(v_low / v_scale + 0.5), -7.0, 7.0)
            .to(tl.int8)
            .to(tl.uint8)
            & 0x0F
        )
        v_h_q = (
            tl.clamp(tl.math.floor(v_high / v_scale + 0.5), -7.0, 7.0)
            .to(tl.int8)
            .to(tl.uint8)
            & 0x0F
        )

        k_packed = k_l_q | (k_h_q << 4)
        v_packed = v_l_q | (v_h_q << 4)

        kc_ptr = (
            key_cache
            + block_idx * stride_kcb
            + block_offset * stride_kcs
            + head_idx * stride_kch
            + off_d_half * stride_kcd
        )
        vc_ptr = (
            value_cache
            + block_idx * stride_vcb
            + block_offset * stride_vcs
            + head_idx * stride_vch
            + off_d_half * stride_vcd
        )
        tl.store(kc_ptr, k_packed)
        tl.store(vc_ptr, v_packed)
    else:
        kc_ptr = (
            key_cache
            + block_idx * stride_kcb
            + block_offset * stride_kcs
            + head_idx * stride_kch
            + off_d * stride_kcd
        )
        vc_ptr = (
            value_cache
            + block_idx * stride_vcb
            + block_offset * stride_vcs
            + head_idx * stride_vch
            + off_d * stride_vcd
        )

        if IS_FP8:
            tl.store(kc_ptr, (k * k_scale).to(FP8_DTYPE))
            tl.store(vc_ptr, (v * v_scale).to(FP8_DTYPE))
        else:
            tl.store(kc_ptr, k.to(key_cache.dtype.element_ty))
            tl.store(vc_ptr, v.to(value_cache.dtype.element_ty))


def reshape_and_cache(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    kv_cache_dtype,
    k_scale=1.0,
    v_scale=1.0,
    k_scale_cache=None,
    v_scale_cache=None,
):
    num_tokens, num_kv_heads, head_dim = key.shape
    block_size = key_cache.shape[1]
    is_fp8 = "fp8" in str(kv_cache_dtype).lower()
    is_int4 = "int4" in str(kv_cache_dtype).lower()

    compute_dynamic = (
        k_scale_cache is not None and v_scale_cache is not None and is_int4
    )

    if compute_dynamic:
        ks, vs = k_scale_cache, v_scale_cache
    else:
        # Fallback path: kernel reads a single-element scale tensor via
        # tl.load(K_Scale_cache) / tl.load(V_Scale_cache). Caller may have
        # passed either a Python scalar or an already-constructed tensor.
        #
        # Tensors are used as-is (preserve legacy semantics for dynamic
        # calibration scales that upstream wants to pin). Scalars are
        # resolved via the module cache so the decode hot path stops
        # issuing a per-layer 4-byte H->D hipMemcpyWithStream.
        if isinstance(k_scale, torch.Tensor):
            ks = k_scale
        else:
            ks = _get_scalar_scale_tensor(k_scale, key.device)
        if isinstance(v_scale, torch.Tensor):
            vs = v_scale
        else:
            vs = _get_scalar_scale_tensor(v_scale, value.device)

    grid = (num_tokens, num_kv_heads)
    _reshape_and_cache_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        ks,
        vs,
        ks.stride(0) if compute_dynamic else 0,
        ks.stride(1) if compute_dynamic else 0,
        ks.stride(2) if compute_dynamic else 0,
        vs.stride(0) if compute_dynamic else 0,
        vs.stride(1) if compute_dynamic else 0,
        vs.stride(2) if compute_dynamic else 0,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        IS_FP8=is_fp8,
        IS_INT4=is_int4,
        COMPUTE_DYNAMIC_SCALE=compute_dynamic,
        FP8_DTYPE=FP8_DTYPE,
    )
