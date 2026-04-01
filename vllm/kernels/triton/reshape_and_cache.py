# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

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

@triton.jit
def _reshape_and_cache_kernel(
    key, value,
    key_cache, value_cache,
    slot_mapping,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_kcb, stride_kcs, stride_kch, stride_kcd,
    stride_vcb, stride_vcs, stride_vch, stride_vcd,
    K_Scale_ptr, V_Scale_ptr, # Now pointers
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    IS_INT4: tl.constexpr,
    HAS_ROW_SCALE: tl.constexpr, # Flag for per-token scale
    FP8_DTYPE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    slot_idx = tl.load(slot_mapping + token_idx)
    if slot_idx < 0: return

    block_idx = slot_idx // BLOCK_SIZE
    block_offset = slot_idx % BLOCK_SIZE
    
    k_scale = tl.load(K_Scale_ptr + token_idx) if HAS_ROW_SCALE else tl.load(K_Scale_ptr)
    v_scale = tl.load(V_Scale_ptr + token_idx) if HAS_ROW_SCALE else tl.load(V_Scale_ptr)
    
    if IS_INT4:
        # For INT4, we process in chunks of 2 to pack into uint8
        off_d_half = tl.arange(0, HEAD_DIM // 2)
        off_d_low = off_d_half * 2
        off_d_high = off_d_half * 2 + 1
        
        k_ptr_low = key + token_idx * stride_kt + head_idx * stride_kh + off_d_low * stride_kd
        k_ptr_high = key + token_idx * stride_kt + head_idx * stride_kh + off_d_high * stride_kd
        v_ptr_low = value + token_idx * stride_vt + head_idx * stride_vh + off_d_low * stride_vd
        v_ptr_high = value + token_idx * stride_vt + head_idx * stride_vh + off_d_high * stride_vd
        
        k_low = tl.load(k_ptr_low).to(tl.float32)
        k_high = tl.load(k_ptr_high).to(tl.float32)
        v_low = tl.load(v_ptr_low).to(tl.float32)
        v_high = tl.load(v_ptr_high).to(tl.float32)
        
        # Symmetric quantization to [-8, 7], then offset to [0, 15] for packing
        # k_scale is expected to be (max_abs / 8.0)
        # Using floor(x + 0.5) for proper rounding across all signs.
        k_l_q = (tl.clamp(tl.math.floor(k_low / k_scale + 0.5), -8.0, 7.0).to(tl.int32) + 8).to(tl.uint8)
        k_h_q = (tl.clamp(tl.math.floor(k_high / k_scale + 0.5), -8.0, 7.0).to(tl.int32) + 8).to(tl.uint8)
        v_l_q = (tl.clamp(tl.math.floor(v_low / v_scale + 0.5), -8.0, 7.0).to(tl.int32) + 8).to(tl.uint8)
        v_h_q = (tl.clamp(tl.math.floor(v_high / v_scale + 0.5), -8.0, 7.0).to(tl.int32) + 8).to(tl.uint8)
        
        k_packed = k_l_q | (k_h_q << 4)
        v_packed = v_l_q | (v_h_q << 4)
        
        kc_ptr = key_cache + block_idx * stride_kcb + block_offset * stride_kcs + head_idx * stride_kch + off_d_half * stride_kcd
        vc_ptr = value_cache + block_idx * stride_vcb + block_offset * stride_vcs + head_idx * stride_vch + off_d_half * stride_vcd
        tl.store(kc_ptr, k_packed)
        tl.store(vc_ptr, v_packed)
    else:
        off_d = tl.arange(0, HEAD_DIM)
        k_ptr = key + token_idx * stride_kt + head_idx * stride_kh + off_d * stride_kd
        v_ptr = value + token_idx * stride_vt + head_idx * stride_vh + off_d * stride_vd
        
        k = tl.load(k_ptr).to(tl.float32)
        v = tl.load(v_ptr).to(tl.float32)
        
        kc_ptr = key_cache + block_idx * stride_kcb + block_offset * stride_kcs + head_idx * stride_kch + off_d * stride_kcd
        vc_ptr = value_cache + block_idx * stride_vcb + block_offset * stride_vcs + head_idx * stride_vch + off_d * stride_vcd
        
        if IS_FP8:
            tl.store(kc_ptr, (k * k_scale).to(FP8_DTYPE))
            tl.store(vc_ptr, (v * v_scale).to(FP8_DTYPE))
        else:
            tl.store(kc_ptr, k.to(key_cache.dtype.element_ty))
            tl.store(vc_ptr, v.to(value_cache.dtype.element_ty))

def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale=1.0, v_scale=1.0):
    num_tokens, num_heads, head_dim = key.shape
    block_size = key_cache.shape[1]
    is_fp8 = "fp8" in str(kv_cache_dtype).lower()
    is_int4 = "int4" in str(kv_cache_dtype).lower()
    
    # Handle scalar vs tensor scales
    if not isinstance(k_scale, torch.Tensor):
        k_scale_t = torch.tensor([k_scale], device=key.device, dtype=torch.float32)
    else:
        k_scale_t = k_scale
    if not isinstance(v_scale, torch.Tensor):
        v_scale_t = torch.tensor([v_scale], device=value.device, dtype=torch.float32)
    else:
        v_scale_t = v_scale
    
    has_row_scale = k_scale_t.numel() > 1

    grid = (num_tokens, num_heads)
    _reshape_and_cache_kernel[grid](
        key, value, key_cache, value_cache, slot_mapping,
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        k_scale_t, v_scale_t,
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim, IS_FP8=is_fp8, IS_INT4=is_int4,
        HAS_ROW_SCALE=has_row_scale,
        FP8_DTYPE=FP8_DTYPE
    )
def reshape_and_cache_flash(*args, **kwargs): pass
