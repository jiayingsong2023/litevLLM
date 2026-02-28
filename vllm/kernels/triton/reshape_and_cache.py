# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def _reshape_and_cache_kernel(
    key, value,
    key_cache, value_cache,
    slot_mapping,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_kcb, stride_kcs, stride_kch, stride_kcd,
    stride_vcb, stride_vcs, stride_vch, stride_vcd,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # 每个 Program 处理一个 (Token, Head) 组合，最大化并行度
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    slot_idx = tl.load(slot_mapping + token_idx)
    if slot_idx < 0:
        return

    block_idx = slot_idx // BLOCK_SIZE
    block_offset = slot_idx % BLOCK_SIZE

    # 向量化加载整个 Head Dimension (例如 128)
    off_d = tl.arange(0, HEAD_DIM)
    
    # 加载 K, V (对齐内存访问)
    k_ptr = key + token_idx * stride_kt + head_idx * stride_kh + off_d * stride_kd
    v_ptr = value + token_idx * stride_vt + head_idx * stride_vh + off_d * stride_vd
    k = tl.load(k_ptr)
    v = tl.load(v_ptr)
    
    # 写入物理 Paged Cache
    kc_ptr = key_cache + block_idx * stride_kcb + block_offset * stride_kcs + head_idx * stride_kch + off_d * stride_kcd
    vc_ptr = value_cache + block_idx * stride_vcb + block_offset * stride_vcs + head_idx * stride_vch + off_d * stride_vcd
    
    tl.store(kc_ptr, k)
    tl.store(vc_ptr, v)

def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_dim = key.shape[2]
    block_size = key_cache.shape[1]
    
    # 并行维度: [Tokens, Heads]
    grid = (num_tokens, num_heads)
    
    _reshape_and_cache_kernel[grid](
        key, value, key_cache, value_cache, slot_mapping,
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
    )
def reshape_and_cache_flash(*args, **kwargs): pass
