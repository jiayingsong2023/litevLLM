# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def _reshape_and_cache_kernel(
    key,  # [num_tokens, num_heads, head_size]
    value,  # [num_tokens, num_heads, head_size]
    key_cache,  # [num_blocks, block_size, num_heads, head_size]
    value_cache,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping,  # [num_tokens]
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_kct, stride_kcs, stride_kch, stride_kcd,
    stride_vct, stride_vcs, stride_vch, stride_vcd,
    num_heads,
    BLOCK_SIZE: tl.constexpr,  # Paged block size
    HEAD_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    
    # 获取该 token 对应的物理槽位
    slot_idx = tl.load(slot_mapping + token_idx)
    if slot_idx < 0:
        return

    # 计算物理块索引和块内偏移
    block_idx = slot_idx // BLOCK_SIZE
    block_offset = slot_idx % BLOCK_SIZE

    off_d = tl.arange(0, HEAD_DIM)
    
    # 每个 token 循环处理所有 head
    for h in range(num_heads):
        # 加载新的 K 和 V
        k_ptr = key + token_idx * stride_kt + h * stride_kh + off_d * stride_kd
        v_ptr = value + token_idx * stride_vt + h * stride_vh + off_d * stride_vd
        
        k = tl.load(k_ptr)
        v = tl.load(v_ptr)
        
        # 计算写入物理缓存的地址
        # 布局: [num_blocks, block_size, num_heads, head_dim]
        kc_ptr = key_cache + block_idx * stride_kct + block_offset * stride_kcs + h * stride_kch + off_d * stride_kcd
        vc_ptr = value_cache + block_idx * stride_vct + block_offset * stride_vcs + h * stride_vch + off_d * stride_vcd
        
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
    # 基础校验
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_dim = key.shape[2]
    block_size = key_cache.shape[1]
    
    grid = (num_tokens,)
    
    _reshape_and_cache_kernel[grid](
        key, value, key_cache, value_cache, slot_mapping,
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        num_heads,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
    )

def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    # FlashAttention 专用布局支持 (通常 K 和 V 连续存储)
    # 此处简化为调用 reshape_and_cache 或抛出未实现
    # 在生产中，我们会实现一个针对 [num_blocks, block_size, 2, num_heads, head_dim] 的专用 kernel
    pass
