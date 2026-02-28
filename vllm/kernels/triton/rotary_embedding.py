# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def _rotary_embedding_kernel(
    Q,  # [num_tokens, num_heads, head_size]
    K,  # [num_tokens, num_kv_heads, head_size]
    Cos,  # [max_seq_len, rotary_dim // 2]
    Sin,  # [max_seq_len, rotary_dim // 2]
    pos_ptr,  # [num_tokens]
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_cs_s, stride_cs_d,
    num_heads, num_kv_heads,
    ROTARY_DIM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_NEOX: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # 获取位置索引
    pos = tl.load(pos_ptr + token_idx)
    
    # 确定是 Q 还是 K
    is_query = head_idx < num_heads
    if is_query:
        ptr = Q + token_idx * stride_qt + head_idx * stride_qh
    else:
        ptr = K + token_idx * stride_kt + (head_idx - num_heads) * stride_kh

    # 旋转维度一半的索引
    half_dim = ROTARY_DIM // 2
    off_d = tl.arange(0, half_dim)
    
    # 加载 Cos 和 Sin
    cos = tl.load(Cos + pos * stride_cs_s + off_d * stride_cs_d)
    sin = tl.load(Sin + pos * stride_cs_s + off_d * stride_cs_d)

    if IS_NEOX:
        # 交错布局 (Interleaved): [x0, x1, x2, x3] -> [x0*cos-x1*sin, x0*sin+x1*cos, ...]
        off_d0 = tl.arange(0, half_dim) * 2
        off_d1 = off_d0 + 1
        
        x0 = tl.load(ptr + off_d0)
        x1 = tl.load(ptr + off_d1)
        
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        
        tl.store(ptr + off_d0, out0)
        tl.store(ptr + off_d1, out1)
    else:
        # Llama 布局 (Half-Half): [x0, x1, x2, x3] -> [x0*cos-x2*sin, x1*cos-x3*sin, ...]
        off_d0 = tl.arange(0, half_dim)
        off_d1 = off_d0 + half_dim
        
        x0 = tl.load(ptr + off_d0)
        x1 = tl.load(ptr + off_d1)
        
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        
        tl.store(ptr + off_d0, out0)
        tl.store(ptr + off_d1, out1)

def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    # 预处理 Cos/Sin Cache
    # 假设 cos_sin_cache 形状为 [max_pos, rot_dim]，前一半是 cos，后一半是 sin
    rotary_dim = cos_sin_cache.shape[1]
    cos, sin = cos_sin_cache.chunk(2, dim=-1)
    
    num_tokens = positions.shape[0]
    num_heads = query.shape[1]
    num_kv_heads = key.shape[1] if key is not None else 0
    total_heads = num_heads + num_kv_heads
    
    # 准备空的 K 如果 key 为 None，方便统一 Kernel 逻辑
    if key is None:
        key = torch.empty((num_tokens, 0, head_size), device=query.device, dtype=query.dtype)

    grid = (num_tokens, total_heads)
    
    _rotary_embedding_kernel[grid](
        query, key, cos, sin, positions,
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        cos.stride(0), cos.stride(1),
        num_heads, num_kv_heads,
        ROTARY_DIM=rotary_dim,
        HEAD_DIM=head_size,
        IS_NEOX=is_neox,
    )
