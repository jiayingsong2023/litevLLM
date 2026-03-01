# SPDX-License-Identifier: Apache-2.0
import torch

def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """
    使用 PyTorch 稳定版实现的 RoPE。
    """
    # query: [num_tokens, num_heads, head_size]
    # cos_sin_cache: [max_pos, rot_dim]
    rot_dim = cos_sin_cache.shape[1]
    cos, sin = cos_sin_cache.chunk(2, dim=-1)
    
    # 提取对应位置的 cos/sin
    cos = cos[positions].unsqueeze(1) # [num_tokens, 1, rot_dim//2]
    sin = sin[positions].unsqueeze(1) # [num_tokens, 1, rot_dim//2]
    
    def apply_rotary(x):
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]
        
        # Llama style (Half-Half)
        d = x_rot.shape[-1] // 2
        x1, x2 = x_rot.chunk(2, dim=-1)
        x_transformed = torch.cat((-x2, x1), dim=-1)
        
        x_rot = (x_rot * cos) + (x_transformed * sin)
        return torch.cat((x_rot, x_pass), dim=-1)

    query.copy_(apply_rotary(query))
    if key is not None:
        key.copy_(apply_rotary(key))
