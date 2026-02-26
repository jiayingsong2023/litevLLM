import torch

def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    # query: [num_tokens, num_heads, head_size]
    # key:   [num_tokens, num_kv_heads, head_size]
    # positions: [num_tokens]
    # cos_sin_cache: [max_pos, rot_dim] -> [cos, sin] concatenated
    
    # We will use the native PyTorch implementation for now.
    # The error was about shape mismatch on the output of the rotary.
    # The input to the original vLLM 'ops.rotary_embedding' was:
    # positions, query, key, self.head_size, self.cos_sin_cache, self.is_neox_style
    # and it was an in-place operation.

    # Reimplementing the logic from vllm.model_executor.layers.rotary_embedding.base.py
    # forward_static here.
    
    num_tokens = positions.shape[0]
    rot_dim = cos_sin_cache.shape[1]
    
    # Select cos/sin
    # cache shape: [max_pos, rot_dim]
    # Select based on positions: [num_tokens, rot_dim]
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    
    # Reshape cos/sin for broadcasting
    # cos: [num_tokens, rot_dim/2] -> [num_tokens, 1, rot_dim/2]
    # The unsqueeze(1) is for num_heads dim, so it broadcasts correctly.
    cos = cos.unsqueeze(1) # [num_tokens, 1, rot_dim/2]
    sin = sin.unsqueeze(1) # [num_tokens, 1, rot_dim/2]
    
    def apply_rotary(x: torch.Tensor):
        # x can be [num_tokens, num_heads, head_size] or [num_tokens, hidden_size]
        # We ensure it is [num_tokens, num_heads, head_size] for the rotary application
        num_tokens = x.shape[0]
        x_reshaped = x.view(num_tokens, -1, head_size)
        num_heads = x_reshaped.shape[1]
        
        x_rot = x_reshaped[..., :rot_dim]
        
        d = rot_dim // 2
        
        if is_neox:
            # Interleaved
            x_rot_reshaped = x_rot.view(num_tokens, num_heads, d, 2)
            x0 = x_rot_reshaped[..., 0]
            x1 = x_rot_reshaped[..., 1]

            y0 = x0 * cos - x1 * sin
            y1 = x0 * sin + x1 * cos

            x_rot_out = torch.stack((y0, y1), dim=-1).view(num_tokens, num_heads, rot_dim)
        else:
            # Half-Half (GPT-J)
            x1 = x_rot[..., :d]
            x2 = x_rot[..., d:]

            y1 = x1 * cos - x2 * sin
            y2 = x1 * sin + x2 * cos

            x_rot_out = torch.cat((y1, y2), dim=-1)
             
        # Update in place in the original tensor
        x_reshaped[..., :rot_dim] = x_rot_out

    apply_rotary(query)
    if key is not None:
        apply_rotary(key)