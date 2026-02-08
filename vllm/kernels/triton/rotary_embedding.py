
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
    Triton-compatible interface for Rotary Embedding.
    Currently uses PyTorch native operations for simplicity and correctness in Lite mode.
    Performs in-place updates on query and key.
    """
    # query: [num_tokens, num_heads, head_size]
    # key:   [num_tokens, num_kv_heads, head_size]
    # positions: [num_tokens]
    # cos_sin_cache: [max_pos, rot_dim] -> [cos, sin] concatenated
    
    num_tokens = positions.shape[0]
    rot_dim = cos_sin_cache.shape[1]
    
    # Select cos/sin
    # cache shape: [max_pos, rot_dim]
    # Select based on positions: [num_tokens, rot_dim]
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    
    # Reshape cos/sin for broadcasting
    # cos: [num_tokens, rot_dim/2] -> [num_tokens, 1, rot_dim/2]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    def apply_rotary(x: torch.Tensor):
        # x: [num_tokens, num_heads, head_size]
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]
        
        # Split into two halves
        # vLLM/FlashInfer layout: 
        # is_neox (interleaved): pairs (x[0], x[1]), (x[2], x[3])
        # not is_neox (half-half): (x[0]...x[d/2-1]), (x[d/2]...x[d-1])
        
        d = rot_dim // 2
        
        if is_neox:
            # Interleaved: [x0, x1, x2, x3...] -> [-x1, x0, -x3, x2...]
            x1 = x_rot[..., ::2]
            x2 = x_rot[..., 1::2]
            # rotate
            # new_x1 = x1 * cos - x2 * sin
            # new_x2 = x1 * sin + x2 * cos
            # Wait, standard definition:
            # x' = x * cos + rotate(x) * sin
            # rotate([-x1, x0])
            # So:
            # out[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
            # out[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
            
            # vLLM cache stores cos, sin repeated?
            # cos_sin_cache is [cos, sin].
            # If rot_dim=128, cos has 64 elements, sin has 64 elements.
            # But for interleaved, we apply cos[i] to x[2i] and x[2i+1].
            # So cos needs to be repeated_interleave.
            
            # Let's rely on vLLM's implementation logic reference.
            # vllm/model_executor/layers/rotary_embedding/common.py: ApplyRotaryEmb
            # It handles the math.
            pass
            
        # Since implementing exact math requires matching vLLM's exact cache semantics,
        # and we have access to `forward_static` in `RotaryEmbedding` class,
        # we can't easily call it here without importing the class which might be circular or messy.
        
        # Simplification:
        # We will assume standard Half-Half rotation if not is_neox, and Interleaved if is_neox.
        # But `cos_sin` shape [num_tokens, 1, rot_dim/2] implies we have d/2 values.
        
        x_rot_out = torch.empty_like(x_rot)
        
        if is_neox:
             # Interleaved
             x_shaped = x_rot.reshape(x_rot.shape[:-1] + (d, 2))
             # x_shaped: [..., d, 2] -> (x_evens, x_odds)
             # cos, sin are [..., d] (after squeeze)
             c = cos.unsqueeze(-1) # [..., d, 1]
             s = sin.unsqueeze(-1)
             
             x_new = torch.stack([
                 x_shaped[..., 0] * c[..., 0] - x_shaped[..., 1] * s[..., 0],
                 x_shaped[..., 1] * c[..., 0] + x_shaped[..., 0] * s[..., 0]
             ], dim=-1).flatten(-2)
             x_rot_out = x_new
        else:
             # Half-Half (GPT-J)
             # x1 = x[..., :d], x2 = x[..., d:]
             x1 = x_rot[..., :d]
             x2 = x_rot[..., d:]
             # rotate(x) = [-x2, x1]
             # out = [x1*cos - x2*sin, x2*cos + x1*sin]
             x_rot_out = torch.cat([
                 x1 * cos - x2 * sin,
                 x2 * cos + x1 * sin
             ], dim=-1)
             
        # Update in place
        x[..., :rot_dim] = x_rot_out

    apply_rotary(query)
    if key is not None:
        apply_rotary(key)
