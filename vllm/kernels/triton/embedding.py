# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def _embedding_kernel(
    IDS, Weight, Out,
    stride_id, stride_weight_row, stride_weight_col,
    stride_out_row, stride_out_col,
    n_ids, n_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Load token ID
    id = tl.load(IDS + pid)
    
    # Offsets for dimension
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_dim
    
    # Load embedding vector
    w_ptr = Weight + id * stride_weight_row + offsets * stride_weight_col
    embedding = tl.load(w_ptr, mask=mask, other=0.0)
    
    # Store to output
    out_ptr = Out + pid * stride_out_row + offsets * stride_out_col
    tl.store(out_ptr, embedding, mask=mask)

def embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    n_ids = ids.numel()
    n_dim = weight.shape[1]
    
    out = torch.empty((n_ids, n_dim), device=ids.device, dtype=weight.dtype)
    
    BLOCK_SIZE = triton.next_power_of_2(n_dim)
    grid = (n_ids,)
    
    _embedding_kernel[grid](
        ids, weight, out,
        ids.stride(0), weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        n_ids, n_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
