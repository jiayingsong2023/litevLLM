# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def _lite_rmsnorm_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_x_row, stride_x_col,
    stride_o_row, stride_o_col,
    n_cols, eps,
    BLOCK_N: tl.constexpr,
):
    """
    Pure RMSNorm Kernel optimized for minimal register footprint on APUs.
    """
    row_idx = tl.program_id(0)
    
    # Load input row
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x_ptr = X_ptr + row_idx * stride_x_row + cols * stride_x_col
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    
    # Compute Variance
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.math.rsqrt(var + eps)
    
    # Normalize and scale
    w_ptr = W_ptr + cols
    w = tl.load(w_ptr, mask=mask).to(tl.float32)
    out = (x * rstd * w).to(Out_ptr.dtype.element_ty)
    
    # Store result
    o_ptr = Out_ptr + row_idx * stride_o_row + cols * stride_o_col
    tl.store(o_ptr, out, mask=mask)

def lite_rmsnorm(x, weight, eps=1e-6, out=None):
    M, N = x.shape
    if out is None:
        out = torch.empty_like(x)
    
    # Optimization: Use the largest power of 2 for BLOCK_N to ensure vectorization
    BLOCK_N = triton.next_power_of_2(N)
    
    grid = (M, )
    _lite_rmsnorm_kernel[grid](
        x, weight, out,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        N, eps,
        BLOCK_N=BLOCK_N,
        num_warps=4 if N <= 8192 else 8
    )
    return out
