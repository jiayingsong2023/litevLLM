# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel(
    X_ptr, Y_ptr, W_ptr,
    stride_x, stride_y,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    X_ptr += row_idx * stride_x
    Y_ptr += row_idx * stride_y

    mask = tl.arange(0, BLOCK_SIZE) < N
    x = tl.load(X_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)

    # Compute variance
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N
    rsqrt = tl.rsqrt(mean_sq + eps)

    # Load weight
    w = tl.load(W_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0).to(tl.float32)

    # Normalize and scale
    y = x * rsqrt * w

    tl.store(Y_ptr + tl.arange(0, BLOCK_SIZE), y, mask=mask)

def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    # Handle multidimensional input by flattening all but the last dimension
    orig_shape = input.shape
    input_2d = input.view(-1, orig_shape[-1])
    out_2d = out.view(-1, orig_shape[-1])
    
    M, N = input_2d.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    grid = (M,)
    
    _rms_norm_kernel[grid](
        input_2d, out_2d, weight,
        input_2d.stride(0), out_2d.stride(0),
        N, epsilon,
        BLOCK_SIZE=BLOCK_SIZE,
    )

@triton.jit
def _fused_add_rms_norm_kernel(
    X_ptr, Res_ptr, W_ptr,
    stride_x, stride_res,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    X_ptr += row_idx * stride_x
    Res_ptr += row_idx * stride_res

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(Res_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    new_x = x + res
    tl.store(Res_ptr + cols, new_x, mask=mask)

    # Compute variance on the sum
    x_sq = new_x * new_x
    mean_sq = tl.sum(x_sq, axis=0) / N
    rsqrt = tl.rsqrt(mean_sq + eps)

    # Load weight
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)

    # Normalize
    y = new_x * rsqrt * w

    # Store normalized output to X
    tl.store(X_ptr + cols, y, mask=mask)

def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    # Handle multidimensional input
    orig_shape = input.shape
    input_2d = input.view(-1, orig_shape[-1])
    residual_2d = residual.view(-1, orig_shape[-1])
    
    M, N = input_2d.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    
    _fused_add_rms_norm_kernel[grid](
        input_2d, residual_2d, weight,
        input_2d.stride(0), residual_2d.stride(0),
        N, epsilon,
        BLOCK_SIZE=BLOCK_SIZE,
    )
