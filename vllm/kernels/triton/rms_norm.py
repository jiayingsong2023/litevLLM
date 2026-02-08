
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
    M, N = input.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # Heuristic for Block Size
    # If N is too large, we might need a different kernel strategy (tiling), 
    # but for typical hidden sizes (up to 8k-16k), a single block is usually fine on modern GPUs.
    # However, triton has a limit on block size. 
    # For simplicity in this POC, we assume N fits in shared memory/block limit.
    
    grid = (M,)
    
    _rms_norm_kernel[grid](
        input, out, weight,
        input.stride(0), out.stride(0),
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

    # Fuse add: x = x + residual
    # Note: vLLM implementation updates residual in-place too? 
    # The signature is fused_add_rms_norm(input, residual, weight, epsilon)
    # Usually this means input += residual, then norm. 
    # Or residual = input + residual?
    # Let's check the pytorch reference in layernorm.py:
    #   x = x + residual
    #   residual = x
    # So both input and residual are updated to the sum.
    
    new_x = x + res
    
    # Store updated residual (which matches the new input before norm)
    # The CUDA kernel usually updates 'input' to be the normalized value, 
    # and 'residual' to be the sum.
    # Wait, vllm/model_executor/layers/layernorm.py says:
    #   ops.fused_add_rms_norm(x, residual, weight, eps)
    #   return x, residual
    # And the native implementation:
    #   x = x + residual
    #   residual = x
    #   variance = ...
    #   x = x * rsqrt * weight
    # So: 'residual' becomes (old_x + old_residual)
    #     'x' becomes Norm(old_x + old_residual)
    
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
    M, N = input.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    
    _fused_add_rms_norm_kernel[grid](
        input, residual, weight,
        input.stride(0), residual.stride(0),
        N, epsilon,
        BLOCK_SIZE=BLOCK_SIZE,
    )
