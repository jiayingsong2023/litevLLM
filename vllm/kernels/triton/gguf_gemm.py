import torch
import triton
import triton.language as tl

@triton.jit
def matmul_q4_k_vec_kernel(
    w_ptr,              # Quantized weights
    x_ptr,              # Input vector (N,)
    out_ptr,            # Output vector (M,)
    n_elements,         # Number of columns (N)
    stride_w_row,       # Stride for W rows (in bytes)
    stride_x,           # Stride for X
    stride_out,         # Stride for Output
    BLOCK_SIZE: tl.constexpr, # Must be 256 for Q4_K superblock
):
    row_idx = tl.program_id(0)
    w_row_ptr = w_ptr + row_idx * stride_w_row
    x_base_ptr = x_ptr

    # Scalar accumulator reduces register pressure vs vector-wide accumulation.
    acc = tl.zeros((1,), dtype=tl.float32)

    # Q4_K superblock: 4B header + 12B scales + 128B quants = 144B.
    bytes_per_block = 144
    offs_128 = tl.arange(0, 128)
    num_blocks = n_elements // 256

    for block_idx in range(num_blocks):
        w_block_ptr = w_row_ptr + block_idx * bytes_per_block

        x_block_base = block_idx * 256
        x_low_ptr = x_base_ptr + (x_block_base + offs_128 * 2) * stride_x
        x_high_ptr = x_low_ptr + stride_x
        x_low = tl.load(x_low_ptr)
        x_high = tl.load(x_high_ptr)

        d_ptr = w_block_ptr.to(tl.pointer_type(tl.float16))
        dmin_ptr = (w_block_ptr + 2).to(tl.pointer_type(tl.float16))
        d = tl.load(d_ptr)
        dmin = tl.load(dmin_ptr)

        qs = tl.load(w_block_ptr + 16 + offs_128)
        q_low = (qs & 0x0F).to(tl.float16)
        q_high = ((qs >> 4) & 0x0F).to(tl.float16)

        w_low = (q_low * d) - dmin
        w_high = (q_high * d) - dmin

        block_dot = tl.sum(w_low * x_low + w_high * x_high)
        acc += block_dot

    tl.store(out_ptr + row_idx * stride_out, acc.to(tl.float16))

def matmul_q4_k_vec(
    W: torch.Tensor,
    X: torch.Tensor,
    out: torch.Tensor,
    n_elements: int,
):
    M = W.shape[0]
    grid = (M, 1, 1)
    
    # Triton needs at least one block size to be a power of 2 for vectorization
    matmul_q4_k_vec_kernel[grid](
        W, X, out,
        n_elements,
        W.stride(0),
        X.stride(0),
        out.stride(0),
        BLOCK_SIZE=256
    )
    return out


@triton.jit
def matmul_q4_k_tokens_kernel(
    w_ptr,                  # Quantized weights [M, encoded_n]
    x_ptr,                  # Input tokens [T, N]
    out_ptr,                # Output [T, M]
    n_elements,             # N
    stride_w_row,           # W stride row
    stride_x_token,         # X stride token
    stride_x_col,           # X stride col
    stride_out_token,       # OUT stride token
    stride_out_col,         # OUT stride col
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    token_idx = tl.program_id(1)

    w_row_ptr = w_ptr + row_idx * stride_w_row
    x_token_ptr = x_ptr + token_idx * stride_x_token

    acc = tl.zeros((1,), dtype=tl.float32)
    bytes_per_block = 144
    offs_128 = tl.arange(0, 128)
    num_blocks = n_elements // 256

    for block_idx in range(num_blocks):
        w_block_ptr = w_row_ptr + block_idx * bytes_per_block
        x_block_base = block_idx * 256
        x_low_ptr = x_token_ptr + (x_block_base + offs_128 * 2) * stride_x_col
        x_high_ptr = x_low_ptr + stride_x_col
        x_low = tl.load(x_low_ptr)
        x_high = tl.load(x_high_ptr)

        d_ptr = w_block_ptr.to(tl.pointer_type(tl.float16))
        dmin_ptr = (w_block_ptr + 2).to(tl.pointer_type(tl.float16))
        d = tl.load(d_ptr)
        dmin = tl.load(dmin_ptr)

        qs = tl.load(w_block_ptr + 16 + offs_128)
        q_low = (qs & 0x0F).to(tl.float16)
        q_high = ((qs >> 4) & 0x0F).to(tl.float16)
        w_low = (q_low * d) - dmin
        w_high = (q_high * d) - dmin
        block_dot = tl.sum(w_low * x_low + w_high * x_high)
        acc += block_dot

    out_ptr_curr = out_ptr + token_idx * stride_out_token + row_idx * stride_out_col
    tl.store(out_ptr_curr, acc.to(tl.float16))


def matmul_q4_k_tokens(
    W: torch.Tensor,
    X: torch.Tensor,
    out: torch.Tensor,
    n_elements: int,
):
    """Compute Q4_K GEMM for token batch: X[T, N] x W[M, encoded_n]^T -> out[T, M]."""
    M = W.shape[0]
    T = X.shape[0]
    grid = (M, T, 1)
    matmul_q4_k_tokens_kernel[grid](
        W,
        X,
        out,
        n_elements,
        W.stride(0),
        X.stride(0),
        X.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE=256,
    )
    return out