# SPDX-License-Identifier: Apache-2.0
"""
Element-wise activation kernels.

Memory layout (all kernels):
  X:    (n_elements,)  contiguous  float16/bfloat16
  Out:  (n_elements,)  contiguous  float16/bfloat16

Tiling:
  Grid:  (ceil(n_elements / BLOCK_SIZE),)
  Each program: BLOCK_SIZE elements, 1D grid over flat tensor.

Register pressure: very low (< 16 fp32 regs). No ILP concerns, no fallback needed.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _silu_kernel(
    X,
    Out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(X + offsets, mask=mask).to(tl.float32)
    # Silu: x * sigmoid(x)
    # sigmoid(x) = 1 / (1 + exp(-x))
    y = x / (1.0 + tl.exp(-x))

    tl.store(Out + offsets, y.to(Out.dtype.element_ty), mask=mask)


@triton.jit
def _silu_and_mul_kernel(
    X,
    Out,
    stride_x,
    stride_out,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused gated-MLP activation: SiLU(x1) * x2.

    Memory layout:
      X:    (n_rows, 2 * n_cols)  row-major  float16/bfloat16
            First half (cols 0..n_cols-1): gate input (x1)
            Second half (cols n_cols..2*n_cols-1): up input (x2)
      Out:  (n_rows, n_cols)       row-major  float16/bfloat16

    Tiling:
      Grid:  (n_rows,)
      Each program: one row, BLOCK_SIZE=n_cols (next power of 2).
      BLOCK_SIZE typically 1024..14336 depending on intermediate_size.

    Register pressure: low (~8 fp32 regs: x1, x2, y + offsets). No fallback needed.
    """
    row_idx = tl.program_id(0)
    X_ptr = X + row_idx * stride_x
    Out_ptr = Out + row_idx * stride_out

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # X layout: [n_rows, 2 * n_cols]
    # We load the first half and second half
    x1 = tl.load(X_ptr + offsets, mask=mask).to(tl.float32)
    x2 = tl.load(X_ptr + n_cols + offsets, mask=mask).to(tl.float32)

    # Silu(x1) * x2
    y = (x1 / (1.0 + tl.exp(-x1))) * x2

    tl.store(Out_ptr + offsets, y.to(Out.dtype.element_ty), mask=mask)


def silu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _silu_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    # x: [..., 2*d]
    orig_shape = x.shape
    d = orig_shape[-1] // 2
    x_2d = x.view(-1, 2 * d)
    n_rows, _ = x_2d.shape

    out = torch.empty((n_rows, d), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = triton.next_power_of_2(d)
    grid = (n_rows,)

    _silu_and_mul_kernel[grid](
        x_2d, out, x_2d.stride(0), out.stride(0), n_rows, d, BLOCK_SIZE=BLOCK_SIZE
    )

    return out.view(*orig_shape[:-1], d)


@triton.jit
def _gelu_kernel(
    X,
    Out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(X + offsets, mask=mask).to(tl.float32)
    # Fast Gelu approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # For simplicity, using a basic version:
    y = 0.5 * x * (1.0 + tl.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

    tl.store(Out + offsets, y.to(Out.dtype.element_ty), mask=mask)


def gelu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out
