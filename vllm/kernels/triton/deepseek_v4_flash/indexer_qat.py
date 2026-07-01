# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _indexer_hadamard128_kernel(
    row_ptr,
    scratch_ptr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    # Memory layout:
    # - row_ptr is one contiguous fp32/fp16/bf16 row [128].
    # - scratch_ptr is one contiguous fp32 row [128].
    # Tiling:
    # - one Triton program computes one Hadamard output lane.
    # - each program loads all 128 input values, applies the lane's sign
    #   pattern, reduces to fp32, and stores one scaled output.
    lane = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    bits = offsets & lane
    bits = bits ^ (bits >> 4)
    bits = bits ^ (bits >> 2)
    bits = bits ^ (bits >> 1)
    signs = tl.where((bits & 1) == 0, 1.0, -1.0)
    values = tl.load(row_ptr + offsets).to(tl.float32)
    total = tl.sum(values * signs, axis=0)
    tl.store(scratch_ptr + lane, total * 0.08838834764831845)


@triton.jit
def _indexer_e2m1_roundtrip_kernel(
    scratch_ptr,
    output_ptr,
    BLOCK_WIDTH: tl.constexpr,
) -> None:
    # Memory layout:
    # - scratch_ptr is the Hadamard output [128] split into four 32-value
    #   quantization blocks.
    # - output_ptr is the dequantized fp32 output [128].
    # Tiling:
    # - one Triton program handles one 32-value E2M1 block.
    # - the program computes the block scale, clamps to the E2M1 range,
    #   applies nearest-value dequantization, and stores 32 values.
    block = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_WIDTH)
    base = block * BLOCK_WIDTH
    values = tl.load(scratch_ptr + base + offsets).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), axis=0), 7.052966104933725e-38)
    scale = tl.exp2(tl.ceil(tl.log2(amax / 6.0)))
    quant_input = tl.minimum(tl.maximum(values / scale, -6.0), 6.0)
    signs = tl.where(quant_input < 0.0, -1.0, 1.0)
    abs_values = tl.abs(quant_input)

    rounded = tl.full((BLOCK_WIDTH,), 0.0, tl.float32)
    rounded = tl.where(abs_values > 0.25, 0.5, rounded)
    rounded = tl.where(abs_values >= 0.75, 1.0, rounded)
    rounded = tl.where(abs_values > 1.25, 1.5, rounded)
    rounded = tl.where(abs_values >= 1.75, 2.0, rounded)
    rounded = tl.where(abs_values > 2.5, 3.0, rounded)
    rounded = tl.where(abs_values >= 3.5, 4.0, rounded)
    rounded = tl.where(abs_values > 5.0, 6.0, rounded)
    tl.store(output_ptr + base + offsets, signs * rounded * scale)


def deepseek_v4_indexer_qat(row: torch.Tensor) -> torch.Tensor:
    """Triton implementation of DeepSeek V4 Flash indexer QAT for one row."""
    if row.shape != (128,):
        raise ValueError(
            f"indexer QAT row must have shape (128,); got {tuple(row.shape)}"
        )
    if not row.is_cuda:
        raise ValueError("indexer QAT row must be a CUDA tensor")

    row_f32 = row.to(torch.float32).contiguous()
    scratch = torch.empty((128,), dtype=torch.float32, device=row.device)
    output = torch.empty((128,), dtype=torch.float32, device=row.device)
    _indexer_hadamard128_kernel[(128,)](
        row_f32,
        scratch,
        BLOCK_SIZE=128,
        num_warps=4,
    )
    _indexer_e2m1_roundtrip_kernel[(4,)](
        scratch,
        output,
        BLOCK_WIDTH=32,
        num_warps=1,
    )
    return output
