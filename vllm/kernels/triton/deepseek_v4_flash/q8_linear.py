# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.model_executor.models.deepseek_v4_flash.quant import q8_0_linear_reference


def q8_0_linear(
    vector: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int = 32,
) -> torch.Tensor:
    """Correctness-first Q8_0 matrix-vector product.

    Memory layout:
    - vector is a contiguous [K] fp32/fp16/bf16 vector.
    - values is a contiguous [N, K] int8 matrix.
    - scales is a contiguous [N, K / block_size] floating-point matrix.
    - output is [N] fp32.

    Tiling:
    - This first implementation uses the PyTorch reference path as the oracle.
    - A future Triton program should tile by output row and K block, load one
      scale per Q8_0 block, and reduce each row into one fp32 output element.
    """
    if vector.ndim != 1:
        raise ValueError(f"Q8_0 linear vector must be 1-D; got {vector.ndim}-D")
    if values.ndim != 2:
        raise ValueError(f"Q8_0 linear values must be 2-D; got {values.ndim}-D")
    if scales.ndim != 2:
        raise ValueError(f"Q8_0 linear scales must be 2-D; got {scales.ndim}-D")
    if not vector.is_contiguous():
        vector = vector.contiguous()
    if not values.is_contiguous():
        values = values.contiguous()
    if not scales.is_contiguous():
        scales = scales.contiguous()
    return q8_0_linear_reference(
        vector,
        values,
        scales,
        block_size=block_size,
    )
