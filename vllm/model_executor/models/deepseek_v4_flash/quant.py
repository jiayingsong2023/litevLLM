# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def _validate_positive_block_size(block_size: int) -> None:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive; got {block_size}")


def _validate_float_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{name} must use a floating-point dtype; got {tensor.dtype}")


def decode_q8_0(
    values: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int = 32,
) -> torch.Tensor:
    """Decode GGML-style Q8_0 blocks from int8 payloads and per-block scales."""
    _validate_positive_block_size(block_size)
    if values.dtype != torch.int8:
        raise ValueError(f"Q8_0 values must be int8; got {values.dtype}")
    if values.ndim != 1:
        raise ValueError(f"Q8_0 values must be 1-D; got {values.ndim}-D")
    _validate_float_tensor("Q8_0 scales", scales)
    if scales.ndim != 1:
        raise ValueError(f"Q8_0 scales must be 1-D; got {scales.ndim}-D")
    if values.numel() % block_size != 0:
        raise ValueError(
            "Q8_0 values length must be divisible by block_size; "
            f"got {values.numel()} values and block_size={block_size}"
        )
    block_count = values.numel() // block_size
    if scales.numel() != block_count:
        raise ValueError(
            "Q8_0 scales length must match block count; "
            f"got {scales.numel()} scales for {block_count} blocks"
        )

    decoded = values.to(torch.float32).reshape(block_count, block_size)
    return (decoded * scales.to(torch.float32).reshape(block_count, 1)).reshape(-1)


def q8_0_dot(
    values: torch.Tensor,
    scales: torch.Tensor,
    vector: torch.Tensor,
    *,
    block_size: int = 32,
) -> torch.Tensor:
    """Return dot(decode_q8_0(values, scales), vector)."""
    _validate_float_tensor("Q8_0 dot vector", vector)
    if vector.ndim != 1:
        raise ValueError(f"Q8_0 dot vector must be 1-D; got {vector.ndim}-D")
    decoded = decode_q8_0(values, scales, block_size=block_size)
    if vector.numel() != decoded.numel():
        raise ValueError(
            "Q8_0 dot vector length must match decoded values length; "
            f"got vector length {vector.numel()} and decoded length {decoded.numel()}"
        )
    return torch.dot(decoded, vector.to(torch.float32))


def q8_0_matvec(
    values: torch.Tensor,
    scales: torch.Tensor,
    vector: torch.Tensor,
    *,
    block_size: int = 32,
) -> torch.Tensor:
    """Decode each Q8_0 matrix row and multiply by a dense vector."""
    _validate_positive_block_size(block_size)
    if values.dtype != torch.int8:
        raise ValueError(f"Q8_0 matrix values must be int8; got {values.dtype}")
    if values.ndim != 2:
        raise ValueError(f"Q8_0 matrix values must be 2-D; got {values.ndim}-D")
    _validate_float_tensor("Q8_0 matrix scales", scales)
    if scales.ndim != 2:
        raise ValueError(f"Q8_0 matrix scales must be 2-D; got {scales.ndim}-D")
    _validate_float_tensor("Q8_0 matvec vector", vector)
    if vector.ndim != 1:
        raise ValueError(f"Q8_0 matvec vector must be 1-D; got {vector.ndim}-D")
    rows, columns = values.shape
    if columns % block_size != 0:
        raise ValueError(
            "Q8_0 matrix column count must be divisible by block_size; "
            f"got {columns} columns and block_size={block_size}"
        )
    if vector.numel() != columns:
        raise ValueError(
            "Q8_0 matvec vector length must match matrix columns; "
            f"got vector length {vector.numel()} and columns {columns}"
        )
    blocks_per_row = columns // block_size
    if scales.shape != (rows, blocks_per_row):
        raise ValueError(
            "Q8_0 matrix scales shape must be (rows, blocks_per_row); "
            f"got {tuple(scales.shape)} for rows={rows}, "
            f"blocks_per_row={blocks_per_row}"
        )

    decoded = values.to(torch.float32).reshape(rows, blocks_per_row, block_size)
    row_scales = scales.to(torch.float32).reshape(rows, blocks_per_row, 1)
    decoded_matrix = (decoded * row_scales).reshape(rows, columns)
    return decoded_matrix.matmul(vector.to(torch.float32))


def q8_0_dequantize_reference(
    values: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int = 32,
) -> torch.Tensor:
    """Decode each Q8_0 matrix row for kernel reference tests."""
    _validate_positive_block_size(block_size)
    if values.ndim != 2:
        raise ValueError(f"Q8_0 matrix values must be 2-D; got {values.ndim}-D")
    if scales.ndim != 2:
        raise ValueError(f"Q8_0 matrix scales must be 2-D; got {scales.ndim}-D")
    rows, columns = values.shape
    if columns % block_size != 0:
        raise ValueError(
            "Q8_0 matrix column count must be divisible by block_size; "
            f"got {columns} columns and block_size={block_size}"
        )
    blocks_per_row = columns // block_size
    if scales.shape != (rows, blocks_per_row):
        raise ValueError(
            "Q8_0 matrix scales shape must be (rows, blocks_per_row); "
            f"got {tuple(scales.shape)} for rows={rows}, "
            f"blocks_per_row={blocks_per_row}"
        )
    decoded_rows = [
        decode_q8_0(row, row_scales, block_size=block_size)
        for row, row_scales in zip(values, scales, strict=True)
    ]
    return torch.stack(decoded_rows) if decoded_rows else values.new_empty(values.shape)


def q8_0_linear_reference(
    vector: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int = 32,
) -> torch.Tensor:
    """Reference Q8_0 matrix-vector product used by kernel tests."""
    return q8_0_matvec(values, scales, vector, block_size=block_size)


def _validate_two_bit_codes(name: str, codes: torch.Tensor) -> None:
    if codes.dtype != torch.uint8:
        raise ValueError(f"{name} codes must be uint8; got {codes.dtype}")
    if codes.numel() > 0 and torch.any(codes > 3):
        raise ValueError(f"{name} synthetic codes must be two-bit values in [0, 3]")


def decode_iq2_xxs_synthetic(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int = 32,
) -> torch.Tensor:
    """Decode deterministic synthetic IQ2_XXS blocks for reference tests.

    This helper intentionally does not model the final GGUF IQ2_XXS bit layout yet.
    The synthetic layout is one uint8 two-bit code per element with shape
    ``(blocks, block_size)``. Codes map to signed levels ``[-3, -1, 1, 3]`` and
    each block has one floating-point scale.
    """
    _validate_positive_block_size(block_size)
    _validate_two_bit_codes("IQ2_XXS", codes)
    if codes.ndim != 2:
        raise ValueError(f"IQ2_XXS synthetic codes must be 2-D; got {codes.ndim}-D")
    _validate_float_tensor("IQ2_XXS synthetic scales", scales)
    if scales.ndim != 1:
        raise ValueError(f"IQ2_XXS synthetic scales must be 1-D; got {scales.ndim}-D")
    if codes.shape[1] != block_size:
        raise ValueError(
            "IQ2_XXS synthetic codes second dimension must match block_size; "
            f"got {codes.shape[1]} and block_size={block_size}"
        )
    if scales.numel() != codes.shape[0]:
        raise ValueError(
            "IQ2_XXS synthetic scales length must match block count; "
            f"got {scales.numel()} scales for {codes.shape[0]} blocks"
        )

    levels = codes.to(torch.float32) * 2.0 - 3.0
    return levels * scales.to(torch.float32).reshape(codes.shape[0], 1)


def decode_q2_k_synthetic(
    codes: torch.Tensor,
    scales: torch.Tensor,
    mins: torch.Tensor,
    *,
    group_size: int = 16,
) -> torch.Tensor:
    """Decode deterministic synthetic Q2_K super-block groups.

    This helper intentionally uses an unpacked synthetic layout until the real
    GGUF Q2_K packed bit layout is bound. Codes are uint8 two-bit values with
    shape ``(super_blocks, groups, group_size)``. Every group has one scale and
    one additive minimum with shape ``(super_blocks, groups)``.
    """
    _validate_positive_block_size(group_size)
    _validate_two_bit_codes("Q2_K", codes)
    if codes.ndim != 3:
        raise ValueError(f"Q2_K synthetic codes must be 3-D; got {codes.ndim}-D")
    _validate_float_tensor("Q2_K synthetic scales", scales)
    _validate_float_tensor("Q2_K synthetic mins", mins)
    if scales.ndim != 2:
        raise ValueError(f"Q2_K synthetic scales must be 2-D; got {scales.ndim}-D")
    if mins.ndim != 2:
        raise ValueError(f"Q2_K synthetic mins must be 2-D; got {mins.ndim}-D")
    if codes.shape[2] != group_size:
        raise ValueError(
            "Q2_K synthetic codes third dimension must match group_size; "
            f"got {codes.shape[2]} and group_size={group_size}"
        )
    group_shape = codes.shape[:2]
    if scales.shape != group_shape:
        raise ValueError(
            "Q2_K synthetic scales shape must match (super_blocks, groups); "
            f"got {tuple(scales.shape)} for {tuple(group_shape)}"
        )
    if mins.shape != group_shape:
        raise ValueError(
            "Q2_K synthetic mins shape must match (super_blocks, groups); "
            f"got {tuple(mins.shape)} for {tuple(group_shape)}"
        )

    return (
        codes.to(torch.float32) * scales.to(torch.float32).unsqueeze(-1)
        + mins.to(torch.float32).unsqueeze(-1)
    )
