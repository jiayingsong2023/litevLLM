# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.model_executor.models.deepseek_v4_flash.quant import q8_0_linear_reference
from vllm.triton_utils import tl, triton


@triton.jit
def _q8_0_raw_matvec_kernel(
    raw_ptr,
    hidden_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_ROW: tl.constexpr,
    Q8_BLOCK_BYTES: tl.constexpr,
) -> None:
    # Memory layout:
    # - raw_ptr stores row-major GGUF Q8_0 blocks:
    #   each block is [2-byte little-endian fp16 scale][32 int8 values].
    # - hidden_ptr is a contiguous fp32/fp16/bf16 vector [columns].
    # Tiling:
    # - one Triton program computes one output row.
    # - the program loops over Q8_0 blocks and reduces 32 values per block.
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    total = tl.full((), 0.0, tl.float32)
    row_offset = row * BLOCKS_PER_ROW * Q8_BLOCK_BYTES
    for block_idx in tl.static_range(0, BLOCKS_PER_ROW):
        block_offset = row_offset + block_idx * Q8_BLOCK_BYTES
        scale_lo = tl.load(raw_ptr + block_offset).to(tl.uint32)
        scale_hi = tl.load(raw_ptr + block_offset + 1).to(tl.uint32)
        scale_bits = ((scale_hi << 8) | scale_lo).to(tl.uint16)
        scale = scale_bits.to(tl.float16, bitcast=True).to(tl.float32)
        values = tl.load(raw_ptr + block_offset + 2 + offsets).to(tl.int8).to(
            tl.float32
        )
        hidden = tl.load(hidden_ptr + block_idx * BLOCK_SIZE + offsets).to(tl.float32)
        total += tl.sum(values * scale * hidden, axis=0)
    tl.store(out_ptr + row, total)


@triton.jit
def _q8_0_raw_gate_up_activation_kernel(
    gate_raw_ptr,
    up_raw_ptr,
    hidden_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_ROW: tl.constexpr,
    Q8_BLOCK_BYTES: tl.constexpr,
) -> None:
    # Memory layout:
    # - gate_raw_ptr and up_raw_ptr store row-major GGUF Q8_0 blocks:
    #   each block is [2-byte little-endian fp16 scale][32 int8 values].
    # - hidden_ptr is a contiguous vector [columns].
    # - out_ptr is activated intermediate [rows].
    # Tiling:
    # - one Triton program computes one shared-expert intermediate row.
    # - the program decodes gate and up Q8_0 rows against the same hidden
    #   vector, clamps DS4 SwiGLU inputs, and stores silu(gate) * up.
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    gate_total = tl.full((), 0.0, tl.float32)
    up_total = tl.full((), 0.0, tl.float32)
    row_offset = row * BLOCKS_PER_ROW * Q8_BLOCK_BYTES
    for block_idx in tl.static_range(0, BLOCKS_PER_ROW):
        block_offset = row_offset + block_idx * Q8_BLOCK_BYTES

        gate_scale_lo = tl.load(gate_raw_ptr + block_offset).to(tl.uint32)
        gate_scale_hi = tl.load(gate_raw_ptr + block_offset + 1).to(tl.uint32)
        gate_scale_bits = ((gate_scale_hi << 8) | gate_scale_lo).to(tl.uint16)
        gate_scale = gate_scale_bits.to(tl.float16, bitcast=True).to(tl.float32)
        gate_values = tl.load(gate_raw_ptr + block_offset + 2 + offsets).to(
            tl.int8
        ).to(tl.float32)

        up_scale_lo = tl.load(up_raw_ptr + block_offset).to(tl.uint32)
        up_scale_hi = tl.load(up_raw_ptr + block_offset + 1).to(tl.uint32)
        up_scale_bits = ((up_scale_hi << 8) | up_scale_lo).to(tl.uint16)
        up_scale = up_scale_bits.to(tl.float16, bitcast=True).to(tl.float32)
        up_values = tl.load(up_raw_ptr + block_offset + 2 + offsets).to(tl.int8).to(
            tl.float32
        )

        hidden = tl.load(hidden_ptr + block_idx * BLOCK_SIZE + offsets).to(tl.float32)
        gate_total += tl.sum(gate_values * gate_scale * hidden, axis=0)
        up_total += tl.sum(up_values * up_scale * hidden, axis=0)

    gate_clamped = tl.minimum(gate_total, 10.0)
    up_clamped = tl.minimum(tl.maximum(up_total, -10.0), 10.0)
    activated = gate_clamped / (1.0 + tl.exp(-gate_clamped)) * up_clamped
    tl.store(out_ptr + row, activated)


def _q8_0_raw_matvec_triton_cuda(
    raw_payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
    columns: int,
    block_size: int,
) -> torch.Tensor:
    if block_size != 32:
        raise ValueError("DeepSeek V4 raw Q8_0 kernel requires block_size=32")
    blocks_per_row = columns // block_size
    q8_block_bytes = 2 + block_size
    expected_bytes = rows * blocks_per_row * q8_block_bytes
    if raw_payload.numel() != expected_bytes:
        raise ValueError(
            "raw Q8_0 payload byte length must match rows * blocks_per_row; "
            f"got {raw_payload.numel()} bytes, expected {expected_bytes}"
        )
    out = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    _q8_0_raw_matvec_kernel[(rows,)](
        raw_payload.contiguous(),
        hidden.contiguous(),
        out,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_ROW=blocks_per_row,
        Q8_BLOCK_BYTES=q8_block_bytes,
        num_warps=1,
    )
    return out


def _q8_0_raw_gate_up_activation_triton_cuda(
    gate_raw_payload: torch.Tensor,
    up_raw_payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
    columns: int,
    block_size: int,
) -> torch.Tensor:
    if block_size != 32:
        raise ValueError("DeepSeek V4 raw Q8_0 kernel requires block_size=32")
    blocks_per_row = columns // block_size
    q8_block_bytes = 2 + block_size
    expected_bytes = rows * blocks_per_row * q8_block_bytes
    if gate_raw_payload.numel() != expected_bytes:
        raise ValueError(
            "raw Q8_0 gate payload byte length must match rows * blocks_per_row; "
            f"got {gate_raw_payload.numel()} bytes, expected {expected_bytes}"
        )
    if up_raw_payload.numel() != expected_bytes:
        raise ValueError(
            "raw Q8_0 up payload byte length must match rows * blocks_per_row; "
            f"got {up_raw_payload.numel()} bytes, expected {expected_bytes}"
        )
    out = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    _q8_0_raw_gate_up_activation_kernel[(rows,)](
        gate_raw_payload.contiguous(),
        up_raw_payload.contiguous(),
        hidden.contiguous(),
        out,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_ROW=blocks_per_row,
        Q8_BLOCK_BYTES=q8_block_bytes,
        num_warps=1,
    )
    return out


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


def q8_0_raw_linear(
    raw_payload: torch.Tensor,
    vector: torch.Tensor,
    *,
    rows: int,
    columns: int,
    block_size: int = 32,
) -> torch.Tensor:
    """Q8_0 matrix-vector product over raw GGUF block bytes.

    Memory layout:
    - raw_payload is a contiguous uint8 tensor containing row-major GGUF Q8_0
      blocks: [fp16 scale][32 int8 values] per block.
    - vector is a contiguous [columns] fp32/fp16/bf16 vector.
    - output is [rows] fp32.

    Tiling:
    - GPU path launches one Triton program per row and reduces one 32-value
      Q8_0 block per loop iteration.
    """
    if vector.ndim != 1:
        raise ValueError(f"raw Q8_0 vector must be 1-D; got {vector.ndim}-D")
    if raw_payload.ndim != 1:
        raise ValueError(
            f"raw Q8_0 payload must be 1-D; got {raw_payload.ndim}-D"
        )
    if raw_payload.dtype != torch.uint8:
        raise ValueError(f"raw Q8_0 payload must be uint8; got {raw_payload.dtype}")
    if rows <= 0:
        raise ValueError(f"raw Q8_0 rows must be positive; got {rows}")
    if columns <= 0 or columns % block_size != 0:
        raise ValueError(
            "raw Q8_0 columns must be positive and divisible by block_size; "
            f"got columns={columns}, block_size={block_size}"
        )
    if vector.numel() != columns:
        raise ValueError(
            f"raw Q8_0 vector length must match columns; got {vector.numel()} "
            f"and {columns}"
        )
    if not raw_payload.is_cuda or not vector.is_cuda:
        raise ValueError("raw Q8_0 linear inputs must be CUDA tensors")
    return _q8_0_raw_matvec_triton_cuda(
        raw_payload,
        vector.to(torch.float32),
        rows=rows,
        columns=columns,
        block_size=block_size,
    )


def q8_0_raw_gate_up_activation(
    gate_raw_payload: torch.Tensor,
    up_raw_payload: torch.Tensor,
    vector: torch.Tensor,
    *,
    rows: int,
    columns: int,
    block_size: int = 32,
) -> torch.Tensor:
    """Fused Q8_0 gate/up activation over raw GGUF block bytes.

    Memory layout:
    - gate/up raw payloads are row-major GGUF Q8_0 blocks:
      [fp16 scale][32 int8 values] per block.
    - vector is a contiguous [columns] hidden vector.
    - output is [rows] fp32 activated intermediate.

    Tiling:
    - one Triton program per row decodes both gate and up rows, reduces both
      dot products, applies DS4 clamp and SiLU, then stores one value.
    """
    if vector.ndim != 1:
        raise ValueError(f"raw Q8_0 vector must be 1-D; got {vector.ndim}-D")
    if gate_raw_payload.ndim != 1 or up_raw_payload.ndim != 1:
        raise ValueError("raw Q8_0 gate/up payloads must be 1-D")
    if gate_raw_payload.dtype != torch.uint8 or up_raw_payload.dtype != torch.uint8:
        raise ValueError("raw Q8_0 gate/up payloads must be uint8")
    if rows <= 0:
        raise ValueError(f"raw Q8_0 rows must be positive; got {rows}")
    if columns <= 0 or columns % block_size != 0:
        raise ValueError(
            "raw Q8_0 columns must be positive and divisible by block_size; "
            f"got columns={columns}, block_size={block_size}"
        )
    if vector.numel() != columns:
        raise ValueError(
            f"raw Q8_0 vector length must match columns; got {vector.numel()} "
            f"and {columns}"
        )
    if not gate_raw_payload.is_cuda or not up_raw_payload.is_cuda or not vector.is_cuda:
        raise ValueError("raw Q8_0 gate/up activation inputs must be CUDA tensors")
    return _q8_0_raw_gate_up_activation_triton_cuda(
        gate_raw_payload,
        up_raw_payload,
        vector.to(torch.float32),
        rows=rows,
        columns=columns,
        block_size=block_size,
    )
