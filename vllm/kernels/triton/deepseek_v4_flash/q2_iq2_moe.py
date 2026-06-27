# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.model_executor.models.deepseek_v4_flash.quant import (
    iq2_xxs_lookup_tensors,
    iq2_xxs_matrix_from_gguf_payload,
    q2_k_matrix_from_gguf_payload,
)
from vllm.triton_utils import tl, triton

_Q2_K_BLOCK_BYTES = 84
_IQ2_XXS_BLOCK_BYTES = 66
_GGUF_BLOCK_COLUMNS = 256
_DS4_SWIGLU_CLAMP = 10.0
_IQ2_XXS_LOOKUP_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _payload_bytes(payload: torch.Tensor) -> bytes:
    if payload.dtype != torch.uint8:
        raise ValueError(f"payload must be torch.uint8; got {payload.dtype}")
    if not payload.is_cuda:
        raise ValueError("payload must be a CUDA tensor")
    return bytes(payload.detach().cpu().contiguous().tolist())


def _validate_payload(
    payload: torch.Tensor,
    *,
    rows: int,
    columns: int,
    block_bytes: int,
) -> None:
    if payload.dtype != torch.uint8:
        raise ValueError(f"payload must be torch.uint8; got {payload.dtype}")
    if not payload.is_cuda:
        raise ValueError("payload must be a CUDA tensor")
    if rows <= 0:
        raise ValueError(f"rows must be positive; got {rows}")
    if columns <= 0:
        raise ValueError(f"columns must be positive; got {columns}")
    if columns % _GGUF_BLOCK_COLUMNS != 0:
        raise ValueError(
            f"columns must be a multiple of 256 for GGUF Q2/IQ2 blocks; got {columns}"
        )
    expected_bytes = rows * (columns // _GGUF_BLOCK_COLUMNS) * block_bytes
    if payload.numel() != expected_bytes:
        raise ValueError(
            "payload byte length does not match matrix shape; "
            f"got {payload.numel()} and expected {expected_bytes}"
        )


def _validate_hidden(hidden: torch.Tensor, *, columns: int) -> torch.Tensor:
    if not hidden.is_cuda:
        raise ValueError("hidden must be a CUDA tensor")
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D; got {hidden.ndim}-D")
    if hidden.numel() != columns:
        raise ValueError(
            "hidden length must match matrix columns; "
            f"got hidden={hidden.numel()} and columns={columns}"
        )
    return hidden.to(torch.float32)


def _q8_k_roundtrip_cuda(row: torch.Tensor, *, block_size: int = 256) -> torch.Tensor:
    if row.ndim != 1:
        raise ValueError(f"Q8_K row must be 1-D; got {row.ndim}-D")
    if row.numel() % block_size != 0:
        raise ValueError(
            "Q8_K row length must be divisible by block_size; "
            f"got {row.numel()} and {block_size}"
        )
    blocks = row.to(torch.float32).reshape(-1, block_size)
    max_indices = torch.argmax(blocks.abs(), dim=1)
    signed_max = blocks.gather(1, max_indices.reshape(-1, 1)).reshape(-1, 1)
    nonzero = signed_max.abs() > 0.0
    iscale = torch.where(
        nonzero,
        torch.full_like(signed_max, -127.0) / signed_max,
        torch.zeros_like(signed_max),
    )
    quantized = torch.round(blocks * iscale).clamp(min=-128, max=127)
    scale = torch.where(nonzero, 1.0 / iscale, torch.zeros_like(iscale))
    return (quantized * scale).reshape_as(row).to(torch.float32)


def _q8_k_roundtrip_rows_cuda(
    rows: torch.Tensor,
    *,
    block_size: int = 256,
) -> torch.Tensor:
    if rows.ndim != 2:
        raise ValueError(f"Q8_K rows must be 2-D; got {rows.ndim}-D")
    if rows.shape[1] % block_size != 0:
        raise ValueError(
            "Q8_K row width must be divisible by block_size; "
            f"got {rows.shape[1]} and {block_size}"
        )
    blocks = rows.to(torch.float32).reshape(-1, block_size)
    max_indices = torch.argmax(blocks.abs(), dim=1)
    signed_max = blocks.gather(1, max_indices.reshape(-1, 1))
    nonzero = signed_max.abs() > 0.0
    iscale = torch.where(
        nonzero,
        torch.full_like(signed_max, -127.0) / signed_max,
        torch.zeros_like(signed_max),
    )
    quantized = torch.round(blocks * iscale).clamp(min=-128, max=127)
    scale = torch.where(nonzero, 1.0 / iscale, torch.zeros_like(iscale))
    return (quantized * scale).reshape_as(rows).to(torch.float32)


# GGUF Q2_K row layout: each 84-byte block stores one 256-value row here.
# Bytes 0..15 are packed scale/min nibbles, bytes 16..79 are 64 quant bytes,
# byte pairs 80..81 and 82..83 are fp16 d and dmin. One Triton program handles
# one matrix row and uses 256 lanes to decode all values, multiply by the
# contiguous hidden vector, reduce to fp32, and store one output element.
@triton.jit
def _q2_k_matvec_kernel(
    payload_ptr,
    payload_half_ptr,
    hidden_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    HALF_WORDS_PER_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    group = offsets // 16
    q_half = offsets // 128
    rem = offsets - q_half * 128
    shift = (rem // 32) * 2
    q_byte_index = q_half * 32 + (rem - (rem // 32) * 32)

    payload_base = row * BLOCK_BYTES
    scale_bytes = tl.load(payload_ptr + payload_base + group).to(tl.uint32)
    q_bytes = tl.load(payload_ptr + payload_base + 16 + q_byte_index).to(tl.uint32)
    codes = ((q_bytes >> shift) & 0x03).to(tl.float32)

    scale_lows = (scale_bytes & 0x0F).to(tl.float32)
    scale_highs = (scale_bytes >> 4).to(tl.float32)
    d = tl.load(payload_half_ptr + row * HALF_WORDS_PER_BLOCK + 40).to(tl.float32)
    dmin = tl.load(payload_half_ptr + row * HALF_WORDS_PER_BLOCK + 41).to(tl.float32)

    decoded = d * scale_lows * codes - dmin * scale_highs
    hidden = tl.load(hidden_ptr + offsets).to(tl.float32)
    total = tl.sum(decoded * hidden, axis=0)
    tl.store(output_ptr + row, total)


# GGUF Q2_K multi-block row layout: each matrix row stores COLUMNS_BLOCKS
# consecutive 84-byte Q2_K blocks. One Triton program handles one output row,
# loops over 256-value K blocks, accumulates fp32 dot products, and stores one
# output element. This preserves the same per-block decode as the single-block
# kernel while supporting real DeepSeek down projections such as columns=2048.
@triton.jit
def _q2_k_matvec_multiblock_kernel(
    payload_ptr,
    payload_half_ptr,
    hidden_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    HALF_WORDS_PER_BLOCK: tl.constexpr,
    COLUMNS_BLOCKS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    group = offsets // 16
    q_half = offsets // 128
    rem = offsets - q_half * 128
    shift = (rem // 32) * 2
    q_byte_index = q_half * 32 + (rem - (rem // 32) * 32)

    total = tl.full((), 0.0, tl.float32)
    for block_idx in tl.static_range(0, COLUMNS_BLOCKS):
        payload_base = (row * COLUMNS_BLOCKS + block_idx) * BLOCK_BYTES
        scale_bytes = tl.load(payload_ptr + payload_base + group).to(tl.uint32)
        q_bytes = tl.load(payload_ptr + payload_base + 16 + q_byte_index).to(tl.uint32)
        codes = ((q_bytes >> shift) & 0x03).to(tl.float32)

        scale_lows = (scale_bytes & 0x0F).to(tl.float32)
        scale_highs = (scale_bytes >> 4).to(tl.float32)
        half_base = (row * COLUMNS_BLOCKS + block_idx) * HALF_WORDS_PER_BLOCK
        d = tl.load(payload_half_ptr + half_base + 40).to(tl.float32)
        dmin = tl.load(payload_half_ptr + half_base + 41).to(tl.float32)

        decoded = d * scale_lows * codes - dmin * scale_highs
        hidden = tl.load(hidden_ptr + block_idx * BLOCK_SIZE + offsets).to(tl.float32)
        total += tl.sum(decoded * hidden, axis=0)
    tl.store(output_ptr + row, total)


def _q2_k_matvec_triton_cuda(
    payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
) -> torch.Tensor:
    output = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    payload_contiguous = payload.contiguous()
    hidden_contiguous = hidden.contiguous()
    _q2_k_matvec_kernel[(rows,)](
        payload_contiguous,
        payload_contiguous.view(torch.float16),
        hidden_contiguous,
        output,
        BLOCK_SIZE=_GGUF_BLOCK_COLUMNS,
        BLOCK_BYTES=_Q2_K_BLOCK_BYTES,
        HALF_WORDS_PER_BLOCK=_Q2_K_BLOCK_BYTES // 2,
        num_warps=8,
    )
    return output


def _q2_k_matvec_multiblock_triton_cuda(
    payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
    columns_blocks: int,
) -> torch.Tensor:
    output = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    payload_contiguous = payload.contiguous()
    hidden_contiguous = hidden.contiguous()
    _q2_k_matvec_multiblock_kernel[(rows,)](
        payload_contiguous,
        payload_contiguous.view(torch.float16),
        hidden_contiguous,
        output,
        BLOCK_SIZE=_GGUF_BLOCK_COLUMNS,
        BLOCK_BYTES=_Q2_K_BLOCK_BYTES,
        HALF_WORDS_PER_BLOCK=_Q2_K_BLOCK_BYTES // 2,
        COLUMNS_BLOCKS=columns_blocks,
        num_warps=8,
    )
    return output


# GGUF IQ2_XXS row layout: each 66-byte block stores one 256-value row here.
# Bytes 0..1 are fp16 d. Bytes 2..65 are eight 8-byte groups; each group has
# four grid-index bytes followed by one little-endian u32 sign/scale word.
# One Triton program handles one matrix row. It uses 256 lanes arranged as
# 8 groups x 4 grid bytes x 8 grid values, decodes all values, multiplies by
# contiguous hidden lanes, reduces in fp32, and stores one output element.
@triton.jit
def _iq2_xxs_matvec_kernel(
    payload_ptr,
    payload_half_ptr,
    hidden_ptr,
    ksigns_ptr,
    grid_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    HALF_WORDS_PER_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    group = offsets // 32
    group_lane = offsets - group * 32
    grid_byte_slot = group_lane // 8
    grid_lane = group_lane - grid_byte_slot * 8

    payload_base = row * BLOCK_BYTES
    group_base = payload_base + 2 + group * 8
    q_grid_bytes = tl.load(payload_ptr + group_base + grid_byte_slot).to(tl.uint32)

    word_byte0 = tl.load(payload_ptr + group_base + 4).to(tl.uint32)
    word_byte1 = tl.load(payload_ptr + group_base + 5).to(tl.uint32)
    word_byte2 = tl.load(payload_ptr + group_base + 6).to(tl.uint32)
    word_byte3 = tl.load(payload_ptr + group_base + 7).to(tl.uint32)
    q_sign_scale = (
        word_byte0 | (word_byte1 << 8) | (word_byte2 << 16) | (word_byte3 << 24)
    )

    sign_shift = grid_byte_slot * 7
    sign_index = (q_sign_scale >> sign_shift) & 0x7F
    packed_signs = tl.load(ksigns_ptr + sign_index).to(tl.uint32)
    sign_bits = (packed_signs >> grid_lane) & 0x01
    signs = 1.0 - sign_bits.to(tl.float32) * 2.0

    scale_code = (q_sign_scale >> 28).to(tl.float32)
    d = tl.load(payload_half_ptr + row * HALF_WORDS_PER_BLOCK).to(tl.float32)
    block_scales = d * (0.5 + scale_code) * 0.25

    grid = tl.load(grid_ptr + q_grid_bytes * 8 + grid_lane).to(tl.float32)
    hidden = tl.load(hidden_ptr + offsets).to(tl.float32)
    total = tl.sum(block_scales * grid * signs * hidden, axis=0)
    tl.store(output_ptr + row, total)


# GGUF IQ2_XXS fused gate/up row layout: gate and up each use the 66-byte
# one-row block described above. One Triton program handles one output row,
# decodes both payload rows against the same contiguous 256-lane hidden vector,
# reduces two fp32 dot products, applies SiLU to the gate dot, multiplies by
# the up dot, and stores one activated intermediate value.
@triton.jit
def _iq2_xxs_gate_up_activation_kernel(
    gate_payload_ptr,
    gate_payload_half_ptr,
    up_payload_ptr,
    up_payload_half_ptr,
    hidden_ptr,
    ksigns_ptr,
    grid_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    HALF_WORDS_PER_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    group = offsets // 32
    group_lane = offsets - group * 32
    grid_byte_slot = group_lane // 8
    grid_lane = group_lane - grid_byte_slot * 8

    payload_base = row * BLOCK_BYTES
    group_offset = 2 + group * 8
    metadata_offset = payload_base + group_offset

    gate_grid_bytes = tl.load(gate_payload_ptr + metadata_offset + grid_byte_slot).to(
        tl.uint32
    )
    up_grid_bytes = tl.load(up_payload_ptr + metadata_offset + grid_byte_slot).to(
        tl.uint32
    )

    gate_word_byte0 = tl.load(gate_payload_ptr + metadata_offset + 4).to(tl.uint32)
    gate_word_byte1 = tl.load(gate_payload_ptr + metadata_offset + 5).to(tl.uint32)
    gate_word_byte2 = tl.load(gate_payload_ptr + metadata_offset + 6).to(tl.uint32)
    gate_word_byte3 = tl.load(gate_payload_ptr + metadata_offset + 7).to(tl.uint32)
    gate_sign_scale = (
        gate_word_byte0
        | (gate_word_byte1 << 8)
        | (gate_word_byte2 << 16)
        | (gate_word_byte3 << 24)
    )

    up_word_byte0 = tl.load(up_payload_ptr + metadata_offset + 4).to(tl.uint32)
    up_word_byte1 = tl.load(up_payload_ptr + metadata_offset + 5).to(tl.uint32)
    up_word_byte2 = tl.load(up_payload_ptr + metadata_offset + 6).to(tl.uint32)
    up_word_byte3 = tl.load(up_payload_ptr + metadata_offset + 7).to(tl.uint32)
    up_sign_scale = (
        up_word_byte0
        | (up_word_byte1 << 8)
        | (up_word_byte2 << 16)
        | (up_word_byte3 << 24)
    )

    sign_shift = grid_byte_slot * 7
    gate_sign_index = (gate_sign_scale >> sign_shift) & 0x7F
    gate_packed_signs = tl.load(ksigns_ptr + gate_sign_index).to(tl.uint32)
    gate_sign_bits = (gate_packed_signs >> grid_lane) & 0x01
    gate_signs = 1.0 - gate_sign_bits.to(tl.float32) * 2.0

    up_sign_index = (up_sign_scale >> sign_shift) & 0x7F
    up_packed_signs = tl.load(ksigns_ptr + up_sign_index).to(tl.uint32)
    up_sign_bits = (up_packed_signs >> grid_lane) & 0x01
    up_signs = 1.0 - up_sign_bits.to(tl.float32) * 2.0

    gate_scale_code = (gate_sign_scale >> 28).to(tl.float32)
    up_scale_code = (up_sign_scale >> 28).to(tl.float32)
    gate_d = tl.load(gate_payload_half_ptr + row * HALF_WORDS_PER_BLOCK).to(tl.float32)
    up_d = tl.load(up_payload_half_ptr + row * HALF_WORDS_PER_BLOCK).to(tl.float32)
    gate_block_scales = gate_d * (0.5 + gate_scale_code) * 0.25
    up_block_scales = up_d * (0.5 + up_scale_code) * 0.25

    gate_grid = tl.load(grid_ptr + gate_grid_bytes * 8 + grid_lane).to(tl.float32)
    up_grid = tl.load(grid_ptr + up_grid_bytes * 8 + grid_lane).to(tl.float32)
    hidden = tl.load(hidden_ptr + offsets).to(tl.float32)
    gate_total = tl.sum(gate_block_scales * gate_grid * gate_signs * hidden, axis=0)
    up_total = tl.sum(up_block_scales * up_grid * up_signs * hidden, axis=0)
    gate_clamped = tl.minimum(gate_total, 10.0)
    up_clamped = tl.minimum(tl.maximum(up_total, -10.0), 10.0)
    activated = gate_clamped / (1.0 + tl.exp(-gate_clamped)) * up_clamped
    tl.store(output_ptr + row, activated)


# GGUF IQ2_XXS multi-block fused gate/up layout: each matrix row stores
# COLUMNS_BLOCKS consecutive 66-byte IQ2_XXS blocks. One Triton program handles
# one output row, loops across the row's 256-value K blocks, accumulates gate
# and up dot products in fp32, then stores silu(gate_total) * up_total.
@triton.jit
def _iq2_xxs_gate_up_activation_multiblock_kernel(
    gate_payload_ptr,
    gate_payload_half_ptr,
    up_payload_ptr,
    up_payload_half_ptr,
    hidden_ptr,
    ksigns_ptr,
    grid_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    HALF_WORDS_PER_BLOCK: tl.constexpr,
    COLUMNS_BLOCKS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    group = offsets // 32
    group_lane = offsets - group * 32
    grid_byte_slot = group_lane // 8
    grid_lane = group_lane - grid_byte_slot * 8

    gate_total = tl.full((), 0.0, tl.float32)
    up_total = tl.full((), 0.0, tl.float32)
    for block_idx in tl.static_range(0, COLUMNS_BLOCKS):
        payload_base = (row * COLUMNS_BLOCKS + block_idx) * BLOCK_BYTES
        group_offset = 2 + group * 8
        metadata_offset = payload_base + group_offset

        gate_grid_bytes = tl.load(
            gate_payload_ptr + metadata_offset + grid_byte_slot
        ).to(tl.uint32)
        up_grid_bytes = tl.load(up_payload_ptr + metadata_offset + grid_byte_slot).to(
            tl.uint32
        )

        gate_word_byte0 = tl.load(gate_payload_ptr + metadata_offset + 4).to(tl.uint32)
        gate_word_byte1 = tl.load(gate_payload_ptr + metadata_offset + 5).to(tl.uint32)
        gate_word_byte2 = tl.load(gate_payload_ptr + metadata_offset + 6).to(tl.uint32)
        gate_word_byte3 = tl.load(gate_payload_ptr + metadata_offset + 7).to(tl.uint32)
        gate_sign_scale = (
            gate_word_byte0
            | (gate_word_byte1 << 8)
            | (gate_word_byte2 << 16)
            | (gate_word_byte3 << 24)
        )

        up_word_byte0 = tl.load(up_payload_ptr + metadata_offset + 4).to(tl.uint32)
        up_word_byte1 = tl.load(up_payload_ptr + metadata_offset + 5).to(tl.uint32)
        up_word_byte2 = tl.load(up_payload_ptr + metadata_offset + 6).to(tl.uint32)
        up_word_byte3 = tl.load(up_payload_ptr + metadata_offset + 7).to(tl.uint32)
        up_sign_scale = (
            up_word_byte0
            | (up_word_byte1 << 8)
            | (up_word_byte2 << 16)
            | (up_word_byte3 << 24)
        )

        sign_shift = grid_byte_slot * 7
        gate_sign_index = (gate_sign_scale >> sign_shift) & 0x7F
        gate_packed_signs = tl.load(ksigns_ptr + gate_sign_index).to(tl.uint32)
        gate_sign_bits = (gate_packed_signs >> grid_lane) & 0x01
        gate_signs = 1.0 - gate_sign_bits.to(tl.float32) * 2.0

        up_sign_index = (up_sign_scale >> sign_shift) & 0x7F
        up_packed_signs = tl.load(ksigns_ptr + up_sign_index).to(tl.uint32)
        up_sign_bits = (up_packed_signs >> grid_lane) & 0x01
        up_signs = 1.0 - up_sign_bits.to(tl.float32) * 2.0

        gate_scale_code = (gate_sign_scale >> 28).to(tl.float32)
        up_scale_code = (up_sign_scale >> 28).to(tl.float32)
        half_base = (row * COLUMNS_BLOCKS + block_idx) * HALF_WORDS_PER_BLOCK
        gate_d = tl.load(gate_payload_half_ptr + half_base).to(tl.float32)
        up_d = tl.load(up_payload_half_ptr + half_base).to(tl.float32)
        gate_block_scales = gate_d * (0.5 + gate_scale_code) * 0.25
        up_block_scales = up_d * (0.5 + up_scale_code) * 0.25

        gate_grid = tl.load(grid_ptr + gate_grid_bytes * 8 + grid_lane).to(tl.float32)
        up_grid = tl.load(grid_ptr + up_grid_bytes * 8 + grid_lane).to(tl.float32)
        hidden = tl.load(hidden_ptr + block_idx * BLOCK_SIZE + offsets).to(tl.float32)
        gate_total += tl.sum(
            gate_block_scales * gate_grid * gate_signs * hidden,
            axis=0,
        )
        up_total += tl.sum(
            up_block_scales * up_grid * up_signs * hidden,
            axis=0,
        )

    gate_clamped = tl.minimum(gate_total, 10.0)
    up_clamped = tl.minimum(tl.maximum(up_total, -10.0), 10.0)
    activated = gate_clamped / (1.0 + tl.exp(-gate_clamped)) * up_clamped
    tl.store(output_ptr + row, activated)


def _iq2_xxs_lookup_tensors_cuda(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    device_key = str(device)
    cached = _IQ2_XXS_LOOKUP_CACHE.get(device_key)
    if cached is None:
        ksigns, grid = iq2_xxs_lookup_tensors()
        cached = (
            ksigns.to(device=device, non_blocking=True),
            grid.to(device=device, non_blocking=True),
        )
        _IQ2_XXS_LOOKUP_CACHE[device_key] = cached
    return cached


def _iq2_xxs_matvec_triton_cuda(
    payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
) -> torch.Tensor:
    output = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    payload_contiguous = payload.contiguous()
    hidden_contiguous = hidden.contiguous()
    ksigns, grid = _iq2_xxs_lookup_tensors_cuda(hidden.device)
    _iq2_xxs_matvec_kernel[(rows,)](
        payload_contiguous,
        payload_contiguous.view(torch.float16),
        hidden_contiguous,
        ksigns,
        grid,
        output,
        BLOCK_SIZE=_GGUF_BLOCK_COLUMNS,
        BLOCK_BYTES=_IQ2_XXS_BLOCK_BYTES,
        HALF_WORDS_PER_BLOCK=_IQ2_XXS_BLOCK_BYTES // 2,
        num_warps=8,
    )
    return output


def _iq2_xxs_gate_up_activation_triton_cuda(
    gate_payload: torch.Tensor,
    up_payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
) -> torch.Tensor:
    output = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    gate_payload_contiguous = gate_payload.contiguous()
    up_payload_contiguous = up_payload.contiguous()
    hidden_contiguous = hidden.contiguous()
    ksigns, grid = _iq2_xxs_lookup_tensors_cuda(hidden.device)
    _iq2_xxs_gate_up_activation_kernel[(rows,)](
        gate_payload_contiguous,
        gate_payload_contiguous.view(torch.float16),
        up_payload_contiguous,
        up_payload_contiguous.view(torch.float16),
        hidden_contiguous,
        ksigns,
        grid,
        output,
        BLOCK_SIZE=_GGUF_BLOCK_COLUMNS,
        BLOCK_BYTES=_IQ2_XXS_BLOCK_BYTES,
        HALF_WORDS_PER_BLOCK=_IQ2_XXS_BLOCK_BYTES // 2,
        num_warps=8,
    )
    return output


def _iq2_xxs_gate_up_activation_multiblock_triton_cuda(
    gate_payload: torch.Tensor,
    up_payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
    columns_blocks: int,
) -> torch.Tensor:
    output = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    gate_payload_contiguous = gate_payload.contiguous()
    up_payload_contiguous = up_payload.contiguous()
    hidden_contiguous = hidden.contiguous()
    ksigns, grid = _iq2_xxs_lookup_tensors_cuda(hidden.device)
    _iq2_xxs_gate_up_activation_multiblock_kernel[(rows,)](
        gate_payload_contiguous,
        gate_payload_contiguous.view(torch.float16),
        up_payload_contiguous,
        up_payload_contiguous.view(torch.float16),
        hidden_contiguous,
        ksigns,
        grid,
        output,
        BLOCK_SIZE=_GGUF_BLOCK_COLUMNS,
        BLOCK_BYTES=_IQ2_XXS_BLOCK_BYTES,
        HALF_WORDS_PER_BLOCK=_IQ2_XXS_BLOCK_BYTES // 2,
        COLUMNS_BLOCKS=columns_blocks,
        num_warps=8,
    )
    return output


def _q2_k_matvec_reference_cuda(
    payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    hidden_f32 = _validate_hidden(hidden, columns=columns)
    matrix = q2_k_matrix_from_gguf_payload(
        _payload_bytes(payload),
        rows=rows,
        columns=columns,
    ).to(device=hidden.device)
    return matrix.matmul(hidden_f32)


def _iq2_xxs_matvec_reference_cuda(
    payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    hidden_f32 = _validate_hidden(hidden, columns=columns)
    matrix = iq2_xxs_matrix_from_gguf_payload(
        _payload_bytes(payload),
        rows=rows,
        columns=columns,
    ).to(device=hidden.device)
    return matrix.matmul(hidden_f32)


def deepseek_v4_q2_k_matvec(
    payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
    columns: int,
    use_triton: bool = True,
) -> torch.Tensor:
    """Q2_K expert matvec.

    GGUF raw Q2_K block memory layout:
    - One block encodes 256 row-major matrix values in 84 bytes.
    - Bytes 0..15 hold 16 packed scale/min pairs, with low nibble as scale
      and high nibble as minimum scale.
    - Bytes 16..79 hold 64 bytes of 2-bit quantized values.
    - Bytes 80..81 hold the fp16 block scale d, and bytes 82..83 hold the
      fp16 minimum scale dmin.

    Intended Triton tiling:
    - Treat the matrix as [rows, columns] and split each row into 256-value
      GGUF blocks.
    - A fused kernel should tile output rows and K blocks, decode Q2_K values
      in registers, multiply by hidden[K], and reduce to one fp32 value per row.
    """
    _validate_payload(
        payload,
        rows=rows,
        columns=columns,
        block_bytes=_Q2_K_BLOCK_BYTES,
    )
    hidden_f32 = _validate_hidden(hidden, columns=columns)
    if not use_triton:
        return _q2_k_matvec_reference_cuda(
            payload,
            hidden,
            rows=rows,
            columns=columns,
        )
    columns_blocks = columns // _GGUF_BLOCK_COLUMNS
    if columns_blocks != 1:
        return _q2_k_matvec_multiblock_triton_cuda(
            payload,
            hidden_f32,
            rows=rows,
            columns_blocks=columns_blocks,
        )
    return _q2_k_matvec_triton_cuda(payload, hidden_f32, rows=rows)


def deepseek_v4_iq2_xxs_matvec(
    payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
    columns: int,
    use_triton: bool = True,
) -> torch.Tensor:
    """IQ2_XXS expert matvec.

    GGUF raw IQ2_XXS block memory layout:
    - One block encodes 256 row-major matrix values in 66 bytes.
    - Bytes 0..1 hold the fp16 block scale d.
    - Bytes 2..65 hold eight 8-byte groups. Each group carries packed grid
      indices and sign/scale metadata that the GGML IQ2_XXS reference expands
      into eight lanes of 2-bit grid values and signs.

    Intended Triton tiling:
    - Treat the matrix as [rows, columns] and split each row into 256-value
      GGUF blocks.
    - A fused kernel should tile output rows and K blocks, expand IQ2_XXS grid
      and sign metadata in registers, multiply by hidden[K], and reduce to one
      fp32 value per row.
    """
    _validate_payload(
        payload,
        rows=rows,
        columns=columns,
        block_bytes=_IQ2_XXS_BLOCK_BYTES,
    )
    hidden_f32 = _validate_hidden(hidden, columns=columns)
    if not use_triton:
        return _iq2_xxs_matvec_reference_cuda(
            payload,
            hidden,
            rows=rows,
            columns=columns,
        )
    if columns != _GGUF_BLOCK_COLUMNS:
        raise NotImplementedError(
            "DeepSeek V4 IQ2_XXS Triton matvec currently supports columns=256; "
            f"got {columns}. Pass use_triton=False for the reference fallback."
        )
    return _iq2_xxs_matvec_triton_cuda(payload, hidden_f32, rows=rows)


def deepseek_v4_iq2_xxs_gate_up(
    gate_payload: torch.Tensor,
    up_payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """IQ2_XXS gate/up expert matvec pair.

    This helper keeps the routed expert gate/up contract explicit while the
    current implementation uses two default IQ2_XXS matvec launches.
    """
    _validate_payload(
        gate_payload,
        rows=rows,
        columns=columns,
        block_bytes=_IQ2_XXS_BLOCK_BYTES,
    )
    _validate_payload(
        up_payload,
        rows=rows,
        columns=columns,
        block_bytes=_IQ2_XXS_BLOCK_BYTES,
    )
    hidden_f32 = _validate_hidden(hidden, columns=columns)
    if columns != _GGUF_BLOCK_COLUMNS:
        raise NotImplementedError(
            "DeepSeek V4 IQ2_XXS gate/up Triton helper currently supports "
            f"columns=256; got {columns}"
        )
    return (
        _iq2_xxs_matvec_triton_cuda(gate_payload, hidden_f32, rows=rows),
        _iq2_xxs_matvec_triton_cuda(up_payload, hidden_f32, rows=rows),
    )


def deepseek_v4_iq2_xxs_gate_up_activation(
    gate_payload: torch.Tensor,
    up_payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """IQ2_XXS fused gate/up activation.

    The fused helper decodes matching gate and up IQ2_XXS rows in one Triton
    program per output row, computes both dot products, and returns the
    activated intermediate consumed by the down projection:
    ``silu(gate @ hidden) * (up @ hidden)``.
    """
    _validate_payload(
        gate_payload,
        rows=rows,
        columns=columns,
        block_bytes=_IQ2_XXS_BLOCK_BYTES,
    )
    _validate_payload(
        up_payload,
        rows=rows,
        columns=columns,
        block_bytes=_IQ2_XXS_BLOCK_BYTES,
    )
    hidden_f32 = _validate_hidden(hidden, columns=columns)
    columns_blocks = columns // _GGUF_BLOCK_COLUMNS
    if columns_blocks != 1:
        return _iq2_xxs_gate_up_activation_multiblock_triton_cuda(
            gate_payload,
            up_payload,
            hidden_f32,
            rows=rows,
            columns_blocks=columns_blocks,
        )
    return _iq2_xxs_gate_up_activation_triton_cuda(
        gate_payload,
        up_payload,
        hidden_f32,
        rows=rows,
    )
