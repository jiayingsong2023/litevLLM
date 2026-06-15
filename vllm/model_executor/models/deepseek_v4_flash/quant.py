# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import struct

import torch

_GGUF_K_BLOCK_VALUES = 256
_IQ2_XXS_BLOCK_BYTES = 66
_Q2_K_BLOCK_BYTES = 84

# GGML IQ2_XXS dequantization uses the ksigns_iq2xs table to expand 7-bit
# sign indices into eight sign bits for each 8-value grid row.
_IQ2_XXS_KSIGNS = (
    b"\x00\x81\x82\x03\x84\x05\x06\x87\x88\x09\x0a\x8b\x0c\x8d\x8e\x0f"
    b"\x90\x11\x12\x93\x14\x95\x96\x17\x18\x99\x9a\x1b\x9c\x1d\x1e\x9f"
    b"\xa0\x21\x22\xa3\x24\xa5\xa6\x27\x28\xa9\xaa\x2b\xac\x2d\x2e\xaf"
    b"\x30\xb1\xb2\x33\xb4\x35\x36\xb7\xb8\x39\x3a\xbb\x3c\xbd\xbe\x3f"
    b"\xc0\x41\x42\xc3\x44\xc5\xc6\x47\x48\xc9\xca\x4b\xcc\x4d\x4e\xcf"
    b"\x50\xd1\xd2\x53\xd4\x55\x56\xd7\xd8\x59\x5a\xdb\x5c\xdd\xde\x5f"
    b"\x60\xe1\xe2\x63\xe4\x65\x66\xe7\xe8\x69\x6a\xeb\x6c\xed\xee\x6f"
    b"\xf0\x71\x72\xf3\x74\xf5\xf6\x77\x78\xf9\xfa\x7b\xfc\x7d\x7e\xff"
)

# GGML stores iq2xxs_grid as 256 rows of 8 values. The upstream Python gguf
# reference keeps the grid compact by packing each row entry into 2 bits and
# mapping packed codes 0/1/2 back to the GGML levels 0x08/0x19/0x2B.
_IQ2_XXS_GRID_HEX = (
    "00000200050008000a00110014002000220028002a0041004400500058006100"
    "6400800082008a00a20001010401100115014001840198010002020222028202"
    "010404041004210424044004420448046004810484049004a404000502050805"
    "200546056905800591050906100640068406a406000805080808140828084108"
    "440850085208880804094009020a140a01100410101021104010601084109010"
    "951000110811201150115a118011241245120014081420142514491480141815"
    "6215001616160118041810184018811800190519a019511a002002200a204420"
    "6120802082202921482100220222012404241024402456240025412564259026"
    "082820289428442a014004401040184021402440404048405640604081408440"
    "9040004120416141804185410142104248425642684200440844204480449944"
    "124524450046014804481048404845480049584961498249454a904a00500850"
    "1150195020508050885004514251a4519152905492540a550156545600581158"
    "195864584059085a046010604060686000615561186260620064056410651265"
    "84654268008002800a8041808280048118814081118201840484108415844084"
    "608400854685948509864086608602880489118a0490109024904090a1901691"
    "8091459200942294449451958198209902a050a085a009a100a218a450a804a9"
)

_iq2_xxs_grid_cache: torch.Tensor | None = None
_iq2_xxs_ksigns_cache: torch.Tensor | None = None


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


def decode_q8_0_gguf_blocks(
    payload: bytes | bytearray | memoryview,
    *,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode raw GGUF Q8_0 blocks into separated int8 values and fp32 scales.

    GGUF/GGML Q8_0 blocks are laid out as one little-endian fp16 scale followed
    by ``block_size`` signed int8 quantized values. The default block size is 32.
    """
    _validate_positive_block_size(block_size)
    block_bytes = 2 + block_size
    if len(payload) % block_bytes != 0:
        raise ValueError(
            "Q8_0 payload length must be a multiple of Q8_0 block bytes; "
            f"got {len(payload)} bytes and block_bytes={block_bytes}"
        )

    block_count = len(payload) // block_bytes
    values: list[int] = []
    scales: list[float] = []
    view = memoryview(payload)
    values_format = "<" + "b" * block_size
    for block_idx in range(block_count):
        offset = block_idx * block_bytes
        scales.append(float(struct.unpack_from("<e", view, offset)[0]))
        values.extend(struct.unpack_from(values_format, view, offset + 2))
    return (
        torch.tensor(values, dtype=torch.int8),
        torch.tensor(scales, dtype=torch.float32),
    )


def q8_0_matrix_from_gguf_payload(
    payload: bytes | bytearray | memoryview,
    *,
    rows: int,
    columns: int,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode raw GGUF Q8_0 matrix payload into row-major values and scales."""
    _validate_positive_block_size(block_size)
    if rows < 0:
        raise ValueError(f"rows must be non-negative; got {rows}")
    if columns <= 0:
        raise ValueError(f"columns must be positive; got {columns}")
    if columns % block_size != 0:
        raise ValueError(
            "Q8_0 matrix columns must be divisible by block_size; "
            f"got columns={columns} and block_size={block_size}"
        )
    values, scales = decode_q8_0_gguf_blocks(payload, block_size=block_size)
    expected_values = rows * columns
    if values.numel() != expected_values:
        raise ValueError(
            "Q8_0 payload value count must match rows * columns; "
            f"got {values.numel()} values for rows={rows}, columns={columns}"
        )
    blocks_per_row = columns // block_size
    return (
        values.reshape(rows, columns),
        scales.reshape(rows, blocks_per_row),
    )


def _uint8_tensor_from_payload(payload: bytes | bytearray | memoryview) -> torch.Tensor:
    return torch.tensor(memoryview(payload).tolist(), dtype=torch.uint8)


def _validate_k_block_values(name: str, values_per_block: int) -> None:
    if values_per_block != _GGUF_K_BLOCK_VALUES:
        raise ValueError(
            f"{name} GGUF reference decoder only supports "
            f"values_per_block={_GGUF_K_BLOCK_VALUES}; got {values_per_block}"
        )


def _validate_matrix_shape(name: str, rows: int, columns: int) -> None:
    if rows < 0:
        raise ValueError(f"{name} rows must be non-negative; got {rows}")
    if columns <= 0:
        raise ValueError(f"{name} columns must be positive; got {columns}")
    if columns % _GGUF_K_BLOCK_VALUES != 0:
        raise ValueError(
            f"{name} columns must be divisible by {_GGUF_K_BLOCK_VALUES}; "
            f"got columns={columns}"
        )


def _half_scales_from_payload(
    payload: bytes | bytearray | memoryview,
    block_bytes: int,
    scale_offset: int,
) -> torch.Tensor:
    view = memoryview(payload)
    block_count = len(view) // block_bytes
    scales = [
        float(struct.unpack_from("<e", view, block_idx * block_bytes + scale_offset)[0])
        for block_idx in range(block_count)
    ]
    return torch.tensor(scales, dtype=torch.float32)


def _little_endian_u32_words(blocks: torch.Tensor) -> torch.Tensor:
    words = blocks.to(torch.int64).reshape(blocks.shape[0], -1, 4)
    return (
        words[..., 0]
        | (words[..., 1] << 8)
        | (words[..., 2] << 16)
        | (words[..., 3] << 24)
    )


def _iq2_xxs_grid() -> torch.Tensor:
    global _iq2_xxs_grid_cache
    if _iq2_xxs_grid_cache is None:
        packed = torch.tensor(
            list(bytes.fromhex(_IQ2_XXS_GRID_HEX)),
            dtype=torch.uint8,
        )
        shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)
        codes = ((packed.reshape(-1, 1) >> shifts) & 0x03).reshape(-1)
        grid_map = torch.tensor([0x08, 0x19, 0x2B], dtype=torch.float32)
        _iq2_xxs_grid_cache = grid_map[codes.to(torch.long)].reshape(256, 8)
    return _iq2_xxs_grid_cache


def iq2_xxs_lookup_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU lookup tensors used by the GGUF IQ2_XXS decoder."""
    global _iq2_xxs_ksigns_cache
    if _iq2_xxs_ksigns_cache is None:
        _iq2_xxs_ksigns_cache = torch.tensor(
            list(_IQ2_XXS_KSIGNS),
            dtype=torch.uint8,
        )
    return _iq2_xxs_ksigns_cache, _iq2_xxs_grid()


def decode_iq2_xxs_gguf_blocks_reference(
    payload: bytes | bytearray | memoryview,
    values_per_block: int = _GGUF_K_BLOCK_VALUES,
) -> torch.Tensor:
    """Decode GGUF IQ2_XXS blocks with the GGML reference layout.

    This is a CPU bring-up/reference decoder for raw GGUF payloads. It is not
    wired into DeepSeek V4 Flash forward execution.
    """
    _validate_k_block_values("IQ2_XXS", values_per_block)
    if len(payload) % _IQ2_XXS_BLOCK_BYTES != 0:
        raise ValueError(
            "IQ2_XXS payload length must be a multiple of IQ2_XXS block bytes; "
            f"got {len(payload)} bytes and block_bytes={_IQ2_XXS_BLOCK_BYTES}"
        )
    if len(payload) == 0:
        return torch.empty((0,), dtype=torch.float32)

    block_count = len(payload) // _IQ2_XXS_BLOCK_BYTES
    raw = _uint8_tensor_from_payload(payload).reshape(block_count, _IQ2_XXS_BLOCK_BYTES)
    d = _half_scales_from_payload(payload, _IQ2_XXS_BLOCK_BYTES, 0)
    qs = raw[:, 2:]
    words = _little_endian_u32_words(qs).reshape(block_count, 8, 2)
    q_sign_scale = words[..., 1]

    block_scales = (
        d.reshape(block_count, 1, 1, 1)
        * (0.5 + (q_sign_scale >> 28).to(torch.float32)).reshape(block_count, 8, 1, 1)
        * 0.25
    )

    sign_shifts = torch.tensor([0, 7, 14, 21], dtype=torch.int64)
    sign_indices = ((q_sign_scale.unsqueeze(-1) >> sign_shifts) & 0x7F).to(torch.long)
    ksigns, iq2_grid = iq2_xxs_lookup_tensors()
    packed_signs = ksigns[sign_indices]
    bit_shifts = torch.arange(8, dtype=torch.uint8)
    sign_bits = (packed_signs.unsqueeze(-1) >> bit_shifts) & 0x01
    signs = torch.where(sign_bits == 0, 1.0, -1.0).to(torch.float32)

    q_grid_bytes = qs.reshape(block_count, 8, 2, 4)[:, :, 0, :]
    grid = iq2_grid[q_grid_bytes.to(torch.long)]

    return (block_scales * grid * signs).reshape(-1)


def decode_q2_k_gguf_blocks_reference(
    payload: bytes | bytearray | memoryview,
    values_per_block: int = _GGUF_K_BLOCK_VALUES,
) -> torch.Tensor:
    """Decode GGUF Q2_K blocks with the GGML reference layout.

    This is a CPU bring-up/reference decoder for raw GGUF payloads. It is not
    wired into DeepSeek V4 Flash forward execution.
    """
    _validate_k_block_values("Q2_K", values_per_block)
    if len(payload) % _Q2_K_BLOCK_BYTES != 0:
        raise ValueError(
            "Q2_K payload length must be a multiple of Q2_K block bytes; "
            f"got {len(payload)} bytes and block_bytes={_Q2_K_BLOCK_BYTES}"
        )
    if len(payload) == 0:
        return torch.empty((0,), dtype=torch.float32)

    block_count = len(payload) // _Q2_K_BLOCK_BYTES
    raw = _uint8_tensor_from_payload(payload).reshape(block_count, _Q2_K_BLOCK_BYTES)
    scales = raw[:, :16]
    qs = raw[:, 16:80]
    d = _half_scales_from_payload(payload, _Q2_K_BLOCK_BYTES, 80)
    dmin = _half_scales_from_payload(payload, _Q2_K_BLOCK_BYTES, 82)

    scale_lows = (scales & 0x0F).to(torch.float32).reshape(block_count, 16, 1)
    scale_highs = (scales >> 4).to(torch.float32).reshape(block_count, 16, 1)
    dl = d.reshape(block_count, 1, 1) * scale_lows
    ml = dmin.reshape(block_count, 1, 1) * scale_highs

    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8).reshape(1, 1, 4, 1)
    codes = ((qs.reshape(block_count, -1, 1, 32) >> shifts) & 0x03).reshape(
        block_count, 16, 16
    )
    decoded = dl * codes.to(torch.float32) - ml
    return decoded.reshape(-1)


def iq2_xxs_matrix_from_gguf_payload(
    payload: bytes | bytearray | memoryview,
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Decode raw GGUF IQ2_XXS payload into a row-major fp32 matrix."""
    _validate_matrix_shape("IQ2_XXS matrix", rows, columns)
    decoded = decode_iq2_xxs_gguf_blocks_reference(payload)
    expected_values = rows * columns
    if decoded.numel() != expected_values:
        raise ValueError(
            "IQ2_XXS payload decoded value count must match rows * columns; "
            f"got {decoded.numel()} values for rows={rows}, columns={columns}"
        )
    return decoded.reshape(rows, columns)


def q2_k_matrix_from_gguf_payload(
    payload: bytes | bytearray | memoryview,
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    """Decode raw GGUF Q2_K payload into a row-major fp32 matrix."""
    _validate_matrix_shape("Q2_K matrix", rows, columns)
    decoded = decode_q2_k_gguf_blocks_reference(payload)
    expected_values = rows * columns
    if decoded.numel() != expected_values:
        raise ValueError(
            "Q2_K payload decoded value count must match rows * columns; "
            f"got {decoded.numel()} values for rows={rows}, columns={columns}"
        )
    return decoded.reshape(rows, columns)


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

    This helper keeps the older unpacked test layout separate from the raw GGUF
    IQ2_XXS reference decoder above. The synthetic layout is one uint8 two-bit
    code per element with shape ``(blocks, block_size)``. Codes map to signed
    levels ``[-3, -1, 1, 3]`` and each block has one floating-point scale.
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

    This helper keeps the older unpacked test layout separate from the raw GGUF
    Q2_K reference decoder above. Codes are uint8 two-bit values with shape
    ``(super_blocks, groups, group_size)``. Every group has one scale and one
    additive minimum with shape ``(super_blocks, groups)``.
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

    return codes.to(torch.float32) * scales.to(torch.float32).unsqueeze(-1) + mins.to(
        torch.float32
    ).unsqueeze(-1)
