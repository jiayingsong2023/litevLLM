import struct

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.quant import (
    decode_iq2_xxs_synthetic,
    decode_q2_k_synthetic,
    decode_q8_0,
    decode_q8_0_gguf_blocks,
    q8_0_dequantize_reference,
    q8_0_dot,
    q8_0_linear_reference,
    q8_0_matrix_from_gguf_payload,
    q8_0_matvec,
)


def test_q8_0_decode_applies_one_scale_per_block() -> None:
    values = torch.tensor([1, -2, 3, -4, 5, 6, -7, 8], dtype=torch.int8)
    scales = torch.tensor([0.5, 0.25], dtype=torch.float32)

    decoded = decode_q8_0(values, scales, block_size=4)

    expected = torch.tensor(
        [0.5, -1.0, 1.5, -2.0, 1.25, 1.5, -1.75, 2.0],
        dtype=torch.float32,
    )
    torch.testing.assert_close(decoded, expected)


def test_q8_0_dot_matches_hand_computed_value() -> None:
    values = torch.tensor([2, -1, 4, 0], dtype=torch.int8)
    scales = torch.tensor([0.5, 0.25], dtype=torch.float32)
    vector = torch.tensor([1.0, 3.0, -2.0, 10.0], dtype=torch.float32)

    result = q8_0_dot(values, scales, vector, block_size=2)

    assert result.item() == pytest.approx(-2.5)


def test_q8_0_matvec_decodes_each_row_before_dot() -> None:
    values = torch.tensor(
        [
            [2, -1, 4, 0],
            [-3, 6, 1, -2],
        ],
        dtype=torch.int8,
    )
    scales = torch.tensor(
        [
            [0.5, 0.25],
            [0.25, 2.0],
        ],
        dtype=torch.float32,
    )
    vector = torch.tensor([1.0, 3.0, -2.0, 10.0], dtype=torch.float32)

    result = q8_0_matvec(values, scales, vector, block_size=2)

    torch.testing.assert_close(result, torch.tensor([-2.5, -40.25]))


def test_q8_0_linear_reference_matches_dequantized_matmul() -> None:
    values = torch.tensor([[1, -2, 3, -4], [4, 3, 2, 1]], dtype=torch.int8)
    scales = torch.tensor([[0.5, 0.25], [0.25, 0.5]], dtype=torch.float32)
    vector = torch.tensor([2.0, -1.0, 0.5, 4.0], dtype=torch.float32)

    weight = q8_0_dequantize_reference(values, scales, block_size=2)
    result = q8_0_linear_reference(vector, values, scales, block_size=2)

    torch.testing.assert_close(result, weight.matmul(vector))


def _pack_q8_0_block(scale: float, values: tuple[int, ...]) -> bytes:
    return struct.pack("<e", scale) + struct.pack(
        "<" + "b" * len(values),
        *values,
    )


def test_q8_0_gguf_payload_decoder_reads_scale_then_int8_values() -> None:
    payload = _pack_q8_0_block(0.5, (1, -2, 3, -4))

    values, scales = decode_q8_0_gguf_blocks(payload, block_size=4)

    torch.testing.assert_close(
        values,
        torch.tensor([1, -2, 3, -4], dtype=torch.int8),
    )
    torch.testing.assert_close(scales, torch.tensor([0.5], dtype=torch.float32))


def test_q8_0_matrix_from_gguf_payload_decodes_rows() -> None:
    payload = b"".join(
        (
            _pack_q8_0_block(0.5, (1, -2, 3, -4)),
            _pack_q8_0_block(0.25, (4, 3, 2, 1)),
        )
    )

    values, scales = q8_0_matrix_from_gguf_payload(
        payload,
        rows=2,
        columns=4,
        block_size=4,
    )

    torch.testing.assert_close(
        values,
        torch.tensor([[1, -2, 3, -4], [4, 3, 2, 1]], dtype=torch.int8),
    )
    torch.testing.assert_close(
        scales,
        torch.tensor([[0.5], [0.25]], dtype=torch.float32),
    )


def test_q8_0_gguf_payload_decoder_rejects_malformed_payload() -> None:
    with pytest.raises(ValueError, match="multiple of Q8_0 block bytes"):
        decode_q8_0_gguf_blocks(b"\x00\x00\x01", block_size=4)

    with pytest.raises(ValueError, match="columns"):
        q8_0_matrix_from_gguf_payload(
            _pack_q8_0_block(0.5, (1, -2, 3, -4)),
            rows=1,
            columns=3,
            block_size=4,
        )


def test_q8_0_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="int8"):
        decode_q8_0(torch.tensor([1, 2], dtype=torch.int16), torch.ones(1))

    with pytest.raises(ValueError, match="divisible"):
        decode_q8_0(torch.tensor([1, 2, 3], dtype=torch.int8), torch.ones(1))

    with pytest.raises(ValueError, match="scales"):
        decode_q8_0(
            torch.tensor([1, 2, 3, 4], dtype=torch.int8),
            torch.ones(1),
            block_size=2,
        )

    with pytest.raises(ValueError, match="vector"):
        q8_0_dot(
            torch.tensor([1, 2], dtype=torch.int8),
            torch.ones(1),
            torch.ones(3),
            block_size=2,
        )


def test_iq2_xxs_synthetic_decode_maps_two_bit_codes() -> None:
    codes = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.uint8)
    scales = torch.tensor([0.5, 2.0], dtype=torch.float32)

    decoded = decode_iq2_xxs_synthetic(codes, scales, block_size=4)

    expected = torch.tensor(
        [
            [-1.5, -0.5, 0.5, 1.5],
            [6.0, 2.0, -2.0, -6.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(decoded, expected)


def test_iq2_xxs_synthetic_decode_allows_empty_blocks() -> None:
    codes = torch.empty((0, 4), dtype=torch.uint8)
    scales = torch.empty((0,), dtype=torch.float32)

    decoded = decode_iq2_xxs_synthetic(codes, scales, block_size=4)

    assert decoded.shape == (0, 4)
    assert decoded.dtype == torch.float32


def test_iq2_xxs_synthetic_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="uint8"):
        decode_iq2_xxs_synthetic(torch.zeros((1, 4), dtype=torch.int8), torch.ones(1))

    with pytest.raises(ValueError, match="block_size"):
        decode_iq2_xxs_synthetic(torch.zeros((1, 3), dtype=torch.uint8), torch.ones(1))

    with pytest.raises(ValueError, match="two-bit"):
        decode_iq2_xxs_synthetic(
            torch.tensor([[4]], dtype=torch.uint8),
            torch.ones(1),
            block_size=1,
        )


def test_q2_k_synthetic_decode_applies_group_scale_and_min() -> None:
    codes = torch.tensor([[[0, 1, 2, 3], [3, 2, 1, 0]]], dtype=torch.uint8)
    scales = torch.tensor([[0.5, 2.0]], dtype=torch.float32)
    mins = torch.tensor([[-1.0, 10.0]], dtype=torch.float32)

    decoded = decode_q2_k_synthetic(codes, scales, mins, group_size=4)

    expected = torch.tensor(
        [[[-1.0, -0.5, 0.0, 0.5], [16.0, 14.0, 12.0, 10.0]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(decoded, expected)


def test_q2_k_synthetic_decode_allows_empty_superblocks() -> None:
    codes = torch.empty((0, 2, 4), dtype=torch.uint8)
    scales = torch.empty((0, 2), dtype=torch.float32)
    mins = torch.empty((0, 2), dtype=torch.float32)

    decoded = decode_q2_k_synthetic(codes, scales, mins, group_size=4)

    assert decoded.shape == (0, 2, 4)
    assert decoded.dtype == torch.float32


def test_q2_k_synthetic_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="3-D"):
        decode_q2_k_synthetic(
            torch.zeros((2, 4), dtype=torch.uint8),
            torch.ones((2,)),
            torch.ones((2,)),
        )

    with pytest.raises(ValueError, match="scales"):
        decode_q2_k_synthetic(
            torch.zeros((1, 2, 4), dtype=torch.uint8),
            torch.ones((1, 1)),
            torch.ones((1, 2)),
            group_size=4,
        )

    with pytest.raises(ValueError, match="two-bit"):
        decode_q2_k_synthetic(
            torch.tensor([[[0, 1, 2, 4]]], dtype=torch.uint8),
            torch.ones((1, 1)),
            torch.zeros((1, 1)),
            group_size=4,
        )
