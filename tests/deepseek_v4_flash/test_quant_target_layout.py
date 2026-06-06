import struct

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.quant import (
    decode_iq2_xxs_gguf_blocks_reference,
    decode_q2_k_gguf_blocks_reference,
    iq2_xxs_matrix_from_gguf_payload,
    q2_k_matrix_from_gguf_payload,
    q8_0_matrix_from_gguf_payload,
)


def test_q8_0_rejects_partial_raw_block_payloads() -> None:
    with pytest.raises(ValueError, match="multiple of Q8_0 block bytes"):
        q8_0_matrix_from_gguf_payload(b"\x00" * 33, rows=1, columns=32)


def _iq2_xxs_unit_block_payload() -> bytes:
    return struct.pack("<e", 1.0) + (b"\x00" * 64)


def test_iq2_xxs_reference_decoder_reads_valid_raw_block() -> None:
    decoded = decode_iq2_xxs_gguf_blocks_reference(_iq2_xxs_unit_block_payload())

    assert decoded.shape == (256,)
    assert decoded.dtype == torch.float32
    torch.testing.assert_close(decoded, torch.ones(256, dtype=torch.float32))


def test_iq2_xxs_reference_decoder_rejects_invalid_payload_length() -> None:
    with pytest.raises(ValueError, match="IQ2_XXS payload length"):
        decode_iq2_xxs_gguf_blocks_reference(b"\x00" * 65)


def _q2_k_repeating_codes_payload() -> bytes:
    scales = b"\x11" * 16
    qs = b"\xe4" * 64
    return scales + qs + struct.pack("<e", 1.0) + struct.pack("<e", 0.5)


def test_q2_k_reference_decoder_reads_valid_raw_block() -> None:
    decoded = decode_q2_k_gguf_blocks_reference(_q2_k_repeating_codes_payload())

    assert decoded.shape == (256,)
    assert decoded.dtype == torch.float32
    expected = torch.tensor(
        (
            [-0.5] * 32
            + [0.5] * 32
            + [1.5] * 32
            + [2.5] * 32
            + [-0.5] * 32
            + [0.5] * 32
            + [1.5] * 32
            + [2.5] * 32
        ),
        dtype=torch.float32,
    )
    torch.testing.assert_close(decoded, expected)


def test_q2_k_reference_decoder_rejects_invalid_payload_length() -> None:
    with pytest.raises(ValueError, match="Q2_K payload length"):
        decode_q2_k_gguf_blocks_reference(b"\x00" * 83)


def test_iq2_xxs_matrix_wrapper_validates_count_and_reshapes() -> None:
    payload = _iq2_xxs_unit_block_payload() * 2

    matrix = iq2_xxs_matrix_from_gguf_payload(payload, rows=2, columns=256)

    assert matrix.shape == (2, 256)
    torch.testing.assert_close(matrix, torch.ones((2, 256), dtype=torch.float32))
    with pytest.raises(ValueError, match=r"rows \* columns"):
        iq2_xxs_matrix_from_gguf_payload(payload, rows=1, columns=256)


def test_q2_k_matrix_wrapper_validates_count_and_reshapes() -> None:
    payload = _q2_k_repeating_codes_payload() * 2

    matrix = q2_k_matrix_from_gguf_payload(payload, rows=2, columns=256)

    assert matrix.shape == (2, 256)
    expected = torch.tensor(
        (
            [-0.5] * 32
            + [0.5] * 32
            + [1.5] * 32
            + [2.5] * 32
            + [-0.5] * 32
            + [0.5] * 32
            + [1.5] * 32
            + [2.5] * 32
        ),
        dtype=torch.float32,
    )
    torch.testing.assert_close(matrix[0], expected)
    torch.testing.assert_close(matrix[1], expected)
    with pytest.raises(ValueError, match=r"rows \* columns"):
        q2_k_matrix_from_gguf_payload(payload, rows=1, columns=256)
