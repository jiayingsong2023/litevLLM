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


def _iq2_xxs_nonuniform_block_payload() -> bytes:
    return bytes.fromhex(
        "003e052a4f748088681622476c9109cd8a273f6489ae9211ad385c81a6cb"
        "1b56cf49799ec3e8a49af15a96bbe0052ddf136cb3d8fd22b623367dd0"
        "f51a3f3f68588e"
    )


def test_iq2_xxs_reference_decoder_reads_valid_raw_block() -> None:
    decoded = decode_iq2_xxs_gguf_blocks_reference(_iq2_xxs_unit_block_payload())

    assert decoded.shape == (256,)
    assert decoded.dtype == torch.float32
    torch.testing.assert_close(decoded, torch.ones(256, dtype=torch.float32))


def test_iq2_xxs_reference_decoder_reads_nonuniform_raw_block() -> None:
    decoded = decode_iq2_xxs_gguf_blocks_reference(_iq2_xxs_nonuniform_block_payload())

    expected_prefix = torch.tensor(
        [
            14.0625,
            4.5,
            14.0625,
            4.5,
            4.5,
            4.5,
            4.5,
            4.5,
            -4.5,
            14.0625,
            4.5,
            24.1875,
            -4.5,
            14.0625,
            4.5,
            4.5,
            4.5,
            -4.5,
            14.0625,
            24.1875,
            4.5,
            -4.5,
            14.0625,
            4.5,
            -4.5,
            -24.1875,
            4.5,
            14.0625,
            -14.0625,
            -4.5,
            24.1875,
            4.5,
            -7.5,
            7.5,
            23.4375,
            -7.5,
            7.5,
            23.4375,
            7.5,
            7.5,
            7.5,
            -23.4375,
            23.4375,
            -7.5,
            -40.3125,
            40.3125,
            7.5,
            -7.5,
            -7.5,
            -7.5,
            7.5,
            -7.5,
            7.5,
            -7.5,
            40.3125,
            7.5,
            7.5,
            7.5,
            -7.5,
            -7.5,
            -23.4375,
            -7.5,
            7.5,
            23.4375,
        ],
        dtype=torch.float32,
    )
    expected_samples = {
        64: 32.8125,
        95: -32.8125,
        127: 42.1875,
        128: 16.5,
        159: -88.6875,
        191: 19.5,
        223: 22.5,
        255: 25.5,
    }
    torch.testing.assert_close(decoded[:64], expected_prefix)
    expected = torch.tensor(
        [expected_samples[idx] for idx in expected_samples],
        dtype=torch.float32,
    )
    actual = decoded[torch.tensor(tuple(expected_samples), dtype=torch.long)]
    torch.testing.assert_close(actual, expected)


def test_iq2_xxs_reference_decoder_rejects_invalid_payload_length() -> None:
    with pytest.raises(ValueError, match="IQ2_XXS payload length"):
        decode_iq2_xxs_gguf_blocks_reference(b"\x00" * 65)


def _q2_k_repeating_codes_payload() -> bytes:
    scales = b"\x11" * 16
    qs = b"\xe4" * 64
    return scales + qs + struct.pack("<e", 1.0) + struct.pack("<e", 0.5)


def _q2_k_nonuniform_block_payload() -> bytes:
    scales = bytes((idx * 7 + 0x21) & 0xFF for idx in range(16))
    qs = bytes((idx * 11 + 0x35) & 0xFF for idx in range(64))
    return scales + qs + struct.pack("<e", 1.25) + struct.pack("<e", -0.75)


def _q2_k_nonuniform_expected() -> torch.Tensor:
    payload = _q2_k_nonuniform_block_payload()
    scales = payload[:16]
    qs = payload[16:80]
    d = 1.25
    dmin = -0.75
    values: list[float] = []
    for group in range(16):
        scale = scales[group] & 0x0F
        minimum = scales[group] >> 4
        for pos in range(16):
            flat = group * 16 + pos
            q_half = flat // 128
            rem = flat % 128
            shift = (rem // 32) * 2
            byte_idx = q_half * 32 + rem % 32
            code = (qs[byte_idx] >> shift) & 0x03
            values.append(d * scale * code - dmin * minimum)
    return torch.tensor(values, dtype=torch.float32)


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


def test_q2_k_reference_decoder_reads_nonuniform_raw_block() -> None:
    decoded = decode_q2_k_gguf_blocks_reference(_q2_k_nonuniform_block_payload())

    torch.testing.assert_close(decoded, _q2_k_nonuniform_expected())


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
    with pytest.raises(TypeError):
        iq2_xxs_matrix_from_gguf_payload(payload, 2, 256)


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
    with pytest.raises(TypeError):
        q2_k_matrix_from_gguf_payload(payload, 2, 256)
