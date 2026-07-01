import struct

import pytest
import torch

from vllm.kernels.triton.deepseek_v4_flash.indexer_qat import (
    deepseek_v4_indexer_qat,
)
from vllm.model_executor.models.deepseek_v4_flash.attention import (
    apply_deepseek_layer_rope_to_tail_reference,
    apply_rope_to_tail_reference,
)
from vllm.model_executor.models.deepseek_v4_flash.ops import (
    deepseek_fp8_kv_qat_reference,
    deepseek_indexer_qat_reference,
    deepseek_q8_k_roundtrip_reference,
    e4m3fn_dequant_reference,
    silu_gate_reference,
)
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


def test_silu_gate_reference_applies_deepseek_clamp() -> None:
    gate = torch.tensor([-20.0, -2.0, 2.0, 20.0])
    up = torch.tensor([-20.0, -2.0, 2.0, 20.0])

    actual = silu_gate_reference(gate, up, clamp=10.0)
    expected = torch.nn.functional.silu(torch.clamp(gate, max=10.0)) * torch.clamp(
        up,
        min=-10.0,
        max=10.0,
    )

    torch.testing.assert_close(actual, expected)
    assert actual[-1] < torch.nn.functional.silu(gate[-1]) * up[-1]


def test_e4m3fn_dequant_reference_matches_ds4_saturation() -> None:
    values = torch.tensor([0.0, 0.002, 1.1, 500.0, -500.0])

    actual = e4m3fn_dequant_reference(values)

    torch.testing.assert_close(
        actual,
        torch.tensor([0.0, 0.001953125, 1.125, 448.0, -448.0]),
    )


def test_deepseek_fp8_kv_qat_quantizes_only_non_rope_blocks() -> None:
    row = torch.cat(
        [
            torch.linspace(-2.0, 2.0, 64),
            torch.linspace(100.0, 200.0, 64),
        ]
    )

    actual = deepseek_fp8_kv_qat_reference(row, head_dim=128, rotary_dim=64)

    assert not torch.allclose(actual[:64], row[:64])
    torch.testing.assert_close(actual[64:], row[64:])


def test_deepseek_indexer_qat_reference_runs_hadamard_fp4_roundtrip() -> None:
    row = torch.zeros(128)
    row[0] = 1.0

    actual = deepseek_indexer_qat_reference(row)

    assert actual.shape == (128,)
    assert torch.count_nonzero(actual).item() == 128
    assert torch.all(torch.abs(actual) <= 0.125)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_deepseek_indexer_qat_triton_matches_reference() -> None:
    row = torch.linspace(-3.0, 3.0, 128, dtype=torch.float32, device="cuda")

    actual = deepseek_v4_indexer_qat(row)
    expected = deepseek_indexer_qat_reference(row).to("cuda")

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_deepseek_layer_rope_keeps_dense_layers_on_default_base() -> None:
    vector = torch.arange(512, dtype=torch.float32)

    actual = apply_deepseek_layer_rope_to_tail_reference(
        vector,
        token_idx=7,
        layer_idx=1,
    )
    expected = apply_rope_to_tail_reference(vector, token_idx=7)

    torch.testing.assert_close(actual, expected)


def test_deepseek_layer_rope_uses_compressed_layer_scaling() -> None:
    vector = torch.arange(512, dtype=torch.float32)

    dense = apply_deepseek_layer_rope_to_tail_reference(
        vector,
        token_idx=7,
        layer_idx=1,
    )
    compressed = apply_deepseek_layer_rope_to_tail_reference(
        vector,
        token_idx=7,
        layer_idx=2,
    )

    assert not torch.allclose(compressed[-64:], dense[-64:])


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


def test_deepseek_q8_k_roundtrip_quantizes_256_value_blocks() -> None:
    row = torch.linspace(-1.0, 2.0, 256, dtype=torch.float32)

    actual = deepseek_q8_k_roundtrip_reference(row)

    signed_max = row[row.abs().argmax()]
    iscale = -127.0 / signed_max
    expected = torch.round(row * iscale).clamp(min=-128, max=127) / iscale
    torch.testing.assert_close(actual, expected)


def test_deepseek_q8_k_roundtrip_preserves_zero_blocks() -> None:
    row = torch.zeros(512, dtype=torch.float32)

    actual = deepseek_q8_k_roundtrip_reference(row)

    torch.testing.assert_close(actual, row)
