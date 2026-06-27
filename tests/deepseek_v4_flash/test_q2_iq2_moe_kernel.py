from __future__ import annotations

import struct
from collections.abc import Callable

import pytest
import torch

from vllm.kernels.triton.deepseek_v4_flash import q2_iq2_moe
from vllm.kernels.triton.deepseek_v4_flash.q2_iq2_moe import (
    deepseek_v4_iq2_xxs_gate_up,
    deepseek_v4_iq2_xxs_gate_up_activation,
    deepseek_v4_iq2_xxs_matvec,
    deepseek_v4_q2_k_matvec,
)
from vllm.model_executor.models.deepseek_v4_flash.quant import (
    iq2_xxs_matrix_from_gguf_payload,
    q2_k_matrix_from_gguf_payload,
)


def _iq2_xxs_unit_block_payload() -> bytes:
    return struct.pack("<e", 1.0) + (b"\x00" * 64)


def _iq2_xxs_deterministic_payload(rows: int) -> bytes:
    blocks: list[bytes] = []
    for row in range(rows):
        groups = bytearray()
        for group in range(8):
            grid_bytes = bytes(
                ((row * 19 + group * 29 + idx * 37 + 0x17) & 0xFF) for idx in range(4)
            )
            sign_indices = [
                (row * 11 + group * 7 + idx * 23 + 0x05) & 0x7F for idx in range(4)
            ]
            scale_code = (row + group * 3) & 0x0F
            q_sign_scale = (
                sign_indices[0]
                | (sign_indices[1] << 7)
                | (sign_indices[2] << 14)
                | (sign_indices[3] << 21)
                | (scale_code << 28)
            )
            groups.extend(grid_bytes)
            groups.extend(struct.pack("<I", q_sign_scale))
        d = 0.03125 * (row % 7 + 1)
        blocks.append(struct.pack("<e", d) + bytes(groups))
    return b"".join(blocks)


def _iq2_xxs_deterministic_payload_blocks(rows: int, blocks_per_row: int) -> bytes:
    if blocks_per_row <= 0:
        raise ValueError("blocks_per_row must be positive")
    blocks: list[bytes] = []
    for row in range(rows):
        for block_idx in range(blocks_per_row):
            groups = bytearray()
            synthetic_row = row * 17 + block_idx
            for group in range(8):
                grid_bytes = bytes(
                    (synthetic_row * 19 + group * 29 + idx * 37 + 0x17) & 0xFF
                    for idx in range(4)
                )
                sign_indices = [
                    (synthetic_row * 11 + group * 7 + idx * 23 + 0x05) & 0x7F
                    for idx in range(4)
                ]
                scale_code = (synthetic_row + group * 3) & 0x0F
                q_sign_scale = (
                    sign_indices[0]
                    | (sign_indices[1] << 7)
                    | (sign_indices[2] << 14)
                    | (sign_indices[3] << 21)
                    | (scale_code << 28)
                )
                groups.extend(grid_bytes)
                groups.extend(struct.pack("<I", q_sign_scale))
            d = 0.03125 * (synthetic_row % 7 + 1)
            blocks.append(struct.pack("<e", d) + bytes(groups))
    return b"".join(blocks)


def _q2_k_repeating_codes_payload() -> bytes:
    scales = b"\x11" * 16
    qs = b"\xe4" * 64
    return scales + qs + struct.pack("<e", 1.0) + struct.pack("<e", 0.5)


def _q2_k_deterministic_payload(rows: int) -> bytes:
    blocks: list[bytes] = []
    for row in range(rows):
        scales = bytes(((row * 5 + idx * 7 + 0x21) & 0xFF) for idx in range(16))
        qs = bytes(((row * 13 + idx * 11 + 0x35) & 0xFF) for idx in range(64))
        d = 0.125 * (row % 5 + 1)
        dmin = -0.0625 * (row % 7 + 1)
        blocks.append(scales + qs + struct.pack("<e", d) + struct.pack("<e", dmin))
    return b"".join(blocks)


def _q2_k_deterministic_payload_blocks(rows: int, blocks_per_row: int) -> bytes:
    if blocks_per_row <= 0:
        raise ValueError("blocks_per_row must be positive")
    blocks: list[bytes] = []
    for row in range(rows):
        for block_idx in range(blocks_per_row):
            synthetic_row = row * 17 + block_idx
            scales = bytes(
                ((synthetic_row * 5 + idx * 7 + 0x21) & 0xFF) for idx in range(16)
            )
            qs = bytes(
                ((synthetic_row * 13 + idx * 11 + 0x35) & 0xFF) for idx in range(64)
            )
            d = 0.125 * (synthetic_row % 5 + 1)
            dmin = -0.0625 * (synthetic_row % 7 + 1)
            blocks.append(scales + qs + struct.pack("<e", d) + struct.pack("<e", dmin))
    return b"".join(blocks)


def _cuda_payload(payload: bytes) -> torch.Tensor:
    return torch.tensor(tuple(payload), dtype=torch.uint8, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_q2_k_default_path_does_not_copy_payload_to_cpu(monkeypatch) -> None:
    def fail_payload_bytes(_payload: torch.Tensor) -> bytes:
        raise AssertionError("CPU fallback used")

    monkeypatch.setattr(q2_iq2_moe, "_payload_bytes", fail_payload_bytes)
    payload = torch.zeros(84, dtype=torch.uint8, device="cuda")
    hidden = torch.ones(256, dtype=torch.float32, device="cuda")

    out = q2_iq2_moe.deepseek_v4_q2_k_matvec(
        payload,
        hidden,
        rows=1,
        columns=256,
    )

    assert out.shape == (1,)
    assert out.is_cuda
    assert out.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_iq2_xxs_default_path_does_not_copy_payload_to_cpu(monkeypatch) -> None:
    def fail_payload_bytes(_payload: torch.Tensor) -> bytes:
        raise AssertionError("CPU fallback used")

    monkeypatch.setattr(q2_iq2_moe, "_payload_bytes", fail_payload_bytes)
    payload = torch.zeros(66, dtype=torch.uint8, device="cuda")
    hidden = torch.ones(256, dtype=torch.float32, device="cuda")

    out = q2_iq2_moe.deepseek_v4_iq2_xxs_matvec(
        payload,
        hidden,
        rows=1,
        columns=256,
    )

    assert out.shape == (1,)
    assert out.is_cuda
    assert out.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("matvec", "decoder", "block_payload"),
    (
        (
            deepseek_v4_q2_k_matvec,
            q2_k_matrix_from_gguf_payload,
            _q2_k_repeating_codes_payload(),
        ),
        (
            deepseek_v4_iq2_xxs_matvec,
            iq2_xxs_matrix_from_gguf_payload,
            _iq2_xxs_unit_block_payload(),
        ),
    ),
)
def test_q2_iq2_matvec_matches_reference_decoded_matrix(
    matvec: Callable[..., torch.Tensor],
    decoder: Callable[..., torch.Tensor],
    block_payload: bytes,
) -> None:
    rows = 2
    columns = 256
    payload = block_payload * rows
    hidden = torch.linspace(-0.5, 0.5, columns, dtype=torch.float32, device="cuda")

    actual = matvec(
        _cuda_payload(payload),
        hidden,
        rows=rows,
        columns=columns,
        use_triton=False,
    )
    matrix = decoder(payload, rows=rows, columns=columns).to(device="cuda")
    expected = matrix.matmul(hidden)

    assert actual.shape == (rows,)
    assert actual.device.type == "cuda"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 2, 17, 64])
@pytest.mark.parametrize("hidden_scale", [0.25, 1.0])
def test_q2_k_default_triton_matvec_matches_reference_decoded_matrix(
    rows: int,
    hidden_scale: float,
) -> None:
    columns = 256
    payload = _q2_k_deterministic_payload(rows)
    hidden = (
        torch.linspace(-1.0, 1.0, columns, dtype=torch.float32, device="cuda")
        * hidden_scale
    )

    actual = deepseek_v4_q2_k_matvec(
        _cuda_payload(payload),
        hidden,
        rows=rows,
        columns=columns,
    )
    expected = (
        q2_k_matrix_from_gguf_payload(
            payload,
            rows=rows,
            columns=columns,
        )
        .to(device="cuda")
        .matmul(hidden)
    )

    assert actual.shape == (rows,)
    assert actual.device.type == "cuda"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("columns", [512, 2048])
def test_q2_k_default_triton_matvec_matches_reference_multi_block(
    columns: int,
) -> None:
    rows = 4
    payload = _q2_k_deterministic_payload_blocks(rows, columns // 256)
    hidden = torch.linspace(-1.0, 1.0, columns, dtype=torch.float32, device="cuda")

    actual = deepseek_v4_q2_k_matvec(
        _cuda_payload(payload),
        hidden,
        rows=rows,
        columns=columns,
    )
    expected = (
        q2_k_matrix_from_gguf_payload(
            payload,
            rows=rows,
            columns=columns,
        )
        .to(device="cuda")
        .matmul(hidden)
    )

    assert actual.shape == (rows,)
    assert actual.device.type == "cuda"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 2, 17, 64])
def test_iq2_xxs_default_triton_matvec_matches_reference_decoded_matrix(
    rows: int,
) -> None:
    columns = 256
    payload = _iq2_xxs_deterministic_payload(rows)
    hidden = torch.linspace(-0.75, 0.9, columns, dtype=torch.float32, device="cuda")

    actual = deepseek_v4_iq2_xxs_matvec(
        _cuda_payload(payload),
        hidden,
        rows=rows,
        columns=columns,
    )
    expected = (
        iq2_xxs_matrix_from_gguf_payload(
            payload,
            rows=rows,
            columns=columns,
        )
        .to(device="cuda")
        .matmul(hidden)
    )

    assert actual.shape == (rows,)
    assert actual.device.type == "cuda"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_iq2_xxs_gate_up_default_triton_matches_reference_decoded_matrix() -> None:
    rows = 4
    columns = 256
    gate_payload = _iq2_xxs_deterministic_payload(rows)
    up_payload = _iq2_xxs_deterministic_payload(rows)
    hidden = torch.linspace(-0.75, 0.9, columns, dtype=torch.float32, device="cuda")

    gate, up = deepseek_v4_iq2_xxs_gate_up(
        _cuda_payload(gate_payload),
        _cuda_payload(up_payload),
        hidden,
        rows=rows,
        columns=columns,
    )
    expected_gate = (
        iq2_xxs_matrix_from_gguf_payload(
            gate_payload,
            rows=rows,
            columns=columns,
        )
        .to(device="cuda")
        .matmul(hidden)
    )
    expected_up = (
        iq2_xxs_matrix_from_gguf_payload(
            up_payload,
            rows=rows,
            columns=columns,
        )
        .to(device="cuda")
        .matmul(hidden)
    )

    assert gate.shape == (rows,)
    assert up.shape == (rows,)
    assert gate.device.type == "cuda"
    assert up.device.type == "cuda"
    assert gate.dtype == torch.float32
    assert up.dtype == torch.float32
    torch.testing.assert_close(gate, expected_gate, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(up, expected_up, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_iq2_xxs_gate_up_activation_matches_two_matvec_reference() -> None:
    rows = 4
    columns = 256
    gate_payload = _iq2_xxs_deterministic_payload(rows)
    up_payload = _iq2_xxs_deterministic_payload(rows)
    hidden = torch.linspace(-0.75, 0.9, columns, dtype=torch.float32, device="cuda")

    actual = deepseek_v4_iq2_xxs_gate_up_activation(
        _cuda_payload(gate_payload),
        _cuda_payload(up_payload),
        hidden,
        rows=rows,
        columns=columns,
    )
    gate, up = deepseek_v4_iq2_xxs_gate_up(
        _cuda_payload(gate_payload),
        _cuda_payload(up_payload),
        hidden,
        rows=rows,
        columns=columns,
    )
    expected = torch.nn.functional.silu(torch.clamp(gate, max=10.0)) * torch.clamp(
        up,
        min=-10.0,
        max=10.0,
    )

    assert actual.shape == (rows,)
    assert actual.device.type == "cuda"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_iq2_xxs_gate_up_activation_applies_deepseek_clamp() -> None:
    rows = 4
    columns = 256
    gate_payload = _iq2_xxs_deterministic_payload(rows)
    up_payload = _iq2_xxs_deterministic_payload(rows)
    hidden = torch.linspace(-75.0, 90.0, columns, dtype=torch.float32, device="cuda")

    actual = deepseek_v4_iq2_xxs_gate_up_activation(
        _cuda_payload(gate_payload),
        _cuda_payload(up_payload),
        hidden,
        rows=rows,
        columns=columns,
    )
    gate, up = deepseek_v4_iq2_xxs_gate_up(
        _cuda_payload(gate_payload),
        _cuda_payload(up_payload),
        hidden,
        rows=rows,
        columns=columns,
    )
    expected = torch.nn.functional.silu(torch.clamp(gate, max=10.0)) * torch.clamp(
        up,
        min=-10.0,
        max=10.0,
    )

    assert torch.any(torch.abs(up) > 10.0) or torch.any(gate > 10.0)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("columns", [512, 1024, 2048])
def test_iq2_xxs_gate_up_activation_matches_reference_multi_block(
    columns: int,
) -> None:
    rows = 4
    blocks_per_row = columns // 256
    gate_payload = _iq2_xxs_deterministic_payload_blocks(rows, blocks_per_row)
    up_payload = _iq2_xxs_deterministic_payload_blocks(rows, blocks_per_row)
    hidden = torch.linspace(-0.75, 0.9, columns, dtype=torch.float32, device="cuda")

    actual = deepseek_v4_iq2_xxs_gate_up_activation(
        _cuda_payload(gate_payload),
        _cuda_payload(up_payload),
        hidden,
        rows=rows,
        columns=columns,
    )
    gate = (
        iq2_xxs_matrix_from_gguf_payload(
            gate_payload,
            rows=rows,
            columns=columns,
        )
        .to(device="cuda")
        .matmul(hidden)
    )
    up = (
        iq2_xxs_matrix_from_gguf_payload(
            up_payload,
            rows=rows,
            columns=columns,
        )
        .to(device="cuda")
        .matmul(hidden)
    )
    expected = torch.nn.functional.silu(torch.clamp(gate, max=10.0)) * torch.clamp(
        up,
        min=-10.0,
        max=10.0,
    )

    assert actual.shape == (rows,)
    assert actual.device.type == "cuda"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_q2_matvec_requires_cuda_payload() -> None:
    payload = torch.tensor(tuple(_q2_k_repeating_codes_payload()), dtype=torch.uint8)
    hidden = torch.ones((256,), dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="payload must be a CUDA tensor"):
        deepseek_v4_q2_k_matvec(payload, hidden, rows=1, columns=256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_q2_iq2_default_triton_path_rejects_multi_block_columns() -> None:
    q2_payload = _cuda_payload(_q2_k_repeating_codes_payload() * 2)
    iq2_payload = _cuda_payload(_iq2_xxs_unit_block_payload() * 2)
    hidden = torch.ones((512,), dtype=torch.float32, device="cuda")

    q2_out = deepseek_v4_q2_k_matvec(q2_payload, hidden, rows=1, columns=512)
    assert q2_out.shape == (1,)
    with pytest.raises(NotImplementedError, match="IQ2_XXS Triton matvec"):
        deepseek_v4_iq2_xxs_matvec(iq2_payload, hidden, rows=1, columns=512)
