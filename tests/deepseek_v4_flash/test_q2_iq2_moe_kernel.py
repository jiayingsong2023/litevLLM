from __future__ import annotations

import struct
from collections.abc import Callable

import pytest
import torch

from vllm.kernels.triton.deepseek_v4_flash import q2_iq2_moe
from vllm.kernels.triton.deepseek_v4_flash.q2_iq2_moe import (
    deepseek_v4_iq2_xxs_matvec,
    deepseek_v4_q2_k_matvec,
)
from vllm.model_executor.models.deepseek_v4_flash.quant import (
    iq2_xxs_matrix_from_gguf_payload,
    q2_k_matrix_from_gguf_payload,
)


def _iq2_xxs_unit_block_payload() -> bytes:
    return struct.pack("<e", 1.0) + (b"\x00" * 64)


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
    expected = q2_k_matrix_from_gguf_payload(
        payload,
        rows=rows,
        columns=columns,
    ).to(device="cuda").matmul(hidden)

    assert actual.shape == (rows,)
    assert actual.device.type == "cuda"
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_q2_matvec_requires_cuda_payload() -> None:
    payload = torch.tensor(tuple(_q2_k_repeating_codes_payload()), dtype=torch.uint8)
    hidden = torch.ones((256,), dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="payload must be a CUDA tensor"):
        deepseek_v4_q2_k_matvec(payload, hidden, rows=1, columns=256)
