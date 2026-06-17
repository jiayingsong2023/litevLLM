import struct

import pytest
import torch

from vllm.kernels.triton.deepseek_v4_flash import q8_linear
from vllm.kernels.triton.deepseek_v4_flash.q8_linear import (
    q8_0_linear,
    q8_0_raw_linear,
)
from vllm.model_executor.models.deepseek_v4_flash.quant import (
    q8_0_linear_reference,
    q8_0_matrix_from_gguf_payload,
)


def test_q8_0_linear_rejects_malformed_cpu_inputs() -> None:
    vector = torch.ones((1, 4), dtype=torch.float32)
    values = torch.ones((2, 4), dtype=torch.int8)
    scales = torch.ones((2, 1), dtype=torch.float32)

    with pytest.raises(ValueError, match="vector"):
        q8_0_linear(vector, values, scales, block_size=4)

    with pytest.raises(ValueError, match="values"):
        q8_0_linear(torch.ones(4), values.reshape(1, 2, 4), scales, block_size=4)

    with pytest.raises(ValueError, match="scales"):
        q8_0_linear(torch.ones(4), values, scales.reshape(2), block_size=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_q8_0_linear_matches_reference_on_gpu() -> None:
    vector = torch.randn(64, device="cuda", dtype=torch.float32)
    values = torch.randint(-8, 8, (32, 64), device="cuda", dtype=torch.int8)
    scales = torch.rand(32, 2, device="cuda", dtype=torch.float32) * 0.1

    got = q8_0_linear(vector, values, scales)
    expected = q8_0_linear_reference(
        vector.cpu(),
        values.cpu(),
        scales.cpu(),
    ).to("cuda")

    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


def _pack_q8_0_block(scale: float, values: tuple[int, ...]) -> bytes:
    if len(values) != 32:
        raise ValueError("Q8_0 test block must contain 32 values")
    return struct.pack("<e", scale) + struct.pack("<" + "b" * 32, *values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_q8_0_raw_linear_matches_reference_without_split_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_split_linear(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("raw Q8_0 path must not use split values/scales")

    monkeypatch.setattr(q8_linear, "q8_0_linear", fail_split_linear)
    rows = 3
    columns = 64
    payload = b"".join(
        _pack_q8_0_block(
            0.125 * (row + block_idx + 1),
            tuple(((row * 13 + block_idx * 7 + col) % 17) - 8 for col in range(32)),
        )
        for row in range(rows)
        for block_idx in range(columns // 32)
    )
    raw = torch.tensor(tuple(payload), dtype=torch.uint8, device="cuda")
    hidden = torch.linspace(-0.5, 0.5, columns, dtype=torch.float32, device="cuda")

    got = q8_0_raw_linear(raw, hidden, rows=rows, columns=columns)
    values, scales = q8_0_matrix_from_gguf_payload(
        payload,
        rows=rows,
        columns=columns,
    )
    expected = q8_0_linear_reference(hidden.cpu(), values, scales).to("cuda")

    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)
