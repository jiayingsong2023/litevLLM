import pytest
import torch

from vllm.kernels.triton.deepseek_v4_flash.q8_linear import q8_0_linear
from vllm.model_executor.models.deepseek_v4_flash.quant import q8_0_linear_reference


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
