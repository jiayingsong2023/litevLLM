from __future__ import annotations

import pytest
import torch

from vllm.kernels.triton.deepseek_v4_flash.output import (
    DeepSeekV4OutputKernelInputs,
    deepseek_v4_output_argmax_with_value,
    deepseek_v4_output_projection,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
)
from vllm.model_executor.models.deepseek_v4_flash.quant import q8_0_matvec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_deepseek_gpu_output_projection_matches_q8_reference_on_device() -> None:
    device = torch.device("cuda")
    streams = torch.arange(128, dtype=torch.float32, device=device).reshape(4, 32)
    output_hc_weight = torch.zeros((4, 128), dtype=torch.float32, device=device)
    output_hc_scale = torch.ones(1, dtype=torch.float32, device=device)
    output_hc_base = torch.zeros(4, dtype=torch.float32, device=device)
    output_norm_weight = torch.ones(32, dtype=torch.float32, device=device)
    lm_head_values = torch.arange(96, dtype=torch.int8, device=device).reshape(3, 32)
    lm_head_scales = torch.full((3, 1), 0.25, dtype=torch.float32, device=device)

    inputs = DeepSeekV4OutputKernelInputs(
        streams=streams,
        lm_head_values=lm_head_values,
        lm_head_scales=lm_head_scales,
        output_hc_weight=output_hc_weight,
        output_hc_scale=output_hc_scale,
        output_hc_base=output_hc_base,
        output_norm_weight=output_norm_weight,
        block_size=32,
    )
    logits = deepseek_v4_output_projection(inputs)

    flat = streams.reshape(-1)
    flat = flat * torch.rsqrt(flat.pow(2).mean() + 1e-6)
    pre = output_hc_weight.matmul(flat)
    weights = torch.sigmoid(pre * output_hc_scale[0] + output_hc_base) + 1e-6
    hidden = (weights.reshape(4, 1) * streams).sum(dim=0)
    hidden = hidden * torch.rsqrt(hidden.pow(2).mean() + 1e-6)
    expected = q8_0_matvec(lm_head_values, lm_head_scales, hidden, block_size=32)

    assert logits.device.type == "cuda"
    assert logits.shape == (3,)
    torch.testing.assert_close(logits, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_deepseek_gpu_output_argmax_with_value_matches_projection() -> None:
    device = torch.device("cuda")
    streams = torch.arange(128, dtype=torch.float32, device=device).reshape(4, 32)
    output_hc_weight = torch.zeros((4, 128), dtype=torch.float32, device=device)
    output_hc_scale = torch.ones(1, dtype=torch.float32, device=device)
    output_hc_base = torch.zeros(4, dtype=torch.float32, device=device)
    output_norm_weight = torch.ones(32, dtype=torch.float32, device=device)
    lm_head_values = torch.arange(128, dtype=torch.int8, device=device).reshape(4, 32)
    lm_head_scales = torch.full((4, 1), 0.25, dtype=torch.float32, device=device)
    inputs = DeepSeekV4OutputKernelInputs(
        streams=streams,
        lm_head_values=lm_head_values,
        lm_head_scales=lm_head_scales,
        output_hc_weight=output_hc_weight,
        output_hc_scale=output_hc_scale,
        output_hc_base=output_hc_base,
        output_norm_weight=output_norm_weight,
        block_size=32,
    )

    token, value = deepseek_v4_output_argmax_with_value(inputs, row_offset=1024)
    logits = deepseek_v4_output_projection(inputs)
    expected_index = torch.argmax(logits)

    assert token.device.type == "cuda"
    assert value.device.type == "cuda"
    torch.testing.assert_close(token, expected_index.to(torch.long) + 1024)
    torch.testing.assert_close(value, logits[expected_index])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_backend_output_path_requires_gpu_tensors() -> None:
    backend = DeepSeekV4FlashGPUBackend()
    streams = torch.zeros((4, 32), dtype=torch.float32)

    with pytest.raises(ValueError, match="must be CUDA tensors"):
        backend.output_logits(
            streams=streams,
            lm_head_values=torch.zeros((1, 32), dtype=torch.int8),
            lm_head_scales=torch.ones((1, 1), dtype=torch.float32),
            output_hc_weight=torch.zeros((4, 128), dtype=torch.float32),
            output_hc_scale=torch.ones(1, dtype=torch.float32),
            output_hc_base=torch.zeros(4, dtype=torch.float32),
            output_norm_weight=torch.ones(32, dtype=torch.float32),
            block_size=32,
        )
