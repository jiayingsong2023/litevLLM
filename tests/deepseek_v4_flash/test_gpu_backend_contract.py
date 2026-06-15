from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_Q2_K,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
    DeepSeekV4FlashGPUCapabilities,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)


def _iq2_xxs_unit_block_payload() -> bytes:
    return struct.pack("<e", 1.0) + (b"\x00" * 64)


def _q2_k_repeating_codes_payload() -> bytes:
    scales = b"\x11" * 16
    qs = b"\xe4" * 64
    return scales + qs + struct.pack("<e", 1.0) + struct.pack("<e", 0.5)


def _cuda_payload(payload: bytes) -> torch.Tensor:
    return torch.tensor(tuple(payload), dtype=torch.uint8, device="cuda")


def _staged_payload(
    *,
    ggml_type: int,
    rows: int,
    columns: int,
    payload: bytes,
) -> SimpleNamespace:
    return SimpleNamespace(
        ggml_type=ggml_type,
        rows=rows,
        columns=columns,
        payload=_cuda_payload(payload),
    )


def test_gpu_backend_reports_missing_required_kernels() -> None:
    caps = DeepSeekV4FlashGPUCapabilities(
        q8_linear=True,
        attention=False,
        compressed_attention=False,
        cache_update=False,
        moe=False,
        output=False,
    )
    backend = DeepSeekV4FlashGPUBackend(capabilities=caps)

    assert backend.is_ready is False
    assert backend.missing_kernels == (
        "attention",
        "compressed_attention",
        "cache_update",
        "moe",
        "output",
    )
    with pytest.raises(RuntimeError, match="missing GPU kernels"):
        backend.require_ready()


def test_gpu_backend_can_report_ready_when_all_kernels_are_enabled() -> None:
    caps = DeepSeekV4FlashGPUCapabilities(
        q8_linear=True,
        attention=True,
        compressed_attention=True,
        cache_update=True,
        moe=True,
        output=True,
    )
    backend = DeepSeekV4FlashGPUBackend(capabilities=caps)

    assert backend.is_ready is True
    assert backend.missing_kernels == ()
    backend.require_ready()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_output_argmax_matches_output_logits() -> None:
    caps = DeepSeekV4FlashGPUCapabilities(
        q8_linear=True,
        attention=True,
        compressed_attention=True,
        cache_update=True,
        moe=True,
        output=True,
    )
    backend = DeepSeekV4FlashGPUBackend(capabilities=caps)
    streams = torch.ones((4, 4), dtype=torch.float32, device="cuda")
    values = torch.tensor(
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
        dtype=torch.int8,
        device="cuda",
    )
    scales = torch.ones((3, 1), dtype=torch.float32, device="cuda")
    hc_weight = torch.zeros((4, 16), dtype=torch.float32, device="cuda")
    hc_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    hc_base = torch.zeros((4,), dtype=torch.float32, device="cuda")
    norm = torch.ones((4,), dtype=torch.float32, device="cuda")

    row_offset = 7
    actual = backend.output_argmax(
        streams=streams,
        lm_head_values=values,
        lm_head_scales=scales,
        output_hc_weight=hc_weight,
        output_hc_scale=hc_scale,
        output_hc_base=hc_base,
        output_norm_weight=norm,
        block_size=4,
        row_offset=row_offset,
    )
    logits = backend.output_logits(
        streams=streams,
        lm_head_values=values,
        lm_head_scales=scales,
        output_hc_weight=hc_weight,
        output_hc_scale=hc_scale,
        output_hc_base=hc_base,
        output_norm_weight=norm,
        block_size=4,
    )

    assert actual.shape == ()
    torch.testing.assert_close(actual, torch.argmax(logits) + row_offset)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("ggml_type", "block_payload", "expected_stats"),
    (
        (
            GGML_TYPE_Q2_K,
            _q2_k_repeating_codes_payload(),
            {
                "quantized_expert_calls": 1,
                "q2_k_triton_calls": 3,
                "iq2_xxs_triton_calls": 0,
                "q2_iq2_reference_fallback_calls": 0,
            },
        ),
        (
            GGML_TYPE_IQ2_XXS,
            _iq2_xxs_unit_block_payload(),
            {
                "quantized_expert_calls": 1,
                "q2_k_triton_calls": 0,
                "iq2_xxs_triton_calls": 3,
                "q2_iq2_reference_fallback_calls": 0,
            },
        ),
    ),
)
def test_quantized_expert_gemm_uses_default_triton_dispatch(
    ggml_type: int,
    block_payload: bytes,
    expected_stats: dict[str, int],
) -> None:
    backend = DeepSeekV4FlashGPUBackend()
    hidden = torch.linspace(-0.5, 0.5, 256, dtype=torch.float32, device="cuda")
    gate_payload = block_payload * 256
    up_payload = block_payload * 256
    down_payload = block_payload * 256

    actual = backend.quantized_expert_gemm(
        hidden=hidden,
        gate_payload=_staged_payload(
            ggml_type=ggml_type,
            rows=256,
            columns=256,
            payload=gate_payload,
        ),
        up_payload=_staged_payload(
            ggml_type=ggml_type,
            rows=256,
            columns=256,
            payload=up_payload,
        ),
        down_payload=_staged_payload(
            ggml_type=ggml_type,
            rows=256,
            columns=256,
            payload=down_payload,
        ),
    )

    assert actual.shape == (256,)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.zeros_like(actual))
    assert backend.stats() == expected_stats


def test_model_keeps_kernel_execution_disabled_until_backend_ready() -> None:
    model = DeepSeekV4FlashForCausalLM(gpu_backend=DeepSeekV4FlashGPUBackend())

    assert model.kernel_execution_available is False
    with pytest.raises(NotImplementedError, match="kernel execution is not available"):
        model.forward_full(torch.tensor([1], dtype=torch.long), use_kernel=True)
