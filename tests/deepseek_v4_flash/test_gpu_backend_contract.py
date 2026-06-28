from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.models.deepseek_v4_flash import gpu_backend as backend_module
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
from vllm.model_executor.models.deepseek_v4_flash.ops import (
    deepseek_q8_k_roundtrip_reference,
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
    blocks: list[bytes] = []
    for row in range(rows):
        for block_idx in range(blocks_per_row):
            synthetic_row = row * 17 + block_idx
            groups = bytearray()
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
                "iq2_xxs_gate_up_fused_calls": 0,
                "q2_iq2_reference_fallback_calls": 0,
                "cpu_token_sync_points": 0,
            },
        ),
        (
            GGML_TYPE_IQ2_XXS,
            _iq2_xxs_unit_block_payload(),
            {
                "quantized_expert_calls": 1,
                "q2_k_triton_calls": 0,
                "iq2_xxs_triton_calls": 1,
                "iq2_xxs_gate_up_fused_calls": 1,
                "q2_iq2_reference_fallback_calls": 0,
                "cpu_token_sync_points": 0,
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
    if ggml_type == GGML_TYPE_Q2_K:
        decoder = q2_k_matrix_from_gguf_payload
    else:
        decoder = iq2_xxs_matrix_from_gguf_payload
    expert_input = deepseek_q8_k_roundtrip_reference(hidden)
    gate = (
        decoder(gate_payload, rows=256, columns=256)
        .to(device="cuda")
        .matmul(expert_input)
    )
    up = (
        decoder(up_payload, rows=256, columns=256)
        .to(device="cuda")
        .matmul(expert_input)
    )
    activated = F.silu(torch.clamp(gate, max=10.0)) * torch.clamp(
        up,
        min=-10.0,
        max=10.0,
    )
    activated = deepseek_q8_k_roundtrip_reference(activated)
    expected = (
        decoder(down_payload, rows=256, columns=256).to(device="cuda").matmul(activated)
    )
    torch.testing.assert_close(actual, expected, rtol=8e-2, atol=8e-2)
    assert backend.stats() == expected_stats


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_quantized_expert_gemm_matches_reference_with_iq2_gate_up_and_q2_down() -> None:
    backend = DeepSeekV4FlashGPUBackend()
    rows = 256
    columns = 256
    hidden = torch.linspace(-0.5, 0.5, columns, dtype=torch.float32, device="cuda")
    gate_payload = _iq2_xxs_deterministic_payload(rows)
    up_payload = _iq2_xxs_deterministic_payload(rows)
    down_payload = _q2_k_deterministic_payload(rows)

    actual = backend.quantized_expert_gemm(
        hidden=hidden,
        gate_payload=_staged_payload(
            ggml_type=GGML_TYPE_IQ2_XXS,
            rows=rows,
            columns=columns,
            payload=gate_payload,
        ),
        up_payload=_staged_payload(
            ggml_type=GGML_TYPE_IQ2_XXS,
            rows=rows,
            columns=columns,
            payload=up_payload,
        ),
        down_payload=_staged_payload(
            ggml_type=GGML_TYPE_Q2_K,
            rows=rows,
            columns=columns,
            payload=down_payload,
        ),
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
    activated = F.silu(torch.clamp(gate, max=10.0)) * torch.clamp(
        up,
        min=-10.0,
        max=10.0,
    )
    expected = (
        q2_k_matrix_from_gguf_payload(
            down_payload,
            rows=rows,
            columns=columns,
        )
        .to(device="cuda")
        .matmul(activated)
    )

    assert actual.shape == (rows,)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=8e-2, atol=8e-2)
    assert backend.stats() == {
        "quantized_expert_calls": 1,
        "q2_k_triton_calls": 1,
        "iq2_xxs_triton_calls": 0,
        "iq2_xxs_gate_up_fused_calls": 1,
        "q2_iq2_reference_fallback_calls": 0,
        "cpu_token_sync_points": 0,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_quantized_selected_experts_gemm_matches_existing_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 256
    columns = 512
    hidden = torch.linspace(-0.5, 0.5, columns, dtype=torch.float32, device="cuda")
    expert_weights = torch.full((6,), 1.0 / 6.0, dtype=torch.float32, device="cuda")
    payloads = [
        (
            expert_id,
            _staged_payload(
                ggml_type=GGML_TYPE_IQ2_XXS,
                rows=rows,
                columns=columns,
                payload=_iq2_xxs_deterministic_payload_blocks(
                    rows,
                    columns // 256,
                ),
            ),
            _staged_payload(
                ggml_type=GGML_TYPE_IQ2_XXS,
                rows=rows,
                columns=columns,
                payload=_iq2_xxs_deterministic_payload_blocks(
                    rows,
                    columns // 256,
                ),
            ),
            _staged_payload(
                ggml_type=GGML_TYPE_Q2_K,
                rows=rows,
                columns=rows,
                payload=_q2_k_deterministic_payload(rows),
            ),
        )
        for expert_id in range(6)
    ]
    reference_backend = DeepSeekV4FlashGPUBackend()
    expected = sum(
        expert_weights[index]
        * reference_backend.quantized_expert_gemm(
            hidden=hidden,
            gate_payload=gate_payload,
            up_payload=up_payload,
            down_payload=down_payload,
        ).to(torch.float32)
        for index, (
            _expert_id,
            gate_payload,
            up_payload,
            down_payload,
        ) in enumerate(payloads)
    )

    roundtrips_by_width: dict[int, int] = {}

    def counted_roundtrip(tensor: torch.Tensor) -> torch.Tensor:
        roundtrips_by_width[tensor.numel()] = (
            roundtrips_by_width.get(
                tensor.numel(),
                0,
            )
            + 1
        )
        return deepseek_q8_k_roundtrip_reference(tensor)

    monkeypatch.setattr(
        backend_module,
        "deepseek_q8_k_roundtrip_reference",
        counted_roundtrip,
    )
    backend = DeepSeekV4FlashGPUBackend()
    actual = backend.quantized_selected_experts_gemm(
        hidden=hidden,
        expert_weights=expert_weights,
        payloads=payloads,
    )

    torch.testing.assert_close(actual, expected, rtol=1.0e-4, atol=1.0e-4)
    assert roundtrips_by_width == {columns: 1, rows: 6}
    assert backend.stats() == {
        "quantized_expert_calls": 6,
        "q2_k_triton_calls": 6,
        "iq2_xxs_triton_calls": 0,
        "iq2_xxs_gate_up_fused_calls": 6,
        "q2_iq2_reference_fallback_calls": 0,
        "cpu_token_sync_points": 0,
        "fused_selected_expert_api_calls": 1,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_quantized_expert_gemm_uses_fused_iq2_gate_up_for_multiblock_columns() -> None:
    backend = DeepSeekV4FlashGPUBackend()
    rows = 256
    gate_columns = 512
    down_columns = rows
    hidden = torch.linspace(
        -0.5,
        0.5,
        gate_columns,
        dtype=torch.float32,
        device="cuda",
    )
    gate_payload = _iq2_xxs_deterministic_payload_blocks(
        rows,
        gate_columns // 256,
    )
    up_payload = _iq2_xxs_deterministic_payload_blocks(
        rows,
        gate_columns // 256,
    )
    down_payload = _q2_k_deterministic_payload(rows)

    actual = backend.quantized_expert_gemm(
        hidden=hidden,
        gate_payload=_staged_payload(
            ggml_type=GGML_TYPE_IQ2_XXS,
            rows=rows,
            columns=gate_columns,
            payload=gate_payload,
        ),
        up_payload=_staged_payload(
            ggml_type=GGML_TYPE_IQ2_XXS,
            rows=rows,
            columns=gate_columns,
            payload=up_payload,
        ),
        down_payload=_staged_payload(
            ggml_type=GGML_TYPE_Q2_K,
            rows=rows,
            columns=down_columns,
            payload=down_payload,
        ),
    )
    gate = (
        iq2_xxs_matrix_from_gguf_payload(
            gate_payload,
            rows=rows,
            columns=gate_columns,
        )
        .to(device="cuda")
        .matmul(hidden)
    )
    up = (
        iq2_xxs_matrix_from_gguf_payload(
            up_payload,
            rows=rows,
            columns=gate_columns,
        )
        .to(device="cuda")
        .matmul(hidden)
    )
    activated = F.silu(torch.clamp(gate, max=10.0)) * torch.clamp(
        up,
        min=-10.0,
        max=10.0,
    )
    expected = (
        q2_k_matrix_from_gguf_payload(
            down_payload,
            rows=rows,
            columns=down_columns,
        )
        .to(device="cuda")
        .matmul(activated)
    )

    assert actual.shape == (rows,)
    torch.testing.assert_close(actual, expected, rtol=8e-2, atol=8e-2)
    assert backend.stats() == {
        "quantized_expert_calls": 1,
        "q2_k_triton_calls": 1,
        "iq2_xxs_triton_calls": 0,
        "iq2_xxs_gate_up_fused_calls": 1,
        "q2_iq2_reference_fallback_calls": 0,
        "cpu_token_sync_points": 0,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_quantized_selected_experts_gemm_matches_loop() -> None:
    rows = 256
    columns = 512
    hidden = torch.linspace(-0.5, 0.5, columns, dtype=torch.float32, device="cuda")
    expert_weights = torch.full((6,), 1.0 / 6.0, dtype=torch.float32, device="cuda")
    payloads = [
        (
            expert_id,
            _staged_payload(
                ggml_type=GGML_TYPE_IQ2_XXS,
                rows=rows,
                columns=columns,
                payload=_iq2_xxs_deterministic_payload_blocks(rows, columns // 256),
            ),
            _staged_payload(
                ggml_type=GGML_TYPE_IQ2_XXS,
                rows=rows,
                columns=columns,
                payload=_iq2_xxs_deterministic_payload_blocks(rows, columns // 256),
            ),
            _staged_payload(
                ggml_type=GGML_TYPE_Q2_K,
                rows=rows,
                columns=rows,
                payload=_q2_k_deterministic_payload(rows),
            ),
        )
        for expert_id in range(6)
    ]
    ref_backend = DeepSeekV4FlashGPUBackend()
    expected = ref_backend.quantized_selected_experts_gemm(
        hidden=hidden,
        expert_weights=expert_weights,
        payloads=payloads,
    )
    backend = DeepSeekV4FlashGPUBackend()
    workspace = torch.empty((6, rows), dtype=torch.float32, device="cuda")
    actual = backend.fused_quantized_selected_experts_gemm(
        hidden=hidden,
        expert_weights=expert_weights,
        payloads=payloads,
        workspace=workspace,
    )
    # The fused path skips the Q8_K round-trip that the looped fallback
    # performs, so allow a slightly looser relative tolerance.
    torch.testing.assert_close(actual, expected, rtol=5.0e-3, atol=1.0e-4)


def test_model_keeps_kernel_execution_disabled_until_backend_ready() -> None:
    model = DeepSeekV4FlashForCausalLM(gpu_backend=DeepSeekV4FlashGPUBackend())

    assert model.kernel_execution_available is False
    with pytest.raises(NotImplementedError, match="kernel execution is not available"):
        model.forward_full(torch.tensor([1], dtype=torch.long), use_kernel=True)
