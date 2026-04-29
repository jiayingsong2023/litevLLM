# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.kernels.triton.awq_fused_gemm import (
    _env_fused_gemm_autotune,
    _select_fused_gemm_blocks,
)


def test_select_fused_gemm_blocks_prefers_wider_n_for_gemma_decode_local_q() -> None:
    assert _select_fused_gemm_blocks(1, 8192, 5376) == (1, 256, 64, 8, 2)
    assert _select_fused_gemm_blocks(4, 8192, 5376) == (16, 256, 64, 8, 2)


def test_select_fused_gemm_blocks_keeps_128_n_for_very_wide_decode_outputs() -> None:
    assert _select_fused_gemm_blocks(1, 16384, 5376) == (1, 128, 64, 8, 2)
    assert _select_fused_gemm_blocks(4, 21504, 5376) == (16, 128, 64, 8, 2)


def test_select_fused_gemm_blocks_uses_32x256_for_gemma_prefill_shapes() -> None:
    assert _select_fused_gemm_blocks(128, 8192, 5376) == (32, 256, 64, 8, 2)
    assert _select_fused_gemm_blocks(128, 5376, 21504) == (64, 64, 64, 8, 2)


def test_select_fused_gemm_blocks_deep_k_narrow_output_decode() -> None:
    # m in {1,2} is the decode-hot MLP down_proj bucket on Gemma4-31B.
    assert _select_fused_gemm_blocks(1, 5376, 21504) == (1, 128, 64, 4, 1)
    assert _select_fused_gemm_blocks(2, 5376, 21504) == (16, 128, 64, 4, 1)
    assert _select_fused_gemm_blocks(4, 5376, 21504) == (16, 128, 64, 8, 2)


def test_env_fused_gemm_autotune_auto_prefers_small_m_decode_autotune(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_AWQ_FUSED_AUTOTUNE", raising=False)
    assert _env_fused_gemm_autotune(1, 8192, 5376) is False
    assert _env_fused_gemm_autotune(4, 8192, 5376) is True
    assert _env_fused_gemm_autotune(128, 21504, 5376) is False
    assert _env_fused_gemm_autotune(256, 2048, 4096) is True


def test_env_fused_gemm_autotune_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_AUTOTUNE", "1")
    assert _env_fused_gemm_autotune(1, 8192, 5376) is False
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_AUTOTUNE", "off")
    assert _env_fused_gemm_autotune(256, 2048, 4096) is False
