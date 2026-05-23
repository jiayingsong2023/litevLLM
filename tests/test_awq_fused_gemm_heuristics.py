# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

from vllm.kernels.triton.awq_fused_gemm import (
    _env_fused_gemm_autotune_tool_override,
    _fused_gemm_blocks_tool_override,
    _load_persistent_profile,
    _lookup_persistent_blocks,
    _persistent_profile_path,
    _resolve_packed_int4_fused_gemm_blocks,
    _select_fused_gemm_blocks,
    set_awq_fused_tuning_config,
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


def test_select_fused_gemm_blocks_ignores_env_tile_override(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_M", "7")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_N", "96")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_K", "48")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_NUM_WARPS", "3")
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_GEMM_NUM_STAGES", "4")
    set_awq_fused_tuning_config({})

    assert _select_fused_gemm_blocks(1, 8192, 5376) == (1, 256, 64, 8, 2)


def test_tool_only_fused_gemm_tile_override_still_supported() -> None:
    set_awq_fused_tuning_config(
        {
            "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_M": "7",
            "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_N": "96",
            "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_K": "48",
            "FASTINFERENCE_AWQ_FUSED_GEMM_NUM_WARPS": "3",
            "FASTINFERENCE_AWQ_FUSED_GEMM_NUM_STAGES": "4",
        },
        locked=False,
    )

    assert _fused_gemm_blocks_tool_override() == (7, 96, 48, 3, 4)


def test_env_fused_gemm_autotune_auto_prefers_small_m_decode_autotune() -> None:
    set_awq_fused_tuning_config({})
    assert _env_fused_gemm_autotune_tool_override(1, 8192, 5376) is False
    assert _env_fused_gemm_autotune_tool_override(4, 8192, 5376) is True
    assert _env_fused_gemm_autotune_tool_override(128, 21504, 5376) is False
    assert _env_fused_gemm_autotune_tool_override(256, 2048, 4096) is True


def test_env_fused_gemm_autotune_explicit_override() -> None:
    set_awq_fused_tuning_config({"FASTINFERENCE_AWQ_FUSED_AUTOTUNE": "1"})
    assert _env_fused_gemm_autotune_tool_override(1, 8192, 5376) is False
    set_awq_fused_tuning_config({"FASTINFERENCE_AWQ_FUSED_AUTOTUNE": "off"})
    assert _env_fused_gemm_autotune_tool_override(256, 2048, 4096) is False


def test_persistent_profile_defaults_to_bundled_profile(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_AWQ_FUSED_PROFILE_JSON", raising=False)
    set_awq_fused_tuning_config({})

    path = _persistent_profile_path()
    profile = _load_persistent_profile()

    assert path.name == "awq_fused_profile.json"
    assert path.is_file()
    assert profile["packed_int4_symmetric"]


def test_lookup_persistent_blocks_uses_bundled_profile(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_AWQ_FUSED_PROFILE_JSON", raising=False)
    set_awq_fused_tuning_config({})

    assert _lookup_persistent_blocks(
        "packed_int4_symmetric",
        m=2,
        n=8192,
        k=5376,
        group_size=32,
    ) == (16, 256, 64, 8, 2)


def test_resolve_packed_int4_fused_gemm_blocks_prefers_persistent_profile(
    tmp_path,
) -> None:
    override_path = tmp_path / "awq_profile.json"
    override_path.write_text(
        json.dumps(
            {
                "version": 1,
                "packed_int4_symmetric": [
                    {
                        "m_min": 2,
                        "m_max": 8,
                        "n": 1234,
                        "k": 5678,
                        "group_size": 32,
                        "block_m": 48,
                        "block_n": 80,
                        "block_k": 40,
                        "num_warps": 5,
                        "num_stages": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    set_awq_fused_tuning_config(
        {"FASTINFERENCE_AWQ_FUSED_PROFILE_JSON": str(override_path)},
        locked=True,
    )

    assert _resolve_packed_int4_fused_gemm_blocks(
        m=4,
        n=1234,
        k=5678,
        group_size=32,
        split_k=1,
    ) == (48, 80, 40, 5, 3)


def test_persistent_profile_env_override_and_disable(monkeypatch, tmp_path) -> None:
    override_path = tmp_path / "awq_profile.json"
    override_path.write_text(
        json.dumps(
            {
                "version": 1,
                "packed_int4_symmetric": [
                    {
                        "m_min": 2,
                        "m_max": 8,
                        "n": 1234,
                        "k": 5678,
                        "group_size": 32,
                        "block_m": 32,
                        "block_n": 64,
                        "block_k": 64,
                        "num_warps": 4,
                        "num_stages": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_PROFILE_JSON", str(override_path))
    set_awq_fused_tuning_config({})

    assert _persistent_profile_path() == override_path
    assert _lookup_persistent_blocks(
        "packed_int4_symmetric",
        m=4,
        n=1234,
        k=5678,
        group_size=32,
    ) == (32, 64, 64, 4, 1)

    monkeypatch.setenv("FASTINFERENCE_AWQ_FUSED_PROFILE_JSON", "off")
    set_awq_fused_tuning_config({})

    assert _persistent_profile_path() == Path()
    assert _load_persistent_profile() == {}
