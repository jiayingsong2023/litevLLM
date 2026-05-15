# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from vllm.model_executor.models import gemma4


def test_gemma4_profile_flags_are_derived_from_tuning_env(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LAYER_PROFILE", "1")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_ROCTX_PROFILE", "1")

    gemma4.set_gemma4_tuning_config(
        {
            "FASTINFERENCE_GEMMA4_LAYER_PROFILE": "0",
            "FASTINFERENCE_GEMMA4_ROCTX_PROFILE": "0",
        },
        locked=True,
    )

    assert not gemma4._GEMMA4_PROFILE_ENABLED
    assert not gemma4._GEMMA4_ROCTX_PROFILE_ENABLED


def test_gemma4_local_decode_triton_reads_attn_config_tuning_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON", "0")
    inf_config = SimpleNamespace(
        tuning_env={"FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON": "1"}
    )

    assert gemma4._gemma4_config_truthy_default_on(
        inf_config,
        "FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON",
    )


def test_gemma4_local_decode_triton_ignores_env_without_config(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON", "0")

    assert gemma4._gemma4_config_truthy_default_on(
        None,
        "FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON",
    )


def test_gemma4_full_decode_ref_reads_attn_config_tuning_env(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN", "0")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN", "0")
    inf_config = SimpleNamespace(
        tuning_env={"FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN": "1"}
    )

    assert gemma4._gemma4_config_truthy_default_off(
        inf_config,
        "FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN",
    )


def test_gemma4_full_decode_ref_legacy_fp16_reads_attn_config_tuning_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN", "0")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN", "0")
    inf_config = SimpleNamespace(
        tuning_env={"FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN": "1"}
    )

    assert gemma4._gemma4_config_truthy_default_off(
        inf_config,
        "FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN",
    )


def test_gemma4_full_decode_ref_ignores_env_without_config(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN", "1")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN", "1")

    assert not gemma4._gemma4_config_truthy_default_off(
        None,
        "FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN",
    )
    assert not gemma4._gemma4_config_truthy_default_off(
        None,
        "FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN",
    )


def test_gemma4_legacy_full_precision_kv_write_reads_attn_config_tuning_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_FULLPREC_KV_WRITE", "0")
    inf_config = SimpleNamespace(
        tuning_env={"FASTINFERENCE_GEMMA4_LEGACY_FULLPREC_KV_WRITE": "1"}
    )

    assert gemma4._gemma4_config_truthy_default_off(
        inf_config,
        "FASTINFERENCE_GEMMA4_LEGACY_FULLPREC_KV_WRITE",
    )


def test_gemma4_legacy_full_precision_kv_write_ignores_env_without_config(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_FULLPREC_KV_WRITE", "1")

    assert not gemma4._gemma4_config_truthy_default_off(
        None,
        "FASTINFERENCE_GEMMA4_LEGACY_FULLPREC_KV_WRITE",
    )


def test_gemma4_legacy_item_path_reads_attn_config_tuning_env(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH", "0")
    inf_config = SimpleNamespace(
        tuning_env={"FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH": "1"}
    )

    assert gemma4._gemma4_config_truthy_default_off(
        inf_config,
        "FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH",
    )


def test_gemma4_legacy_item_path_ignores_env_without_config(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH", "1")

    assert not gemma4._gemma4_config_truthy_default_off(
        None,
        "FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH",
    )


def test_gemma4_rope_cache_max_pos_uses_runtime_config(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_ROPE_CACHE_MAX_POS", "1024")
    runtime_config = SimpleNamespace(gemma4_rope_cache_max_pos=1536)
    config = SimpleNamespace(max_position_embeddings=4096)

    assert gemma4._resolve_gemma4_rope_cache_max_pos(config, runtime_config) == 1536


def test_gemma4_rope_cache_max_pos_ignores_env_without_runtime_config(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_ROPE_CACHE_MAX_POS", "1024")
    config = SimpleNamespace(max_position_embeddings=4096)

    assert gemma4._resolve_gemma4_rope_cache_max_pos(config, None) == 4096


def test_gemma4_rope_cache_pool_limit_uses_runtime_config(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_ROPE_CACHE_POOL_MAX", "2")
    runtime_config = SimpleNamespace(gemma4_rope_cache_pool_max=12)

    assert gemma4._resolve_gemma4_rope_cache_pool_limit(runtime_config) == 12


def test_gemma4_rope_cache_pool_limit_ignores_env_without_runtime_config(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_ROPE_CACHE_POOL_MAX", "2")

    assert gemma4._resolve_gemma4_rope_cache_pool_limit(None) == 8


def test_gemma4_fp32_residual_guard_reads_runtime_config(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD", "0")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_START", "8")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_SPAN", "3")
    runtime_config = SimpleNamespace(
        gemma4_26b_fp32_residual_guard_enabled=True,
        gemma4_26b_fp32_residual_guard_start=12,
        gemma4_26b_fp32_residual_guard_span=4,
    )

    assert gemma4._gemma4_fp32_residual_guard_policy(runtime_config) == (
        True,
        12,
        4,
    )


def test_gemma4_fp32_residual_guard_uses_defaults_without_runtime_config(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD", "1")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_START", "11")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_SPAN", "5")

    assert gemma4._gemma4_fp32_residual_guard_policy(None) == (False, 8, 3)
