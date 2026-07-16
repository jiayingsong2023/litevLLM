# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from vllm.adapters.gemma4 import Gemma4Adapter
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


def test_gemma4_tuning_config_rejects_migrated_production_policy_names() -> None:
    config = gemma4.set_gemma4_tuning_config(
        {
            "FASTINFERENCE_GEMMA4_LAYER_PROFILE": "1",
            "FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON": "0",
            "FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION": "0",
            "FASTINFERENCE_KV_MAX_MODEL_LEN": "2048",
        },
        locked=True,
    )

    assert config.tuning == {
        "FASTINFERENCE_GEMMA4_LAYER_PROFILE": "1",
    }


def test_gemma4_adapter_exposes_production_model_and_kernel_policy() -> None:
    runtime_config = SimpleNamespace(
        tuning_env={},
        kv_cache_dtype="fp8",
        gemma4_26b_fp32_residual_guard_enabled=True,
        gemma4_26b_fp32_residual_guard_start=10,
        gemma4_26b_fp32_residual_guard_span=2,
        gemma4_moe_expert_cache_size=24,
        gemma4_moe_compute_dtype="fp16",
        gemma4_moe_int4_kernel_enabled=False,
        gemma4_moe_int4_kernel_strategy="batched",
        gemma4_moe_prefill_grouped_enabled=True,
        gemma4_moe_prefill_grouped_min_tokens=33,
        gemma4_moe_prefill_grouped_strategy="fused",
        gemma4_moe_batch_materialize_enabled=True,
        gemma4_rope_cache_max_pos=1536,
        gemma4_rope_cache_pool_max=12,
    )
    model_config = SimpleNamespace(hf_config=SimpleNamespace())

    policy = Gemma4Adapter().runtime_policy(model_config, runtime_config)

    assert policy.model_policy == {
        "local_decode_triton": True,
        "force_full_ref_attn": False,
        "legacy_fp16_ref_attn": False,
        "legacy_fullprec_kv_write": False,
        "legacy_item_path": False,
        "mlp_pair_fusion": True,
        "fp32_residual_guard_enabled": True,
        "fp32_residual_guard_start": 10,
        "fp32_residual_guard_span": 2,
        "moe_expert_cache_size": 24,
        "moe_compute_dtype": "fp16",
        "moe_int4_kernel_enabled": False,
        "moe_int4_kernel_strategy": "batched",
        "moe_prefill_grouped_enabled": True,
        "moe_prefill_grouped_min_tokens": 33,
        "moe_prefill_grouped_strategy": "fused",
        "moe_batch_materialize_enabled": True,
        "rope_cache_max_pos": 1536,
        "rope_cache_pool_max": 12,
    }
    assert policy.kernel_policy["awq_fused_scope"] == "all"
    assert policy.kernel_policy["awq_fused_gemm"] is True
    assert policy.kernel_policy["awq_decode_gemv"] is True
    assert policy.kernel_policy["awq_fused_gate_up"] is True
    assert policy.kernel_policy["awq_group32_gemv_all"] is True
    assert policy.kernel_policy["gemma4_dense_down_proj"] is True


def test_gemma4_12b_adapter_selects_verified_m1_fast_paths() -> None:
    runtime_config = SimpleNamespace(tuning_env={}, kv_cache_dtype="fp8")
    text_config = SimpleNamespace(
        hidden_size=3840,
        num_hidden_layers=48,
        intermediate_size=15360,
        num_attention_heads=16,
    )
    model_config = SimpleNamespace(hf_config=SimpleNamespace(text_config=text_config))

    policy = Gemma4Adapter().runtime_policy(model_config, runtime_config)

    assert "awq_rows_exact_msmall" not in policy.kernel_policy
    assert policy.kernel_policy["awq_fused_gate_up"] is True
    assert policy.kernel_policy["awq_fused_gate_up_group32"] is True
    assert policy.kernel_policy["gemma4_dense_down_proj"] is True
    assert policy.model_policy["mlp_pair_fusion"] is True
    assert policy.model_policy["gemma4_c1_preset"] is True
    assert policy.verified_decode_batch_sizes == (1,)


def test_gemma4_local_decode_triton_reads_attn_config_model_policy(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON", "0")
    inf_config = SimpleNamespace(model_policy={"local_decode_triton": True})

    assert gemma4._gemma4_model_policy_truthy(
        inf_config,
        "local_decode_triton",
        default=True,
    )


def test_gemma4_local_decode_triton_ignores_env_without_config(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON", "0")

    assert gemma4._gemma4_model_policy_truthy(
        None,
        "local_decode_triton",
        default=True,
    )


def test_gemma4_full_decode_ref_reads_attn_config_model_policy(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN", "0")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN", "0")
    inf_config = SimpleNamespace(model_policy={"force_full_ref_attn": True})

    assert gemma4._gemma4_model_policy_truthy(
        inf_config,
        "force_full_ref_attn",
        default=False,
    )


def test_gemma4_full_decode_ref_legacy_fp16_reads_attn_config_model_policy(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN", "0")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN", "0")
    inf_config = SimpleNamespace(model_policy={"legacy_fp16_ref_attn": True})

    assert gemma4._gemma4_model_policy_truthy(
        inf_config,
        "legacy_fp16_ref_attn",
        default=False,
    )


def test_gemma4_full_decode_ref_ignores_env_without_config(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN", "1")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN", "1")

    assert not gemma4._gemma4_model_policy_truthy(
        None,
        "force_full_ref_attn",
        default=False,
    )
    assert not gemma4._gemma4_model_policy_truthy(
        None,
        "legacy_fp16_ref_attn",
        default=False,
    )


def test_gemma4_legacy_full_precision_kv_write_reads_attn_config_model_policy(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_FULLPREC_KV_WRITE", "0")
    inf_config = SimpleNamespace(model_policy={"legacy_fullprec_kv_write": True})

    assert gemma4._gemma4_model_policy_truthy(
        inf_config,
        "legacy_fullprec_kv_write",
        default=False,
    )


def test_gemma4_legacy_full_precision_kv_write_ignores_env_without_config(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_FULLPREC_KV_WRITE", "1")

    assert not gemma4._gemma4_model_policy_truthy(
        None,
        "legacy_fullprec_kv_write",
        default=False,
    )


def test_gemma4_legacy_item_path_reads_attn_config_model_policy(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH", "0")
    inf_config = SimpleNamespace(model_policy={"legacy_item_path": True})

    assert gemma4._gemma4_model_policy_truthy(
        inf_config,
        "legacy_item_path",
        default=False,
    )


def test_gemma4_legacy_item_path_ignores_env_without_config(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH", "1")

    assert not gemma4._gemma4_model_policy_truthy(
        None,
        "legacy_item_path",
        default=False,
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


def test_gemma4_model_policy_covers_all_runtime_config_fields() -> None:
    """Every RuntimeConfig.gemma4_* field should have a corresponding
    model_policy key from Gemma4Adapter."""
    from vllm.adapters.policy_keys import (
        GEMMA4_C1_PRESET,
        GEMMA4_FP32_RESIDUAL_GUARD_ENABLED,
        GEMMA4_FP32_RESIDUAL_GUARD_SPAN,
        GEMMA4_FP32_RESIDUAL_GUARD_START,
        GEMMA4_MOE_BATCH_MATERIALIZE_ENABLED,
        GEMMA4_MOE_COMPUTE_DTYPE,
        GEMMA4_MOE_EXPERT_CACHE_SIZE,
        GEMMA4_MOE_INT4_KERNEL_ENABLED,
        GEMMA4_MOE_INT4_KERNEL_STRATEGY,
        GEMMA4_MOE_PREFILL_GROUPED_ENABLED,
        GEMMA4_MOE_PREFILL_GROUPED_MIN_TOKENS,
        GEMMA4_MOE_PREFILL_GROUPED_STRATEGY,
        GEMMA4_ROPE_CACHE_MAX_POS,
        GEMMA4_ROPE_CACHE_POOL_MAX,
    )

    RUNTIME_CONFIG_TO_POLICY_KEY = {
        "gemma4_26b_fp32_residual_guard_enabled": GEMMA4_FP32_RESIDUAL_GUARD_ENABLED,
        "gemma4_26b_fp32_residual_guard_start": GEMMA4_FP32_RESIDUAL_GUARD_START,
        "gemma4_26b_fp32_residual_guard_span": GEMMA4_FP32_RESIDUAL_GUARD_SPAN,
        "gemma4_moe_expert_cache_size": GEMMA4_MOE_EXPERT_CACHE_SIZE,
        "gemma4_moe_compute_dtype": GEMMA4_MOE_COMPUTE_DTYPE,
        "gemma4_moe_int4_kernel_enabled": GEMMA4_MOE_INT4_KERNEL_ENABLED,
        "gemma4_moe_int4_kernel_strategy": GEMMA4_MOE_INT4_KERNEL_STRATEGY,
        "gemma4_moe_prefill_grouped_enabled": GEMMA4_MOE_PREFILL_GROUPED_ENABLED,
        "gemma4_moe_prefill_grouped_min_tokens": GEMMA4_MOE_PREFILL_GROUPED_MIN_TOKENS,
        "gemma4_moe_prefill_grouped_strategy": GEMMA4_MOE_PREFILL_GROUPED_STRATEGY,
        "gemma4_moe_batch_materialize_enabled": GEMMA4_MOE_BATCH_MATERIALIZE_ENABLED,
        "gemma4_rope_cache_max_pos": GEMMA4_ROPE_CACHE_MAX_POS,
        "gemma4_rope_cache_pool_max": GEMMA4_ROPE_CACHE_POOL_MAX,
        "gemma4_c1_preset": GEMMA4_C1_PRESET,
    }

    # Verify all gemma4_* RuntimeConfig fields have a mapping
    assert len(RUNTIME_CONFIG_TO_POLICY_KEY) == 14, (
        f"Expected 14 gemma4_* fields, got {len(RUNTIME_CONFIG_TO_POLICY_KEY)}"
    )
