# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from vllm.adapters.base import ModelCapabilities, RuntimeModelPolicy
from vllm.adapters.gemma4 import Gemma4Adapter
from vllm.adapters.qwen3_5 import Qwen35Adapter
from vllm.adapters.registry import get_model_adapter
from vllm.engine.runtime_config import RuntimeConfig
from vllm.engine.runtime_planner import RuntimePlanner


def _runtime_config(**overrides: object) -> RuntimeConfig:
    values = dict(
        model_path="models/mock",
        tokenizer_path="models/mock",
        dtype="float16",
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=0,
        block_size=16,
        kv_cache_dtype="turbo_int4",
        kv_max_model_len=None,
        kv_max_active_requests=4,
        fusion_level=2,
        policy_mode="auto",
        enable_decode_priority=True,
        prefill_chunk_size=0,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        min_prefill_chunk_size=128,
        max_prefill_chunk_size=2048,
        prefill_sla_ttft_ms=2000.0,
        tuning_env={},
    )
    values.update(overrides)
    return RuntimeConfig(**values)


def _caps(model_type: str = "llama") -> ModelCapabilities:
    return ModelCapabilities(
        model_type=model_type,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        max_model_len=4096,
        supports_moe=False,
        supports_fp8_kv=True,
        supports_int4_kv=True,
        supports_paged_prefill=True,
        preferred_kv_dtype="float16",
    )


def test_gemma4_runtime_policy_forces_fp8_kv_and_sets_fused_defaults() -> None:
    cfg = _runtime_config(kv_cache_dtype="turbo_int4", tuning_env={})

    policy = Gemma4Adapter().runtime_policy(SimpleNamespace(), cfg)

    assert policy.force_kv_cache_dtype == "fp8"
    assert policy.force_kv_cache_dtype_when == ("turbo_int4", "int4")
    assert policy.tuning_env_overrides["FASTINFERENCE_AWQ_FUSED_SCOPE"] == "all"
    assert policy.tuning_env_overrides["FASTINFERENCE_AWQ_FUSED_GEMM"] == "1"
    assert policy.tuning_env_overrides["FASTINFERENCE_AWQ_FUSED_GEMM_FORCE"] == "0"


def test_gemma4_runtime_policy_honors_int4_kv_override() -> None:
    cfg = _runtime_config(
        kv_cache_dtype="turbo_int4",
        tuning_env={"FASTINFERENCE_GEMMA4_ALLOW_INT4_KV": "1"},
    )

    policy = Gemma4Adapter().runtime_policy(SimpleNamespace(), cfg)

    assert policy.force_kv_cache_dtype is None


def test_gemma4_detect_reports_moe_for_26b_a4b_like_config() -> None:
    hf_config = SimpleNamespace(
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        num_experts=128,
        top_k_experts=8,
        moe_intermediate_size=704,
    )
    model_config = SimpleNamespace(
        hf_config=hf_config,
        get_num_kv_heads=lambda _: 8,
        get_head_size=lambda: 256,
        get_num_layers=lambda _: 30,
        get_max_model_len=lambda: 512,
    )

    caps = Gemma4Adapter().detect(SimpleNamespace(), model_config)

    assert caps.supports_moe is True


def test_qwen_runtime_policy_owns_prefill_chunk_preference() -> None:
    policy = Qwen35Adapter().runtime_policy(SimpleNamespace(), _runtime_config())

    assert policy.prefill_chunk_size_high_end == 2048
    assert policy.prefill_chunk_size_standard == 1024


def test_registry_detects_gemma4_text_config_before_model_load() -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="paligemma",
            text_config=SimpleNamespace(model_type="gemma4_text"),
        )
    )

    adapter = get_model_adapter(None, model_config)

    assert isinstance(adapter, Gemma4Adapter)


def test_registry_detects_qwen35_text_config_before_model_load() -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="wrapper",
            text_config=SimpleNamespace(model_type="qwen3_5_text"),
        )
    )

    adapter = get_model_adapter(None, model_config)

    assert isinstance(adapter, Qwen35Adapter)


def test_runtime_planner_uses_policy_prefill_chunks(monkeypatch) -> None:
    cfg = _runtime_config(prefill_chunk_size=0)
    policy = RuntimeModelPolicy(
        prefill_chunk_size_high_end=1536,
        prefill_chunk_size_standard=768,
    )
    monkeypatch.setattr(
        "vllm.engine.runtime_planner.get_total_gpu_memory_gb", lambda: 16.0
    )

    plan = RuntimePlanner(cfg, _caps("qwen3_5"), policy).build_execution_plan(4)

    assert plan.prefill_chunk_size == 768


def test_runtime_planner_honors_explicit_prefill_chunk_over_policy(monkeypatch) -> None:
    cfg = _runtime_config(prefill_chunk_size=256)
    policy = RuntimeModelPolicy(
        prefill_chunk_size_high_end=1536,
        prefill_chunk_size_standard=768,
    )
    monkeypatch.setattr(
        "vllm.engine.runtime_planner.get_total_gpu_memory_gb", lambda: 64.0
    )

    plan = RuntimePlanner(cfg, _caps("qwen3_5"), policy).build_execution_plan(4)

    assert plan.prefill_chunk_size == 256


def test_lite_engine_does_not_import_model_specific_tuning_installers() -> None:
    source = Path("vllm/engine/lite_engine.py").read_text(encoding="utf-8")

    assert "vllm.model_executor.models.gemma4" not in source
    assert "vllm.model_executor.models.qwen3_5" not in source
