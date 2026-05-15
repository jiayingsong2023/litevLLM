# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from vllm.engine.inference_config import LiteInferenceConfig
from vllm.engine.runtime_config import RuntimeConfig


def _mock_vllm_config() -> object:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="models/mock",
            tokenizer="models/mock",
            dtype="float16",
            max_model_len=1024,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=512,
        ),
        runtime_policy_mode="auto",
    )


def test_runtime_config_defaults_to_turbo_int4(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_KV_TYPE", raising=False)
    monkeypatch.delenv("FASTINFERENCE_KV_FP8", raising=False)
    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())
    assert cfg.kv_cache_dtype == "turbo_int4"


def test_runtime_config_auto_respects_legacy_fp8_toggle(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "auto")
    monkeypatch.setenv("FASTINFERENCE_KV_FP8", "1")
    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())
    assert cfg.kv_cache_dtype == "fp8"


def test_runtime_config_explicit_kv_type_wins(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "fp16")
    monkeypatch.setenv("FASTINFERENCE_KV_FP8", "1")
    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())
    assert cfg.kv_cache_dtype == "fp16"


def test_runtime_config_owns_factory_scheduler_env(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_MAX_DECODE_STREAK", "7")
    monkeypatch.setenv("FASTINFERENCE_QUEUE_AGING_THRESHOLD_S", "3.5")
    monkeypatch.setenv("FASTINFERENCE_SERVICE_CLASS_WEIGHTS", "latency=8,batch=2")
    monkeypatch.setenv("FASTINFERENCE_ADMISSION_SERVICE_CLASS_QUOTAS", "latency=1")
    monkeypatch.setenv(
        "FASTINFERENCE_FAIRNESS_GUARDRAIL_SERVICE_CLASSES",
        "latency,batch",
    )

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    scheduler = cfg.scheduler_policy
    assert scheduler.max_decode_streak == 7
    assert scheduler.queue_aging_threshold_s == 3.5
    assert scheduler.service_class_weights == {"latency": 8, "batch": 2}
    assert scheduler.admission_service_class_quotas == {"latency": 1}
    assert scheduler.fairness_guardrail_service_classes == {"latency", "batch"}


def test_runtime_config_owns_factory_backend_env(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PREFIX_CACHE_MAX_ENTRIES", "13")
    monkeypatch.setenv("FASTINFERENCE_PREEMPTION_MODE", "off")
    monkeypatch.setenv("FASTINFERENCE_PREEMPT_MIN_BACKLOG", "4")
    monkeypatch.setenv("FASTINFERENCE_PREEMPT_MIN_DECODES", "3")
    monkeypatch.setenv("FASTINFERENCE_PREEMPT_MAX_QUEUE_WAIT_S", "1.25")
    monkeypatch.setenv("FASTINFERENCE_PREEMPTIBLE_SERVICE_CLASSES", "throughput,batch")
    monkeypatch.setenv("FASTINFERENCE_PREEMPT_MULTIMODAL_PREFILLS", "1")
    monkeypatch.setenv("FASTINFERENCE_PREEMPT_MULTIMODAL_MAX_QUEUE_WAIT_S", "2.5")
    monkeypatch.setenv("FASTINFERENCE_MULTIMODAL_PREFIX_CACHE_PROTECT_THRESHOLD", "0.7")
    monkeypatch.setenv("FASTINFERENCE_GPU_GREEDY_SAMPLING", "1")
    monkeypatch.setenv("FASTINFERENCE_GPU_GREEDY_MAX_TOKENS_ONLY", "1")
    monkeypatch.setenv("FASTINFERENCE_GPU_GREEDY_BYPASS_CPU_POLICIES", "1")
    monkeypatch.setenv("FASTINFERENCE_GPU_GREEDY_IGNORE_EOS", "1")

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    backend = cfg.backend_policy
    assert backend.max_prefix_cache_entries == 13
    assert backend.preemption_mode == "off"
    assert backend.preemption_min_backlog == 4
    assert backend.preemption_min_decodes == 3
    assert backend.preemption_max_queue_wait_s == 1.25
    assert backend.preemptible_service_classes == {"throughput", "batch"}
    assert backend.preempt_multimodal_prefills is True
    assert backend.preempt_multimodal_max_queue_wait_s == 2.5
    assert backend.multimodal_prefix_cache_protect_threshold == 0.7
    assert backend.gpu_greedy_sampling is True
    assert backend.gpu_greedy_max_tokens_only is True
    assert backend.gpu_greedy_bypass_cpu_policies is True
    assert backend.gpu_greedy_ignore_eos is True


def test_runtime_config_owns_request_builder_default_min_new_tokens(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_LITE_DEFAULT_MIN_NEW_TOKENS", "3")

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.default_min_new_tokens == 3


def test_runtime_config_invalid_default_min_new_tokens_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_LITE_DEFAULT_MIN_NEW_TOKENS", "banana")

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.default_min_new_tokens == 1


def test_runtime_config_owns_lite_engine_operational_env(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_LITE_QUEUE_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("FASTINFERENCE_MEM_AUDIT_TOPN", "7")

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.queue_timeout_s == 12.5
    assert cfg.memory_audit_topn == 7


def test_runtime_config_invalid_mem_audit_topn_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_MEM_AUDIT_TOPN", "banana")

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.memory_audit_topn == 20


def test_runtime_config_owns_gemma4_fp32_residual_guard_env(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD", "1")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_START", "11")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_26B_FP32_RESIDUAL_GUARD_SPAN", "5")

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.gemma4_26b_fp32_residual_guard_enabled is True
    assert cfg.gemma4_26b_fp32_residual_guard_start == 11
    assert cfg.gemma4_26b_fp32_residual_guard_span == 5


def test_runtime_config_owns_gemma4_moe_expert_cache_env(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_MOE_EXPERT_CACHE_SIZE", "13")

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.gemma4_moe_expert_cache_size == 13


def test_runtime_config_owns_gemma4_rope_cache_env(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_ROPE_CACHE_MAX_POS", "1536")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_ROPE_CACHE_POOL_MAX", "12")

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.gemma4_rope_cache_max_pos == 1536
    assert cfg.gemma4_rope_cache_pool_max == 12


def test_inference_config_defaults_to_turbo_int4(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_KV_TYPE", raising=False)
    monkeypatch.delenv("FASTINFERENCE_KV_FP8", raising=False)
    cfg = LiteInferenceConfig.from_env()
    assert cfg.kv_type == "turbo_int4"


def test_inference_config_auto_respects_legacy_fp8_toggle(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "auto")
    monkeypatch.setenv("FASTINFERENCE_KV_FP8", "1")
    cfg = LiteInferenceConfig.from_env()
    assert cfg.kv_type == "fp8"
