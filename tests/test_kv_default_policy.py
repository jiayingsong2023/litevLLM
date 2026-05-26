# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from vllm.engine.fastinference_config import FastInferenceConfig, LegacyEnvConfig
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
        fastinference_config=FastInferenceConfig(),
    )


def test_runtime_config_uses_profile_defaults(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_PROFILE", raising=False)
    monkeypatch.delenv("FASTINFERENCE_ALLOW_LEGACY_ENV", raising=False)
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_ALLOW_INT4_KV", "1")
    monkeypatch.delenv("FASTINFERENCE_KV_TYPE", raising=False)

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.profile.requested_name == "auto"
    assert cfg.profile.effective_name == "benchmark"
    assert cfg.kv_cache_dtype == "turbo_int4"
    assert cfg.block_size == 16
    assert cfg.fusion_level == 2
    assert cfg.backend_policy.gpu_greedy_sampling is True
    assert cfg.tuning_env == {"FASTINFERENCE_PROFILE": "auto"}


def test_runtime_config_config_kv_type_overrides_profile(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "fp16")
    vllm_config = _mock_vllm_config()
    vllm_config.fastinference_config = FastInferenceConfig(kv_type="fp8")

    cfg = RuntimeConfig.from_vllm_config(vllm_config)

    assert cfg.profile.requested_name == "auto"
    assert cfg.kv_cache_dtype == "fp8"
    assert cfg.tuning_env == {"FASTINFERENCE_PROFILE": "auto"}


def test_runtime_config_config_kv_type_auto_uses_profile(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "benchmark")
    vllm_config = _mock_vllm_config()
    vllm_config.fastinference_config = FastInferenceConfig(
        profile="accuracy",
        kv_type="auto",
    )

    cfg = RuntimeConfig.from_vllm_config(vllm_config)

    assert cfg.profile.requested_name == "accuracy"
    assert cfg.kv_cache_dtype == "fp8"


def test_runtime_config_collects_deprecated_env_when_compat_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_ALLOW_LEGACY_ENV", "1")
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_ALLOW_INT4_KV", "1")
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "fp16")
    vllm_config = _mock_vllm_config()
    vllm_config.fastinference_config = FastInferenceConfig(
        legacy_env=LegacyEnvConfig(enabled=True),
    )

    cfg = RuntimeConfig.from_vllm_config(vllm_config)

    assert cfg.tuning_env["FASTINFERENCE_GEMMA4_ALLOW_INT4_KV"] == "1"
    assert "FASTINFERENCE_KV_TYPE" not in cfg.tuning_env
    assert cfg.tuning_env["FASTINFERENCE_PROFILE"] == "auto"


def test_runtime_config_accuracy_profile_defaults_to_fp8(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "accuracy")
    monkeypatch.delenv("FASTINFERENCE_KV_TYPE", raising=False)
    vllm_config = _mock_vllm_config()
    vllm_config.fastinference_config = FastInferenceConfig(profile="accuracy")

    cfg = RuntimeConfig.from_vllm_config(vllm_config)

    assert cfg.profile.requested_name == "accuracy"
    assert cfg.kv_cache_dtype == "fp8"


def test_runtime_config_config_kv_type_overrides_accuracy_profile(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "accuracy")
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "fp8")
    vllm_config = _mock_vllm_config()
    vllm_config.fastinference_config = FastInferenceConfig(
        profile="accuracy",
        kv_type="turbo_int4",
    )

    cfg = RuntimeConfig.from_vllm_config(vllm_config)

    assert cfg.profile.requested_name == "accuracy"
    assert cfg.kv_cache_dtype == "turbo_int4"


def test_runtime_config_latency_profile_caps_kv_shape(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "latency")
    vllm_config = _mock_vllm_config()
    vllm_config.fastinference_config = FastInferenceConfig(profile="latency")

    cfg = RuntimeConfig.from_vllm_config(vllm_config)

    assert cfg.kv_max_active_requests == 1
    assert cfg.kv_max_model_len == 512


def test_runtime_config_default_max_prefill_chunk_preserves_planner(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FASTINFERENCE_PROFILE", raising=False)

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.max_prefill_chunk_size is None


def test_inference_config_from_env_defaults_to_turbo_int4(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_KV_TYPE", raising=False)
    monkeypatch.delenv("FASTINFERENCE_KV_FP8", raising=False)

    cfg = LiteInferenceConfig.from_env()

    assert cfg.kv_type == "turbo_int4"


def test_inference_config_from_env_auto_respects_legacy_fp8_toggle(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "auto")
    monkeypatch.setenv("FASTINFERENCE_KV_FP8", "1")

    cfg = LiteInferenceConfig.from_env()

    assert cfg.kv_type == "fp8"
