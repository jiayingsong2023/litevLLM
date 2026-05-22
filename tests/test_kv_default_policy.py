# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

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


def test_runtime_config_uses_profile_defaults(monkeypatch) -> None:
    monkeypatch.delenv("FASTINFERENCE_PROFILE", raising=False)
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_ALLOW_INT4_KV", "1")
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "fp16")

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.profile.requested_name == "auto"
    assert cfg.profile.effective_name == "benchmark"
    assert cfg.kv_cache_dtype == "turbo_int4"
    assert cfg.block_size == 16
    assert cfg.fusion_level == 2
    assert cfg.backend_policy.gpu_greedy_sampling is True
    assert cfg.tuning_env["FASTINFERENCE_GEMMA4_ALLOW_INT4_KV"] == "1"
    assert cfg.tuning_env["FASTINFERENCE_KV_TYPE"] == "fp16"
    assert cfg.tuning_env["FASTINFERENCE_PROFILE"] == "auto"


def test_runtime_config_accuracy_profile_is_conservative(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "accuracy")
    monkeypatch.setenv("FASTINFERENCE_KV_TYPE", "turbo_int4")

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.profile.requested_name == "accuracy"
    assert cfg.kv_cache_dtype == "fp8"


def test_runtime_config_latency_profile_caps_kv_shape(monkeypatch) -> None:
    monkeypatch.setenv("FASTINFERENCE_PROFILE", "latency")

    cfg = RuntimeConfig.from_vllm_config(_mock_vllm_config())

    assert cfg.kv_max_active_requests == 1
    assert cfg.kv_max_model_len == 512
