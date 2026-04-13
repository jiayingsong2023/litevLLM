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
