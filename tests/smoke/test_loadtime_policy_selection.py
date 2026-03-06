# SPDX-License-Identifier: Apache-2.0

from vllm.engine.loadtime_policy import (
    AGGRESSIVE_PROFILE,
    STABLE_PROFILE,
    estimate_model_size_billion,
    select_loadtime_policy,
)


class _FakeHFConfig:
    def __init__(
        self,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        intermediate_size: int = 11008,
        vocab_size: int = 151936,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size


class _FakeModelConfig:
    def __init__(self, model: str, hf_config=None):
        self.model = model
        self.hf_config = hf_config


def test_estimate_model_size_from_model_name_fallback() -> None:
    model_config = _FakeModelConfig("models/Qwen3.5-32B")
    estimated = estimate_model_size_billion(model_config)
    assert estimated == 32.0


def test_auto_mode_selects_aggressive_for_small_model() -> None:
    model_config = _FakeModelConfig("models/small", _FakeHFConfig(hidden_size=2048, num_hidden_layers=24))
    selected = select_loadtime_policy(
        model_config=model_config,
        quant_config=None,
        policy_mode="auto",
        total_gpu_memory_gb=60.0,
    )
    assert selected == AGGRESSIVE_PROFILE


def test_auto_mode_selects_stable_for_large_model() -> None:
    model_config = _FakeModelConfig("models/Qwen3.5-35B")
    selected = select_loadtime_policy(
        model_config=model_config,
        quant_config=None,
        policy_mode="auto",
        total_gpu_memory_gb=60.0,
    )
    assert selected == STABLE_PROFILE


def test_manual_mode_overrides_auto_decision() -> None:
    model_config = _FakeModelConfig("models/Qwen3.5-35B")
    selected = select_loadtime_policy(
        model_config=model_config,
        quant_config=None,
        policy_mode="aggressive",
        total_gpu_memory_gb=4.0,
    )
    assert selected == AGGRESSIVE_PROFILE
