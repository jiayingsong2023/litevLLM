# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

from vllm.model_executor.layers.quantization.tensor import _env_awq_fused_scope
from vllm.model_executor.model_loader import (
    _awq_profile_hint_from_model_path,
    _looks_like_gemma4_31b_model_path,
)


def test_looks_like_gemma4_31b_model_path_accepts_common_variants(tmp_path: Path) -> None:
    good = [
        tmp_path / "models" / "gemma-4-31B-it-AWQ-4bit",
        tmp_path / "models" / "Gemma4-31B-Q4",
        tmp_path / "models" / "foo-gemma-31b-awq",
    ]
    bad = [
        tmp_path / "models" / "gemma-4-26B-A4B-it-AWQ-4bit",
        tmp_path / "models" / "Qwen3.5-9B-AWQ",
    ]

    for path in good:
        assert _looks_like_gemma4_31b_model_path(str(path)) is True
        assert _awq_profile_hint_from_model_path(str(path)) == "gemma4_31b_q4"

    for path in bad:
        assert _looks_like_gemma4_31b_model_path(str(path)) is False


def test_env_awq_fused_scope_has_gemma4_31b_profile_defaults(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FASTINFERENCE_AWQ_FUSED_SCOPE", raising=False)

    monkeypatch.setenv("FASTINFERENCE_AWQ_POLICY_MATRIX", "safe")
    assert _env_awq_fused_scope("gemma4_31b_q4") == "attention_only"

    monkeypatch.setenv("FASTINFERENCE_AWQ_POLICY_MATRIX", "balanced")
    assert _env_awq_fused_scope("gemma4_31b_q4") == "all"

    monkeypatch.setenv("FASTINFERENCE_AWQ_POLICY_MATRIX", "throughput")
    assert _env_awq_fused_scope("gemma4_31b_q4") == "all"

    monkeypatch.setenv("FASTINFERENCE_AWQ_POLICY_MATRIX", "strict")
    assert _env_awq_fused_scope("gemma4_31b_q4") == "off"
