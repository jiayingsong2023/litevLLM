# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

from vllm.model_executor.layers.quantization.tensor import awq_fused_scope
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


def test_awq_fused_scope_uses_kernel_policy_before_profile_hint() -> None:
    assert (
        awq_fused_scope(
            {"kernel_policy": {"awq_fused_scope": "attention_only"}},
            "gemma4_31b_q4",
        )
        == "attention_only"
    )
    assert (
        awq_fused_scope({"kernel_policy": {"awq_fused_scope": "off"}}, "qwen35_9b_awq")
        == "off"
    )


def test_awq_fused_scope_keeps_profile_defaults_without_env() -> None:
    assert awq_fused_scope(None, "gemma4_31b_q4") == "all"
    assert awq_fused_scope(None, "qwen35_9b_awq") == "all"
    assert awq_fused_scope(None, "") == "attention_only"
