# SPDX-License-Identifier: Apache-2.0
"""Model-path hints used only to select an established AWQ policy profile."""

from __future__ import annotations

import os


def looks_like_qwen35_9b_awq_model_path(model_path: str) -> bool:
    base = os.path.basename(os.path.abspath(model_path)).lower()
    return "qwen3.5-9b-awq" in base or (
        "qwen3.5" in base and "9b" in base and "awq" in base
    )


def qwen35_awq_profile_hint_from_model_path(model_path: str) -> str:
    return "qwen35_9b_awq" if looks_like_qwen35_9b_awq_model_path(model_path) else ""


def looks_like_gemma4_31b_model_path(model_path: str) -> bool:
    base = os.path.basename(os.path.abspath(model_path)).lower()
    return (
        "gemma-4-31b" in base
        or ("gemma4" in base and "31b" in base)
        or ("gemma" in base and "31b" in base and "awq" in base)
    )


def awq_profile_hint_from_model_path(model_path: str) -> str:
    if looks_like_qwen35_9b_awq_model_path(model_path):
        return "qwen35_9b_awq"
    return "gemma4_31b_q4" if looks_like_gemma4_31b_model_path(model_path) else ""
