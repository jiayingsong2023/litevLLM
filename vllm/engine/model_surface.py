# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import Any


@dataclass(frozen=True)
class ModelSurface:
    event_name: str
    model_name: str
    model_type: str
    status: str
    reason: str


_SUPPORTED_MODEL_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tinyllama", ("tinyllama-1.1b-chat-v1.0",)),
    ("qwen3_5", ("qwen3.5-9b-awq",)),
    ("gemma4", ("gemma-4-31b-it-awq-4bit", "gemma-4-26b-a4b-it-awq-4bit")),
)


def resolve_model_surface(*, model_name: str, capabilities: Any) -> ModelSurface:
    normalized_name = _normalize_model_name(model_name)
    model_type = str(getattr(capabilities, "model_type", "") or "unknown")
    if _is_supported_regression_model(model_type, normalized_name):
        return ModelSurface(
            event_name="supported_model_surface",
            model_name=model_name,
            model_type=model_type,
            status="supported",
            reason="model_in_regression_surface",
        )
    return ModelSurface(
        event_name="experimental_model_surface",
        model_name=model_name,
        model_type=model_type,
        status="experimental",
        reason="model_not_in_regression_surface",
    )


def _is_supported_regression_model(model_type: str, normalized_name: str) -> bool:
    for supported_type, name_patterns in _SUPPORTED_MODEL_PATTERNS:
        if model_type != supported_type:
            continue
        if any(pattern in normalized_name for pattern in name_patterns):
            return True
    return False


def _normalize_model_name(model_name: str) -> str:
    raw = str(model_name or "").strip().lower()
    return PurePath(raw).name if raw else ""
