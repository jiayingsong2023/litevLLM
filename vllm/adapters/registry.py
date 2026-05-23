# SPDX-License-Identifier: Apache-2.0
from typing import Any

from .base import ModelAdapter
from .gemma4 import Gemma4Adapter
from .llama import LlamaAdapter
from .qwen3_5 import Qwen35Adapter


def _hf_config_candidates(model_config: Any) -> tuple[Any, ...]:
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return ()
    text_config = getattr(hf_config, "text_config", None)
    if text_config is None:
        return (hf_config,)
    return (hf_config, text_config)


def _looks_like_qwen35(model: Any, model_config: Any) -> bool:
    name = type(model).__name__.lower()
    if "qwen3_5" in name or "qwen35" in name:
        return True
    for config in _hf_config_candidates(model_config):
        model_type = str(getattr(config, "model_type", "") or "").lower()
        if model_type in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe_text"):
            return True
    return False


def _looks_like_gemma4(model: Any, model_config: Any) -> bool:
    name = type(model).__name__.lower()
    if "gemma4" in name:
        return True
    for config in _hf_config_candidates(model_config):
        model_type = str(getattr(config, "model_type", "") or "").lower()
        if model_type in ("gemma4", "gemma4_text"):
            return True
        archs = getattr(config, "architectures", [])
        if any("gemma4" in str(a).lower() for a in (archs or [])):
            return True
    return False


def get_model_adapter(model: Any, model_config: Any) -> ModelAdapter:
    if _looks_like_gemma4(model, model_config):
        return Gemma4Adapter()
    if _looks_like_qwen35(model, model_config):
        return Qwen35Adapter()
    return LlamaAdapter()
