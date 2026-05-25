# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .base import ModelAdapter
from .gemma4 import Gemma4Adapter
from .llama import LlamaAdapter
from .qwen3_5 import Qwen35Adapter


def _hf_config_candidates(model_config: Any) -> tuple[Any, ...]:
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        hf_config = _load_local_hf_config(model_config)
    if hf_config is None:
        return ()
    text_config = getattr(hf_config, "text_config", None)
    if text_config is None:
        return (hf_config,)
    return (hf_config, text_config)


def _load_local_hf_config(model_config: Any) -> Any | None:
    model_path = getattr(model_config, "model", None)
    if not model_path:
        return None
    config_path = Path(str(model_path)) / "config.json"
    if not config_path.is_file():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    text_config = data.get("text_config")
    return SimpleNamespace(
        model_type=data.get("model_type"),
        architectures=data.get("architectures", []),
        text_config=SimpleNamespace(
            model_type=text_config.get("model_type"),
            architectures=text_config.get("architectures", []),
        )
        if isinstance(text_config, dict)
        else None,
    )


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
