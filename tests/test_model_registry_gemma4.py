# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch.nn as nn

from vllm.model_executor.models.registry import _ModelRegistry


def _cfg_with_model_types(model_type: str | None, text_model_type: str | None) -> Any:
    text_cfg = SimpleNamespace(model_type=text_model_type)
    hf_cfg = SimpleNamespace(model_type=model_type, text_config=text_cfg)
    return SimpleNamespace(hf_config=hf_cfg)


def test_infer_arch_from_gemma4_top_level_model_type() -> None:
    reg = _ModelRegistry()
    model_cfg = _cfg_with_model_types("gemma4", None)
    assert reg._infer_architectures_from_model_config(model_cfg) == [
        "Gemma4ForConditionalGeneration"
    ]


def test_infer_arch_from_gemma4_text_model_type() -> None:
    reg = _ModelRegistry()
    model_cfg = _cfg_with_model_types("paligemma", "gemma4_text")
    assert reg._infer_architectures_from_model_config(model_cfg) == [
        "Gemma4ForConditionalGeneration"
    ]


def test_resolve_model_cls_uses_inferred_gemma4_arch_when_architectures_empty() -> None:
    reg = _ModelRegistry()
    model_cfg = _cfg_with_model_types("gemma4", None)

    model_cls, arch = reg.resolve_model_cls([], model_cfg)
    assert issubclass(model_cls, nn.Module)
    assert model_cls.__name__ == "Gemma4ForConditionalGeneration"
    assert arch == "Gemma4ForConditionalGeneration"


def test_resolve_model_cls_raises_for_empty_architectures_without_inference() -> None:
    reg = _ModelRegistry()
    model_cfg = SimpleNamespace(hf_config=SimpleNamespace(model_type="", text_config=None))
    with pytest.raises(ValueError, match=r"Unsupported architectures: \[\]"):
        reg.resolve_model_cls([], model_cfg)
