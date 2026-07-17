# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.adapters.base import ModelCapabilities
from vllm.engine.model_surface import resolve_model_surface


def _caps(model_type: str) -> ModelCapabilities:
    return ModelCapabilities(
        model_type=model_type,
        num_layers=1,
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=64,
        max_model_len=128,
        supports_moe=False,
        supports_fp8_kv=False,
        supports_int4_kv=True,
        supports_paged_prefill=True,
        preferred_kv_dtype="float16",
    )


def test_resolve_model_surface_marks_known_regression_models_supported() -> None:
    surface = resolve_model_surface(
        model_name="models/gemma-4-31B-it-AWQ-4bit",
        capabilities=_caps("gemma4"),
    )

    assert surface.status == "supported"
    assert surface.event_name == "supported_model_surface"
    assert surface.model_type == "gemma4"


def test_resolve_model_surface_marks_unqualified_registry_models_experimental() -> None:
    surface = resolve_model_surface(
        model_name="models/custom-llama-8b",
        capabilities=_caps("llama"),
    )

    assert surface.status == "experimental"
    assert surface.event_name == "experimental_model_surface"
    assert surface.reason == "model_not_in_regression_surface"


def test_resolve_model_surface_marks_tinyllama_supported_despite_llama_adapter() -> (
    None
):
    surface = resolve_model_surface(
        model_name="models/TinyLlama-1.1B-Chat-v1.0",
        capabilities=_caps("llama"),
    )

    assert surface.status == "supported"
    assert surface.event_name == "supported_model_surface"
    assert surface.reason == "model_in_regression_surface"
