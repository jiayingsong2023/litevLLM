# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.model_executor.model_loader import (
    _find_non_layer_module_attr_key,
    _is_dense_safetensors_weight,
)


def test_non_layer_module_key_lookup_prefers_exact_multimodal_path() -> None:
    keys = [
        "model.embed_audio.embedding_projection.weight",
        "model.embed_vision.embedding_projection.weight",
    ]

    key = _find_non_layer_module_attr_key(
        keys,
        m_name="model.embed_vision.embedding_projection",
        proj="embedding_projection",
        alt_suffixes=["weight"],
    )

    assert key == "model.embed_vision.embedding_projection.weight"


def test_non_layer_module_key_lookup_keeps_loose_suffix_fallback() -> None:
    keys = ["language_model.lm_head.weight"]

    key = _find_non_layer_module_attr_key(
        keys,
        m_name="lm_head",
        proj="lm_head",
        alt_suffixes=["weight"],
    )

    assert key == "language_model.lm_head.weight"


def test_non_layer_module_key_lookup_prefers_language_model_alias() -> None:
    keys = [
        "model.audio_tower.subsample_conv_projection.layer0.norm.weight",
        "model.language_model.norm.weight",
    ]

    key = _find_non_layer_module_attr_key(
        keys,
        m_name="model.norm",
        proj="norm",
        alt_suffixes=["weight"],
    )

    assert key == "model.language_model.norm.weight"


def test_dense_safetensors_weight_accepts_fp32_projector_weight() -> None:
    assert _is_dense_safetensors_weight(torch.empty((3840, 3840), dtype=torch.float32))
