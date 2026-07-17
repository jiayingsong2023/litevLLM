# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.verify_semantic_integrity import (
    _assign_gemma4_reference_weights,
    _gemma4_map_ref_key,
    _looks_like_gemma4_model_path,
)


class _FakeLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qweight = torch.nn.Parameter(
            torch.zeros((2, 2), dtype=torch.int32), requires_grad=False
        )
        self.scales = torch.nn.Parameter(
            torch.zeros((2, 2), dtype=torch.float32), requires_grad=False
        )
        self.weight_shape = (0, 0)


class _FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = _FakeLinear()


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_FakeLayer()])


class _FakeSafeOpen:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def __enter__(self) -> _FakeSafeOpen:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        return None

    def keys(self) -> list[str]:
        return list(self._tensors.keys())

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._tensors[key]


def test_gemma4_map_ref_key_covers_text_and_ignores_multimodal() -> None:
    assert (
        _gemma4_map_ref_key(
            "model.language_model.layers.0.self_attn.q_proj.weight_packed"
        )
        == "model.layers.0.self_attn.q_proj.qweight"
    )
    assert (
        _gemma4_map_ref_key(
            "model.language_model.layers.0.self_attn.q_proj.weight_scale"
        )
        == "model.layers.0.self_attn.q_proj.scales"
    )
    assert (
        _gemma4_map_ref_key(
            "model.language_model.layers.0.self_attn.q_proj.weight_shape"
        )
        == "model.layers.0.self_attn.q_proj.weight_shape"
    )
    assert _gemma4_map_ref_key("model.embed_vision.patch_embedding.weight") is None
    assert _gemma4_map_ref_key("model.audio_tower.encoder.weight") is None


def test_looks_like_gemma4_model_path_accepts_top_level_or_text_config(
    tmp_path: Path,
) -> None:
    top_level = tmp_path / "top_level"
    nested = tmp_path / "nested"
    other = tmp_path / "other"
    top_level.mkdir()
    nested.mkdir()
    other.mkdir()

    (top_level / "config.json").write_text(
        json.dumps({"model_type": "gemma4"}), encoding="utf-8"
    )
    (nested / "config.json").write_text(
        json.dumps(
            {"model_type": "paligemma", "text_config": {"model_type": "gemma4_text"}}
        ),
        encoding="utf-8",
    )
    (other / "config.json").write_text(
        json.dumps({"model_type": "llama"}), encoding="utf-8"
    )

    assert _looks_like_gemma4_model_path(str(top_level)) is True
    assert _looks_like_gemma4_model_path(str(nested)) is True
    assert _looks_like_gemma4_model_path(str(other)) is False


def test_assign_gemma4_reference_weights_sets_qweight_scales_and_weight_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard_dir = tmp_path / "gemma4-ref"
    shard_dir.mkdir()
    (shard_dir / "model-00001-of-00001.safetensors").touch()

    tensors = {
        "model.language_model.layers.0.self_attn.q_proj.weight_packed": torch.tensor(
            [[1, 2], [3, 4]], dtype=torch.int32
        ),
        "model.language_model.layers.0.self_attn.q_proj.weight_scale": torch.tensor(
            [[0.5, 0.25], [0.125, 0.75]], dtype=torch.float32
        ),
        "model.language_model.layers.0.self_attn.q_proj.weight_shape": torch.tensor(
            [4096, 2048], dtype=torch.int32
        ),
        "model.embed_vision.patch_embedding.weight": torch.ones(
            (1,), dtype=torch.float32
        ),
    }

    def _fake_safe_open(path: str, framework: str, device: str) -> _FakeSafeOpen:
        assert path.endswith(".safetensors")
        assert framework == "pt"
        assert device == "cpu"
        return _FakeSafeOpen(tensors)

    model = _FakeModel()
    assigned = _assign_gemma4_reference_weights(
        model,
        str(shard_dir),
        torch.device("cpu"),
        safe_open_fn=_fake_safe_open,
    )

    q_proj = model.model.layers[0].self_attn.q_proj
    assert assigned == 3
    assert torch.equal(
        q_proj.qweight.detach(),
        tensors["model.language_model.layers.0.self_attn.q_proj.weight_packed"],
    )
    assert torch.equal(
        q_proj.scales.detach(),
        tensors["model.language_model.layers.0.self_attn.q_proj.weight_scale"],
    )
    assert q_proj.weight_shape == (4096, 2048)


def test_assign_gemma4_reference_weights_raises_when_no_supported_weights(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "gemma4-empty"
    shard_dir.mkdir()
    (shard_dir / "model-00001-of-00001.safetensors").touch()

    tensors = {
        "model.embed_vision.patch_embedding.weight": torch.ones(
            (1,), dtype=torch.float32
        ),
        "model.audio_tower.encoder.weight": torch.ones((1,), dtype=torch.float32),
    }

    def _fake_safe_open(path: str, framework: str, device: str) -> _FakeSafeOpen:
        assert path.endswith(".safetensors")
        assert framework == "pt"
        assert device == "cpu"
        return _FakeSafeOpen(tensors)

    with pytest.raises(RuntimeError, match="assigned no weights"):
        _assign_gemma4_reference_weights(
            _FakeModel(),
            str(shard_dir),
            torch.device("cpu"),
            safe_open_fn=_fake_safe_open,
        )
