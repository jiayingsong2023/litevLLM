# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from transformers import Qwen2VLConfig

from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLForCausalLM,
    build_qwen2_vl_vision_tower,
    qwen2_vl_visual_state_dict_from_checkpoint,
)


def test_qwen2_vl_merges_embeddings_by_image_placeholders() -> None:
    model = object.__new__(Qwen2VLForCausalLM)
    torch.nn.Module.__init__(model)
    inner = torch.nn.Module()
    inner.embed_tokens = torch.nn.Embedding(100, 2)
    inner.config = SimpleNamespace(hidden_size=2)
    with torch.no_grad():
        inner.embed_tokens.weight.zero_()
        inner.embed_tokens.weight[10] = torch.tensor([1.0, 1.0])
        inner.embed_tokens.weight[11] = torch.tensor([4.0, 4.0])
    model.model = inner

    merged = Qwen2VLForCausalLM._merge_multimodal_embeddings(
        model,
        input_ids=torch.tensor([[10, 77, 77, 11]], dtype=torch.long),
        multimodal_embeddings=torch.tensor([[[8.0, 1.0], [9.0, 2.0]]]),
        image_token_id=77,
        image_token_count=2,
    )

    assert merged.tolist() == [
        [[1.0, 1.0], [8.0, 1.0], [9.0, 2.0], [4.0, 4.0]]
    ]


def test_qwen2_vl_builds_vision_tower_from_fallback_vision_config() -> None:
    hf_config = SimpleNamespace(
        vision_config=SimpleNamespace(
            depth=1,
            embed_dim=1280,
            hidden_size=1536,
            in_chans=3,
            mlp_ratio=4,
            num_heads=16,
            patch_size=14,
            spatial_merge_size=2,
            spatial_patch_size=14,
            temporal_patch_size=2,
        )
    )

    with torch.device("meta"):
        tower = build_qwen2_vl_vision_tower(hf_config)

    assert "patch_embed.proj.weight" in tower.state_dict()


def test_qwen2_vl_computes_mrope_positions_for_image_tokens() -> None:
    model = object.__new__(Qwen2VLForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        image_token_id=77,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )

    positions = Qwen2VLForCausalLM._qwen2_vl_positions(
        model,
        input_ids=torch.tensor([[10, 77, 77, 77, 77, 11]], dtype=torch.long),
        positions=torch.arange(6).view(1, 6),
        image_grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.long),
        image_token_id=77,
    )

    assert tuple(positions.shape) == (3, 1, 6)
    assert positions[:, 0, 1:5].tolist() == [
        [1, 1, 1, 1],
        [1, 1, 2, 2],
        [1, 2, 1, 2],
    ]
    assert positions[:, 0, 5].tolist() == [3, 3, 3]


def test_qwen2_vl_installs_mrope_on_decoder_layers() -> None:
    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = SimpleNamespace(head_dim=128)

    model = object.__new__(Qwen2VLForCausalLM)
    torch.nn.Module.__init__(model)
    inner = torch.nn.Module()
    inner.config = SimpleNamespace(
        max_position_embeddings=32768,
        rope_theta=1000000.0,
    )
    inner.layers = torch.nn.ModuleList([Layer()])
    model.model = inner

    Qwen2VLForCausalLM._install_mrope(
        model,
        SimpleNamespace(rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]}),
    )

    assert type(model.model.layers[0].rotary_emb).__name__ == "MRotaryEmbedding"


def test_qwen2_vl_visual_checkpoint_keys_match_vision_tower() -> None:
    root = Path("models/Qwen2-VL-2B-Instruct")
    if not (root / "model.safetensors.index.json").is_file():
        pytest.skip("Qwen2VL checkpoint is not available locally")

    hf_config = Qwen2VLConfig.from_dict(
        json.loads((root / "config.json").read_text(encoding="utf-8"))
    )
    with torch.device("meta"):
        tower = build_qwen2_vl_vision_tower(hf_config)

    expected = set(tower.state_dict())
    visual_state = qwen2_vl_visual_state_dict_from_checkpoint(root)
    missing, unexpected = tower.load_state_dict(visual_state, assign=True)

    assert set(visual_state) == expected
    assert missing == []
    assert unexpected == []


def test_qwen2_vl_visual_checkpoint_shard_opens() -> None:
    root = Path("models/Qwen2-VL-2B-Instruct")
    shard = root / "model-00001-of-00002.safetensors"
    if not shard.is_file():
        pytest.skip("Qwen2VL checkpoint is not available locally")

    visual_state = qwen2_vl_visual_state_dict_from_checkpoint(root)
    assert len(visual_state) == 391
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        assert handle.get_tensor("visual.patch_embed.proj.weight").shape == (
            1280,
            3,
            2,
            14,
            14,
        )
