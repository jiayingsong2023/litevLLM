# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm.lora.manager import LoRAManager
from vllm.lora.weights import LoRALayerWeights
from vllm.model_executor.models.gemma4.model import (
    Gemma4ForConditionalGeneration,
    Gemma4TextModel,
    _replace_image_placeholders,
)
from vllm.model_executor.models.gemma4.vision import (
    Gemma4VisionProjector,
    Gemma4VisionTower,
)


def test_replace_image_placeholders_inserts_image_embeddings() -> None:
    input_ids = torch.tensor([[10, 77, 77, 11]], dtype=torch.long)
    text_embeddings = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]]
    )
    image_embeddings = torch.tensor([[[8.0, 1.0], [9.0, 2.0]]])

    replaced = _replace_image_placeholders(
        input_ids=input_ids,
        text_embeddings=text_embeddings,
        multimodal_embeddings=image_embeddings,
        image_token_id=77,
        image_token_count=2,
    )

    assert replaced.tolist() == [
        [[1.0, 1.0], [8.0, 1.0], [9.0, 2.0], [4.0, 4.0]]
    ]


def test_replace_image_placeholders_inserts_batched_image_embeddings() -> None:
    input_ids = torch.tensor(
        [
            [10, 77, 77, 11],
            [12, 77, 77, 13],
        ],
        dtype=torch.long,
    )
    text_embeddings = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0]],
        ]
    )
    image_embeddings = torch.tensor(
        [
            [[10.0, 1.0], [11.0, 2.0]],
            [[12.0, 3.0], [13.0, 4.0]],
        ]
    )

    replaced = _replace_image_placeholders(
        input_ids=input_ids,
        text_embeddings=text_embeddings,
        multimodal_embeddings=image_embeddings,
        image_token_id=77,
        image_token_count=4,
    )

    assert replaced.tolist() == [
        [[1.0, 1.0], [10.0, 1.0], [11.0, 2.0], [4.0, 4.0]],
        [[5.0, 5.0], [12.0, 3.0], [13.0, 4.0], [8.0, 8.0]],
    ]


def test_replace_image_placeholders_handles_mixed_text_and_image_batch() -> None:
    input_ids = torch.tensor(
        [
            [10, 77, 77, 11],
            [12, 13, 14, 15],
            [16, 77, 77, 17],
        ],
        dtype=torch.long,
    )
    text_embeddings = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0]],
            [[9.0, 9.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0]],
        ]
    )
    image_embeddings = torch.tensor(
        [
            [20.0, 1.0],
            [21.0, 2.0],
            [22.0, 3.0],
            [23.0, 4.0],
        ]
    )

    replaced = _replace_image_placeholders(
        input_ids=input_ids,
        text_embeddings=text_embeddings,
        multimodal_embeddings=image_embeddings,
        image_token_id=77,
        image_token_count=4,
    )

    assert replaced.tolist() == [
        [[1.0, 1.0], [20.0, 1.0], [21.0, 2.0], [4.0, 4.0]],
        [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0]],
        [[9.0, 9.0], [22.0, 3.0], [23.0, 4.0], [12.0, 12.0]],
    ]


def test_replace_image_placeholders_rejects_count_mismatch() -> None:
    input_ids = torch.tensor([[10, 77, 11]], dtype=torch.long)
    text_embeddings = torch.zeros((1, 3, 2))
    image_embeddings = torch.zeros((1, 2, 2))

    try:
        _replace_image_placeholders(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            multimodal_embeddings=image_embeddings,
            image_token_id=77,
            image_token_count=2,
        )
    except ValueError as exc:
        assert "image placeholder count" in str(exc)
    else:
        raise AssertionError("expected image placeholder count mismatch")


def test_gemma4_text_model_forward_replaces_multimodal_embeddings(monkeypatch) -> None:
    model = object.__new__(Gemma4TextModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=2, tie_word_embeddings=False)
    model.embed_scale = 1.0
    model.embed_tokens = torch.nn.Embedding(100, 2)
    with torch.no_grad():
        model.embed_tokens.weight.zero_()
        model.embed_tokens.weight[10] = torch.tensor([1.0, 1.0])
        model.embed_tokens.weight[11] = torch.tensor([4.0, 4.0])
    model.layers = []
    model.norm = torch.nn.Identity()

    output = Gemma4TextModel.forward(
        model,
        torch.tensor([[10, 77, 77, 11]], dtype=torch.long),
        torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        [],
        {"image_token_id": 77, "image_token_count": 2},
        multimodal_embeddings=torch.tensor([[[8.0, 1.0], [9.0, 2.0]]]),
    )

    assert output.tolist() == [
        [[1.0, 1.0], [8.0, 1.0], [9.0, 2.0], [4.0, 4.0]]
    ]


def test_gemma4_forward_accepts_multimodal_embeddings() -> None:
    model = object.__new__(Gemma4ForConditionalGeneration)
    torch.nn.Module.__init__(model)

    class _Inner(torch.nn.Module):
        config = SimpleNamespace(tie_word_embeddings=False)

        def __init__(self) -> None:
            super().__init__()
            self.received = None

        def forward(
            self,
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            lora_mapping=None,
            multimodal_embeddings=None,
        ):
            del positions, kv_caches, lora_mapping
            self.received = (input_ids, attn_metadata, multimodal_embeddings)
            return torch.ones((1, 4, 2))

    class _Head(torch.nn.Module):
        def forward(self, hidden, lora_mapping=None):
            del lora_mapping
            return torch.nn.functional.pad(hidden, (0, 3))

    model.model = _Inner()
    model.lm_head = _Head()

    logits = Gemma4ForConditionalGeneration.forward(
        model,
        torch.tensor([[10, 77, 77, 11]], dtype=torch.long),
        torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        [],
        {"image_token_id": 77, "image_token_count": 2},
        multimodal_embeddings=torch.ones((1, 2, 2)),
    )

    assert tuple(logits.shape) == (1, 1, 5)
    assert model.model.received[2].shape == (1, 2, 2)


def test_gemma4_multimodal_interface_is_advertised() -> None:
    model = object.__new__(Gemma4ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    assert getattr(model, "supports_multimodal", False) is True
    assert callable(getattr(model, "get_multimodal_embeddings", None))


def test_gemma4_vision_tower_outputs_patch_embeddings() -> None:
    config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
        num_hidden_layers=1,
        patch_size=2,
        position_embedding_size=16,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
    )
    tower = Gemma4VisionTower(config)

    embeddings = tower(torch.ones((1, 3, 4, 4)))

    assert tuple(embeddings.shape) == (4, 4)


def test_gemma4_vision_tower_pools_patch_embeddings() -> None:
    config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=2,
        head_dim=2,
        num_hidden_layers=0,
        patch_size=2,
        pooling_kernel_size=2,
        position_embedding_size=16,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
    )
    tower = Gemma4VisionTower(config)

    embeddings = tower(torch.ones((1, 3, 8, 8)))

    assert tuple(embeddings.shape) == (4, 4)


def test_gemma4_get_multimodal_embeddings_projects_and_truncates() -> None:
    model = object.__new__(Gemma4ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    inner = torch.nn.Module()
    inner.vision_tower = torch.nn.Identity()
    inner.embed_vision = Gemma4VisionProjector(4, 6)
    model.model = inner

    embeddings = Gemma4ForConditionalGeneration.get_multimodal_embeddings(
        model,
        pixel_values=torch.ones((1, 5, 4)),
        image_token_count=3,
    )

    assert tuple(embeddings.shape) == (1, 3, 6)


def test_gemma4_get_multimodal_embeddings_applies_projector_lora() -> None:
    model = object.__new__(Gemma4ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    inner = torch.nn.Module()
    inner.vision_tower = torch.nn.Identity()
    inner.embed_vision = Gemma4VisionProjector(2, 2)
    inner.embed_vision.embedding_projection.weight = torch.nn.Parameter(
        torch.eye(2),
        requires_grad=False,
    )
    manager = LoRAManager(inner)
    manager.add_adapter_weights(
        "adapter-a",
        {
            "embed_vision.embedding_projection": LoRALayerWeights(
                lora_name="adapter-a",
                rank=1,
                alpha=1,
                lora_a=torch.tensor([[1.0], [0.0]]),
                lora_b=torch.tensor([[10.0, 20.0]]),
            )
        },
    )
    manager.bind_to_model()
    model.model = inner

    embeddings = Gemma4ForConditionalGeneration.get_multimodal_embeddings(
        model,
        pixel_values=torch.tensor([[[2.0, 3.0]]]),
        image_token_count=1,
        lora_mapping=["adapter-a"],
    )

    assert torch.allclose(embeddings, torch.tensor([[[22.0, 43.0]]]))


def test_gemma4_get_multimodal_embeddings_flattens_multiple_images() -> None:
    model = object.__new__(Gemma4ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    inner = torch.nn.Module()
    inner.vision_tower = torch.nn.Identity()
    inner.embed_vision = Gemma4VisionProjector(4, 6)
    model.model = inner

    embeddings = Gemma4ForConditionalGeneration.get_multimodal_embeddings(
        model,
        pixel_values=torch.ones((2, 5, 4)),
        image_token_count=6,
    )

    assert tuple(embeddings.shape) == (6, 6)
