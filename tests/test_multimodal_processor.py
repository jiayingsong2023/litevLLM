# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import torch
from PIL import Image

from vllm.engine.multimodal_processor import LiteMultiModalProcessor


class _Gemma4LikeModel:
    supports_multimodal = True

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(image_size=8, patch_size=4),
            image_token="<image>",
        )

    def get_multimodal_embeddings(self, **kwargs):
        del kwargs
        return torch.zeros((1, 4, 3))


def _data_url_image() -> str:
    image = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_multimodal_processor_expands_single_image_placeholder_before_tokenize() -> (
    None
):
    processor = LiteMultiModalProcessor(
        model=_Gemma4LikeModel(),
        device=torch.device("cpu"),
    )

    prompt, prepared = processor.prepare_before_tokenize(
        "describe <image> now",
        {"image": [{"image": _data_url_image()}]},
    )

    assert prompt == "describe <image> <image> <image> <image> now"
    assert prepared["image_token_count"] == 4
    assert len(prepared["image"]) == 1
    assert prepared["image"][0]["prepared_image"].size == (8, 8)


def test_multimodal_processor_prepare_request_uses_prepared_image() -> None:
    processor = LiteMultiModalProcessor(
        model=_Gemma4LikeModel(),
        device=torch.device("cpu"),
    )
    _, prepared = processor.prepare_before_tokenize(
        "describe <image>",
        {"image": [{"image": _data_url_image()}]},
    )
    prepared["image_token_id"] = 77
    from vllm.engine.request_state import RequestState
    from vllm.sampling_params import SamplingParams

    request = RequestState(
        request_id="req-mm",
        prompt="describe <image>",
        guarded_prompt="describe <image> <image> <image> <image>",
        input_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=1),
        multi_modal_data=prepared,
        is_multimodal=True,
    )

    processor.prepare_request(request)

    assert request.multi_modal_inputs["image_token_count"] == 4
    assert request.multi_modal_inputs["image_token_id"] == 77
    assert tuple(request.multi_modal_inputs["pixel_values"].shape) == (1, 3, 8, 8)
    assert processor.prepared_requests == 1
    assert processor.prepared_images == 1


def test_multimodal_processor_build_prefill_inputs_carries_image_metadata() -> None:
    processor = LiteMultiModalProcessor(
        model=_Gemma4LikeModel(),
        device=torch.device("cpu"),
    )
    from vllm.engine.request_state import RequestState
    from vllm.sampling_params import SamplingParams

    request = RequestState(
        request_id="req-mm",
        prompt="describe <image>",
        input_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=1),
        is_multimodal=True,
    )
    request.multi_modal_inputs = {
        "pixel_values": torch.ones((1, 3, 8, 8)),
        "image_token_count": 4,
        "image_token_id": 77,
    }

    inputs = processor.build_prefill_inputs([request])

    assert tuple(inputs["pixel_values"].shape) == (1, 3, 8, 8)
    assert inputs["image_token_count"] == 4
    assert inputs["image_token_id"] == 77


def test_multimodal_processor_keeps_per_placeholder_embeddings() -> None:
    processor = LiteMultiModalProcessor(
        model=_Gemma4LikeModel(),
        device=torch.device("cpu"),
    )

    embeddings = processor.get_multimodal_embeddings({
        "pixel_values": torch.ones((1, 3, 8, 8)),
        "image_token_count": 4,
        "image_token_id": 77,
    })

    assert tuple(embeddings.shape) == (1, 4, 3)
    assert processor.embeddings_computed == 1
    assert processor.embedding_feature_dim == 3


def test_multimodal_processor_prefers_gemma4_default_output_length() -> None:
    model = _Gemma4LikeModel()
    model.config.vision_config = SimpleNamespace(
        default_output_length=280,
        image_size=8,
        patch_size=4,
    )
    processor = LiteMultiModalProcessor(model=model, device=torch.device("cpu"))

    assert processor._image_token_count() == 280
