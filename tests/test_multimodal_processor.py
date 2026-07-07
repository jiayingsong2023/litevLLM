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
            model_type="gemma4",
            vision_config=SimpleNamespace(image_size=8, patch_size=4),
        )
        self.last_multimodal_kwargs = None

    def get_multimodal_embeddings(self, **kwargs):
        self.last_multimodal_kwargs = dict(kwargs)
        return torch.zeros((1, 4, 3))


class _Qwen2VLLikeModel:
    supports_multimodal = True

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_type="qwen2_vl",
            vision_config=SimpleNamespace(
                patch_size=14,
                spatial_merge_size=2,
                temporal_patch_size=2,
            ),
            image_token="<|image_pad|>",
        )

    def get_multimodal_embeddings(self, **kwargs):
        del kwargs
        return torch.zeros((4, 3))


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

    assert prompt == "describe <|image><|image|><|image|><|image|><|image|><image|> now"
    assert prepared["image_token_count"] == 4
    assert len(prepared["image"]) == 1
    assert tuple(prepared["image"][0]["prepared_pixel_values"].shape) == (1, 4, 48)
    assert tuple(prepared["image"][0]["image_position_ids"].shape) == (1, 4, 2)


def test_multimodal_processor_accepts_gemma_chat_template_image_marker() -> None:
    processor = LiteMultiModalProcessor(
        model=_Gemma4LikeModel(),
        device=torch.device("cpu"),
    )

    prompt, prepared = processor.prepare_before_tokenize(
        "<|image|>\ndescribe",
        {"image": [{"image": _data_url_image()}]},
    )

    assert prompt == "<|image><|image|><|image|><|image|><|image|><image|>\ndescribe"
    assert prepared["image_token_count"] == 4


def test_multimodal_processor_carries_config_image_token_id_before_tokenize() -> None:
    model = _Gemma4LikeModel()
    model.config.image_token_id = 258880
    processor = LiteMultiModalProcessor(
        model=model,
        device=torch.device("cpu"),
    )

    _, prepared = processor.prepare_before_tokenize(
        "describe <image>",
        {"image": [{"image": _data_url_image()}]},
    )

    assert prepared["image_token_id"] == 258880


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
    assert tuple(request.multi_modal_inputs["pixel_values"].shape) == (1, 4, 48)
    assert tuple(request.multi_modal_inputs["image_position_ids"].shape) == (1, 4, 2)
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


def test_multimodal_processor_build_prefill_inputs_batches_multiple_requests() -> None:
    processor = LiteMultiModalProcessor(
        model=_Gemma4LikeModel(),
        device=torch.device("cpu"),
    )
    from vllm.engine.request_state import RequestState
    from vllm.sampling_params import SamplingParams

    req1 = RequestState(
        request_id="req-mm-1",
        prompt="describe <image>",
        input_ids=[1, 77, 77, 77, 77],
        sampling_params=SamplingParams(max_tokens=1),
        is_multimodal=True,
    )
    req1.multi_modal_inputs = {
        "pixel_values": torch.ones((1, 3, 8, 8)),
        "image_token_count": 4,
        "image_token_id": 77,
    }
    req2 = RequestState(
        request_id="req-mm-2",
        prompt="describe <image>",
        input_ids=[2, 77, 77, 77, 77],
        sampling_params=SamplingParams(max_tokens=1),
        is_multimodal=True,
    )
    req2.multi_modal_inputs = {
        "pixel_values": torch.full((1, 3, 8, 8), 2.0),
        "image_token_count": 4,
        "image_token_id": 77,
    }

    inputs = processor.build_prefill_inputs([req1, req2])

    assert tuple(inputs["pixel_values"].shape) == (2, 3, 8, 8)
    assert inputs["image_token_count"] == 8
    assert inputs["image_token_counts"] == [4, 4]
    assert inputs["image_token_id"] == 77


def test_multimodal_processor_expands_multiple_images_before_tokenize() -> None:
    processor = LiteMultiModalProcessor(
        model=_Gemma4LikeModel(),
        device=torch.device("cpu"),
    )

    prompt, prepared = processor.prepare_before_tokenize(
        "compare <image> and <image>",
        {"image": [{"image": _data_url_image()}, {"image": _data_url_image()}]},
    )

    image_span = "<|image><|image|><|image|><|image|><|image|><image|>"
    assert prompt == f"compare {image_span} and {image_span}"
    assert prepared["image_token_count"] == 8
    assert len(prepared["image"]) == 2


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


def test_multimodal_processor_forwards_lora_mapping_to_model() -> None:
    model = _Gemma4LikeModel()
    processor = LiteMultiModalProcessor(
        model=model,
        device=torch.device("cpu"),
    )

    processor.get_multimodal_embeddings({
        "pixel_values": torch.ones((1, 3, 8, 8)),
        "image_token_count": 4,
        "image_token_id": 77,
        "lora_mapping": ["adapter-a"],
    })

    assert model.last_multimodal_kwargs["lora_mapping"] == ["adapter-a"]


def test_multimodal_processor_prefers_gemma4_default_output_length() -> None:
    model = _Gemma4LikeModel()
    model.config.vision_config = SimpleNamespace(
        default_output_length=280,
        image_size=8,
        patch_size=4,
    )
    processor = LiteMultiModalProcessor(model=model, device=torch.device("cpu"))

    assert processor._image_token_count() == 280


def test_multimodal_processor_scales_pixel_grid_for_gemma4_pooling() -> None:
    model = _Gemma4LikeModel()
    model.config.vision_config = SimpleNamespace(
        default_output_length=280,
        image_size=8,
        patch_size=16,
        pooling_kernel_size=3,
    )
    processor = LiteMultiModalProcessor(model=model, device=torch.device("cpu"))

    assert processor._image_patch_grid() == (42, 60)


def test_multimodal_processor_prefers_gemma4_unified_num_soft_tokens() -> None:
    model = _Gemma4LikeModel()
    model.config.vision_config = SimpleNamespace(
        num_soft_tokens=280,
        patch_size=16,
    )
    processor = LiteMultiModalProcessor(model=model, device=torch.device("cpu"))

    assert processor._image_token_count() == 280


def test_qwen2_vl_processor_expands_prompt_from_image_grid() -> None:
    processor = LiteMultiModalProcessor(
        model=_Qwen2VLLikeModel(),
        device=torch.device("cpu"),
    )

    prompt, prepared = processor.prepare_before_tokenize(
        "describe <image>",
        {"image": [{"image": _data_url_image()}]},
    )

    assert prompt == (
        "describe <|image_pad|> <|image_pad|> <|image_pad|> <|image_pad|>"
    )
    assert prepared["image_token_count"] == 4
    assert prepared["image_token_counts"] == [4]
    assert tuple(prepared["image"][0]["prepared_pixel_values"].shape) == (16, 1176)
    assert prepared["image"][0]["image_grid_thw"].tolist() == [[1, 4, 4]]


def test_qwen2_vl_processor_uses_image_pad_when_config_has_only_token_id() -> None:
    model = _Qwen2VLLikeModel()
    delattr(model.config, "image_token")
    processor = LiteMultiModalProcessor(model=model, device=torch.device("cpu"))

    prompt, prepared = processor.prepare_before_tokenize(
        "describe <image>",
        {"image": [{"image": _data_url_image()}]},
    )

    assert "<|image_pad|>" in prompt
    assert prepared["image_token"] == "<|image_pad|>"


def test_qwen2_vl_processor_prepare_request_carries_grid_thw() -> None:
    processor = LiteMultiModalProcessor(
        model=_Qwen2VLLikeModel(),
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
        request_id="req-qwen-mm",
        prompt="describe <image>",
        guarded_prompt="describe <image> <image> <image> <image>",
        input_ids=[1, 77, 77, 77, 77],
        sampling_params=SamplingParams(max_tokens=1),
        multi_modal_data=prepared,
        is_multimodal=True,
    )

    processor.prepare_request(request)

    assert tuple(request.multi_modal_inputs["pixel_values"].shape) == (16, 1176)
    assert request.multi_modal_inputs["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert request.multi_modal_inputs["image_token_count"] == 4
    assert request.multi_modal_inputs["image_token_id"] == 77


def test_qwen2_vl_build_prefill_inputs_batches_grid_thw() -> None:
    processor = LiteMultiModalProcessor(
        model=_Qwen2VLLikeModel(),
        device=torch.device("cpu"),
    )
    from vllm.engine.request_state import RequestState
    from vllm.sampling_params import SamplingParams

    req1 = RequestState(
        request_id="req-qwen-mm-1",
        prompt="describe <image>",
        input_ids=[1, 77, 77, 77, 77],
        sampling_params=SamplingParams(max_tokens=1),
        is_multimodal=True,
    )
    req1.multi_modal_inputs = {
        "pixel_values": torch.ones((16, 1176)),
        "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
        "image_token_count": 4,
        "image_token_id": 77,
    }
    req2 = RequestState(
        request_id="req-qwen-mm-2",
        prompt="describe <image>",
        input_ids=[2, 77, 77, 77, 77],
        sampling_params=SamplingParams(max_tokens=1),
        is_multimodal=True,
    )
    req2.multi_modal_inputs = {
        "pixel_values": torch.full((16, 1176), 2.0),
        "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
        "image_token_count": 4,
        "image_token_id": 77,
    }

    inputs = processor.build_prefill_inputs([req1, req2])

    assert tuple(inputs["pixel_values"].shape) == (32, 1176)
    assert inputs["image_grid_thw"].tolist() == [[1, 4, 4], [1, 4, 4]]
    assert inputs["image_token_count"] == 8
    assert inputs["image_token_counts"] == [4, 4]
