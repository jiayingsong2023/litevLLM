# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from vllm.engine.input_batch_builder import InputBatchBuilder
from vllm.engine.kv_block_manager import KVBlockManager
from vllm.engine.multimodal_processor import LiteMultiModalProcessor
from vllm.engine.prefill_executor import PrefillExecutor
from vllm.engine.request_scheduler import RequestScheduler
from vllm.model_executor.models.qwen2_vl import Qwen2VLForCausalLM


def _stack(req_dicts, num_layers: int, key: str):
    out = []
    for li in range(num_layers):
        parts = [r[key][li] for r in req_dicts]
        out.append(None if all(p is None for p in parts) else torch.cat(parts, dim=0))
    return out


def _split(stacked, req_dicts, key: str):
    for li, t in enumerate(stacked):
        for i, r in enumerate(req_dicts):
            r[key][li] = None if t is None else t[i : i + 1].contiguous()


def _image_url(tmp_path: Path) -> str:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_path)
    return f"file://{image_path}"


def _image_urls(tmp_path: Path, count: int) -> list[str]:
    urls: list[str] = []
    for index in range(count):
        image_path = tmp_path / f"sample_{index}.png"
        Image.new("RGB", (4, 4), color=(255, index, 0)).save(image_path)
        urls.append(f"file://{image_path}")
    return urls


class _FakeMultiModalModel:
    supports_multimodal = True

    def __init__(self) -> None:
        self.last_pixel_values = None
        self.last_attn_metadata = None
        self.last_multimodal_embeddings = None
        self.last_lora_mapping = None

    def get_multimodal_embeddings(self, **kwargs) -> torch.Tensor:
        self.last_pixel_values = kwargs["pixel_values"]
        return kwargs["pixel_values"][:, :8]

    def __call__(
        self,
        input_ids,
        positions,
        kv_caches,
        attn_metadata,
        multimodal_embeddings=None,
        lora_mapping=None,
        **kwargs,
    ):
        del positions, kv_caches, kwargs
        self.last_attn_metadata = attn_metadata
        self.last_multimodal_embeddings = multimodal_embeddings
        self.last_lora_mapping = lora_mapping
        batch, seq = input_ids.shape
        logits = torch.zeros((batch, seq, 16), dtype=torch.float32)
        if multimodal_embeddings is not None:
            logits = logits + 1.0
        if any(item is not None for item in (lora_mapping or [])):
            logits = logits + 2.0
        return logits


class _FakeTextOnlyModel:
    def __call__(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("should not be called in this test")


def _builder() -> InputBatchBuilder:
    return InputBatchBuilder(
        device=torch.device("cpu"),
        max_model_len=8,
        num_layers=1,
        kv_block_manager=KVBlockManager(
            kv_caches=[],
            kv_scale_caches=[],
            num_blocks_per_seq=2,
            block_size=2,
        ),
        inf_config=type("Cfg", (), {"kv_type": "fp16", "k_scale": 1.0, "v_scale": 1.0})(),
        stack_per_layer_carries=_stack,
        split_per_layer_carries=_split,
    )


def test_multimodal_processor_prepares_single_image_request(tmp_path: Path) -> None:
    processor = LiteMultiModalProcessor(
        model=_FakeMultiModalModel(),
        device=torch.device("cpu"),
    )
    request = {
        "multi_modal_data": {"image": [{"image": _image_url(tmp_path)}]},
    }

    processor.prepare_request(request)

    pixel_values = request["multi_modal_inputs"]["pixel_values"]
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.shape == (1, 1024)


def test_multimodal_processor_prepares_multi_image_request(tmp_path: Path) -> None:
    processor = LiteMultiModalProcessor(
        model=_FakeMultiModalModel(),
        device=torch.device("cpu"),
    )
    urls = _image_urls(tmp_path, 2)
    request = {
        "multi_modal_data": {"image": [{"image": urls[0]}, {"image": urls[1]}]},
    }

    processor.prepare_request(request)

    pixel_values = request["multi_modal_inputs"]["pixel_values"]
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.shape == (2, 1024)


def test_prefill_executor_computes_multimodal_embeddings(tmp_path: Path) -> None:
    model = _FakeMultiModalModel()
    processor = LiteMultiModalProcessor(model=model, device=torch.device("cpu"))
    scheduler = RequestScheduler(max_active_requests=1)
    request = {
        "slot_idx": 0,
        "is_prefill": True,
        "seq_len": 0,
        "input_ids": [11, 12, 13, 14],
        "generated_ids": [],
        "linear_attn_carry": [None],
        "linear_conv_carry": [None],
        "lora_id": None,
        "multi_modal_data": {"image": [{"image": _image_url(tmp_path)}]},
    }
    processor.prepare_request(request)
    scheduler.add_request("r1", request)

    executor = PrefillExecutor(
        model=model,
        input_batch_builder=_builder(),
        kv_caches=[],
        multimodal_processor=processor,
    )

    logits, reqs, last_flags = executor.execute(["r1"], scheduler, 2)

    assert logits.shape == (1, 2, 16)
    assert torch.all(logits == 1.0)
    assert reqs[0]["multi_modal_inputs"]["pixel_values"].shape == (1, 1024)
    assert model.last_pixel_values is not None
    assert model.last_attn_metadata is not None
    assert model.last_multimodal_embeddings is not None
    assert model.last_attn_metadata["multimodal_embeddings"].shape == (1, 8)
    assert last_flags == [False]


def test_prefill_executor_aggregates_multi_image_embeddings(tmp_path: Path) -> None:
    model = _FakeMultiModalModel()
    processor = LiteMultiModalProcessor(model=model, device=torch.device("cpu"))
    urls = _image_urls(tmp_path, 2)
    scheduler = RequestScheduler(max_active_requests=1)
    request = {
        "slot_idx": 0,
        "is_prefill": True,
        "seq_len": 0,
        "input_ids": [11, 12],
        "generated_ids": [],
        "linear_attn_carry": [None],
        "linear_conv_carry": [None],
        "lora_id": None,
        "multi_modal_data": {"image": [{"image": urls[0]}, {"image": urls[1]}]},
    }
    processor.prepare_request(request)
    scheduler.add_request("r1", request)

    executor = PrefillExecutor(
        model=model,
        input_batch_builder=_builder(),
        kv_caches=[],
        multimodal_processor=processor,
    )

    logits, _reqs, _last_flags = executor.execute(["r1"], scheduler, 2)

    assert logits.shape == (1, 2, 16)
    assert model.last_pixel_values.shape == (2, 1024)
    assert model.last_multimodal_embeddings is not None
    assert model.last_multimodal_embeddings.shape == (1, 8)


def test_prefill_executor_passes_multimodal_lora_contract(tmp_path: Path) -> None:
    model = _FakeMultiModalModel()
    processor = LiteMultiModalProcessor(model=model, device=torch.device("cpu"))
    scheduler = RequestScheduler(max_active_requests=1)
    request = {
        "slot_idx": 0,
        "is_prefill": True,
        "seq_len": 0,
        "input_ids": [11, 12],
        "generated_ids": [],
        "linear_attn_carry": [None],
        "linear_conv_carry": [None],
        "lora_id": "adapter-a",
        "lora_int_id": 7,
        "multi_modal_data": {"image": [{"image": _image_url(tmp_path)}]},
        "is_multimodal": True,
        "is_multimodal_lora": True,
    }
    processor.prepare_request(request)
    scheduler.add_request("r1", request)

    executor = PrefillExecutor(
        model=model,
        input_batch_builder=_builder(),
        kv_caches=[],
        multimodal_processor=processor,
    )

    logits, reqs, _last_flags = executor.execute(["r1"], scheduler, 2)

    assert logits.shape == (1, 2, 16)
    assert torch.all(logits == 3.0)
    assert model.last_multimodal_embeddings is not None
    assert model.last_lora_mapping == ["adapter-a"]
    assert model.last_attn_metadata["multimodal_lora_request_count"] == 1
    assert model.last_attn_metadata["has_multimodal_lora_requests"] is True
    assert reqs[0]["is_multimodal_lora"] is True


def test_multimodal_processor_rejects_text_only_model(tmp_path: Path) -> None:
    processor = LiteMultiModalProcessor(
        model=_FakeTextOnlyModel(),
        device=torch.device("cpu"),
    )
    request = {
        "multi_modal_data": {"image": [{"image": _image_url(tmp_path)}]},
    }

    try:
        processor.prepare_request(request)
    except ValueError as exc:
        assert str(exc) == "model does not support multimodal inputs"
    else:
        raise AssertionError("expected multimodal request to be rejected")
    assert processor.stats()["prepare_failures"] == 1


class _IdentityBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 4)
        self.last_input = None

    def forward(self, input_ids, positions, kv_caches, attn_metadata, lora_mapping=None):
        del positions, kv_caches, attn_metadata, lora_mapping
        self.last_input = input_ids
        return input_ids


def test_qwen2_vl_merges_multimodal_embeddings_into_hidden_inputs() -> None:
    model = Qwen2VLForCausalLM.__new__(Qwen2VLForCausalLM)
    nn.Module.__init__(model)
    model.model = _IdentityBackbone()
    model.lm_head = nn.Identity()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
    multimodal_embeddings = torch.tensor([[0.5, 1.0, 1.5, 2.0]], dtype=torch.float32)

    hidden_states = model(
        input_ids,
        positions,
        [None],
        {},
        multimodal_embeddings=multimodal_embeddings,
    )

    expected = model.model.embed_tokens(input_ids) + multimodal_embeddings.unsqueeze(1)
    assert torch.allclose(hidden_states, expected)
    assert torch.allclose(model.model.last_input, expected)


def test_multimodal_processor_stats_track_prepare_and_embeddings(tmp_path: Path) -> None:
    model = _FakeMultiModalModel()
    processor = LiteMultiModalProcessor(model=model, device=torch.device("cpu"))
    request = {
        "multi_modal_data": {"image": [{"image": _image_url(tmp_path)}]},
    }

    processor.prepare_request(request)
    embeddings = processor.get_multimodal_embeddings(request["multi_modal_inputs"])
    stats = processor.stats()

    assert embeddings is not None
    assert stats["prepared_requests"] == 1
    assert stats["prepared_images"] == 1
    assert stats["embedding_requests"] == 1
    assert stats["embeddings_computed"] == 1
    assert stats["avg_embedding_feature_dim"] == 8.0
    assert stats["prepared_images"] == 1
