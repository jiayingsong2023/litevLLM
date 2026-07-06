# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.engine.decode_executor import DecodeExecutor
from vllm.engine.prefill_executor import PrefillExecutor


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, object, dict]] = []

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: object,
        attn_metadata: dict,
        **kwargs,
    ) -> torch.Tensor:
        self.calls.append((input_ids, positions, kv_caches, dict(attn_metadata)))
        assert kwargs == {"lora_mapping": attn_metadata.get("lora_mapping")}
        return torch.tensor([[1.0, 2.0]], dtype=torch.float32)


class _FakeBatchBuilder:
    def __init__(self) -> None:
        self.split_calls: list[tuple[dict, list[dict]]] = []

    def build_prefill(self, request_ids, scheduler, chunk_len):
        assert request_ids == ["prefill-1"]
        assert scheduler == "scheduler"
        assert chunk_len == 2
        return (
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
            {"lora_mapping": {"prefill-1": 0}},
            [{"request_id": "prefill-1"}],
            [True],
        )

    def build_decode_fast(
        self,
        request_ids,
        scheduler,
        *,
        fast_input_ids,
        fast_positions,
        fast_slot_mapping,
        fast_seq_lens,
    ):
        assert request_ids == ["decode-1"]
        assert scheduler == "scheduler"
        assert fast_input_ids.tolist() == [0]
        assert fast_positions.tolist() == [0]
        assert fast_slot_mapping.tolist() == [0]
        assert fast_seq_lens.tolist() == [1]
        return (
            torch.tensor([3], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
            {"lora_mapping": {"decode-1": 1}},
            [{"request_id": "decode-1"}],
        )

    def build_decode_batch(self, request_ids, scheduler):
        assert request_ids == ["decode-2"]
        assert scheduler == "scheduler"
        return (
            torch.tensor([4], dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
            {"lora_mapping": {"decode-2": 2}},
            [{"request_id": "decode-2"}],
        )

    def split_per_layer_carries(
        self,
        attn_metadata: dict,
        req_dicts: list[dict],
    ) -> None:
        self.split_calls.append((dict(attn_metadata), list(req_dicts)))


def test_prefill_executor_is_thin_model_call_wrapper() -> None:
    model = _FakeModel()
    builder = _FakeBatchBuilder()
    kv_caches = object()
    executor = PrefillExecutor(
        model=model,
        input_batch_builder=builder,
        kv_caches=kv_caches,
    )

    logits, req_dicts, is_last_chunk_flags = executor.execute(
        ["prefill-1"],
        "scheduler",
        chunk_len=2,
    )

    assert logits.tolist() == [[1.0, 2.0]]
    assert req_dicts == [{"request_id": "prefill-1"}]
    assert is_last_chunk_flags == [True]
    assert len(model.calls) == 1
    input_ids, positions, passed_kv_caches, attn_metadata = model.calls[0]
    assert input_ids.tolist() == [1, 2]
    assert positions.tolist() == [0, 1]
    assert passed_kv_caches is kv_caches
    assert attn_metadata == {"lora_mapping": {"prefill-1": 0}}
    assert builder.split_calls == [
        ({"lora_mapping": {"prefill-1": 0}}, [{"request_id": "prefill-1"}])
    ]


class _FakeMultimodalModel:
    def __init__(self) -> None:
        self.kwargs = None
        self.attn_metadata = None

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: object,
        attn_metadata: dict,
        **kwargs,
    ) -> torch.Tensor:
        del input_ids, positions, kv_caches
        self.attn_metadata = dict(attn_metadata)
        self.kwargs = dict(kwargs)
        return torch.tensor([[3.0, 4.0]], dtype=torch.float32)


class _FakeMultimodalProcessor:
    def build_prefill_inputs(self, req_dicts):
        assert req_dicts == [{"request_id": "prefill-1"}]
        return {
            "pixel_values": torch.ones((1, 1024)),
            "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
            "image_token_count": 4,
            "image_token_counts": [4],
            "image_token_id": 77,
        }

    def get_multimodal_embeddings(self, mm_inputs):
        assert mm_inputs["image_token_count"] == 4
        return torch.ones((1, 4, 8))


def test_prefill_executor_passes_multimodal_placeholder_metadata() -> None:
    model = _FakeMultimodalModel()
    builder = _FakeBatchBuilder()
    executor = PrefillExecutor(
        model=model,
        input_batch_builder=builder,
        kv_caches=object(),
        multimodal_processor=_FakeMultimodalProcessor(),
    )

    logits, _, _ = executor.execute(["prefill-1"], "scheduler", chunk_len=2)

    assert logits.tolist() == [[3.0, 4.0]]
    assert model.attn_metadata["image_token_count"] == 4
    assert model.attn_metadata["image_token_counts"] == [4]
    assert model.attn_metadata["image_token_id"] == 77
    assert model.attn_metadata["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert tuple(model.attn_metadata["multimodal_embeddings"].shape) == (1, 4, 8)
    assert tuple(model.kwargs["multimodal_embeddings"].shape) == (1, 4, 8)


def test_decode_executor_fast_path_is_thin_model_call_wrapper() -> None:
    model = _FakeModel()
    builder = _FakeBatchBuilder()
    kv_caches = object()
    executor = DecodeExecutor(
        model=model,
        input_batch_builder=builder,
        kv_caches=kv_caches,
        fast_input_ids=torch.tensor([0]),
        fast_positions=torch.tensor([0]),
        fast_slot_mapping=torch.tensor([0]),
        fast_seq_lens=torch.tensor([1]),
    )

    logits, req_dicts = executor.execute_sync_fast(["decode-1"], "scheduler")

    assert logits.tolist() == [[1.0, 2.0]]
    assert req_dicts == [{"request_id": "decode-1"}]
    assert len(model.calls) == 1
    input_ids, positions, passed_kv_caches, attn_metadata = model.calls[0]
    assert input_ids.tolist() == [3]
    assert positions.tolist() == [2]
    assert passed_kv_caches is kv_caches
    assert attn_metadata == {"lora_mapping": {"decode-1": 1}}
    assert builder.split_calls == [
        ({"lora_mapping": {"decode-1": 1}}, [{"request_id": "decode-1"}])
    ]


def test_decode_executor_batch_path_is_thin_model_call_wrapper() -> None:
    model = _FakeModel()
    builder = _FakeBatchBuilder()
    kv_caches = object()
    executor = DecodeExecutor(
        model=model,
        input_batch_builder=builder,
        kv_caches=kv_caches,
        fast_input_ids=torch.tensor([0]),
        fast_positions=torch.tensor([0]),
        fast_slot_mapping=torch.tensor([0]),
        fast_seq_lens=torch.tensor([1]),
    )

    logits, req_dicts = executor.execute_batch(["decode-2"], "scheduler")

    assert logits.tolist() == [[1.0, 2.0]]
    assert req_dicts == [{"request_id": "decode-2"}]
    assert len(model.calls) == 1
    input_ids, positions, passed_kv_caches, attn_metadata = model.calls[0]
    assert input_ids.tolist() == [4]
    assert positions.tolist() == [3]
    assert passed_kv_caches is kv_caches
    assert attn_metadata == {"lora_mapping": {"decode-2": 2}}
    assert builder.split_calls == [
        ({"lora_mapping": {"decode-2": 2}}, [{"request_id": "decode-2"}])
    ]
