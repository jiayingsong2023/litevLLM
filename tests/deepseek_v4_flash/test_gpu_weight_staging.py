from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.expert_cache import (
    DeepSeekV4FlashCacheAdmissionPolicy,
    DeepSeekV4FlashHotExpertPolicy,
)
from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_Q8_0,
    DeepSeekV4FlashTensor,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_weight_staging import (
    DeepSeekV4FlashGPUWeightStager,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)
from vllm.model_executor.models.deepseek_v4_flash.moe import grouped_expert_reference
from vllm.model_executor.models.deepseek_v4_flash.profiler import (
    DeepSeekV4FlashProfiler,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashGroupedExpertTensors,
    DeepSeekV4FlashQuantizedExpertPayload,
    open_deepseek_v4_flash_weight_store,
)

TARGET_GGUF = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


class _FakeGroupedExpertStore:
    def __init__(self) -> None:
        self.decode_count = 0
        self.raw_payload_read_count = 0
        self.tensor_payload_read_count = 0
        self.matrices: dict[tuple[str, int], torch.Tensor] = {}
        self.raw_payloads: dict[tuple[str, int], bytes] = {}
        self.tensor_payloads: dict[str, bytes] = {}

    def decode_grouped_expert_matrix(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> torch.Tensor:
        self.decode_count += 1
        return self.matrices[(tensor.name, expert_id)].clone()

    def raw_grouped_expert_payload(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> DeepSeekV4FlashQuantizedExpertPayload:
        self.raw_payload_read_count += 1
        input_size, output_size, _expert_count = tensor.dims
        payload = self.raw_payloads[(tensor.name, expert_id)]
        return DeepSeekV4FlashQuantizedExpertPayload(
            tensor_name=tensor.name,
            expert_id=expert_id,
            ggml_type=tensor.tensor_type,
            rows=output_size,
            columns=input_size,
            payload=memoryview(bytearray(payload)),
        )

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        raise AssertionError(f"unexpected matrix decode for {tensor.name}")

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise AssertionError(f"unexpected vector read for {tensor.name} as {dtype}")

    def tensor_payload(self, tensor: DeepSeekV4FlashTensor) -> memoryview:
        self.tensor_payload_read_count += 1
        return memoryview(bytearray(self.tensor_payloads[tensor.name]))


class _FakeStagingStore(_FakeGroupedExpertStore):
    def __init__(self) -> None:
        super().__init__()
        self.matrix_decode_count = 0
        self.vector_read_count = 0
        self.generic_matrices: dict[str, torch.Tensor] = {}
        self.vectors: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        self.matrix_decode_count += 1
        return self.generic_matrices[tensor.name].clone()

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        self.vector_read_count += 1
        return self.vectors[(tensor.name, dtype)].clone()


def _tensor(name: str, dims: tuple[int, ...] = (2, 2, 1)) -> DeepSeekV4FlashTensor:
    return DeepSeekV4FlashTensor(
        name=name,
        dims=dims,
        tensor_type=GGML_TYPE_Q8_0,
        offset=0,
        nbytes=0,
    )


def _patch_fake_cuda_to(monkeypatch: pytest.MonkeyPatch) -> None:
    original_to = torch.Tensor.to

    def fake_cuda_to(
        tensor: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        device = kwargs.get("device")
        if device is None and args and isinstance(args[0], torch.device | str):
            device = args[0]
        if torch.device(device).type == "cuda" if device is not None else False:
            filtered_kwargs = dict(kwargs)
            filtered_kwargs.pop("device", None)
            filtered_args = args[1:] if args and device == args[0] else args
            return original_to(tensor, *filtered_args, **filtered_kwargs).clone()
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_cuda_to)


class _FakeHotLayer:
    def __init__(
        self,
        layer_index: int,
        grouped_experts: DeepSeekV4FlashGroupedExpertTensors | None,
    ) -> None:
        self.layer_index = layer_index
        self.grouped_experts = grouped_experts


class _FakeHotBindings:
    def __init__(self) -> None:
        self.token_embedding = _tensor("token_embd.weight", dims=(2, 8))
        self.layers = (
            _FakeHotLayer(0, _hot_grouped_experts(0)),
            _FakeHotLayer(1, _hot_grouped_experts(1)),
            _FakeHotLayer(3, _hot_grouped_experts(3)),
        )


class _FakeHotStore(_FakeStagingStore):
    def __init__(self) -> None:
        super().__init__()
        self.bindings = _FakeHotBindings()


def _hot_grouped_experts(layer_idx: int) -> DeepSeekV4FlashGroupedExpertTensors:
    return DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor(f"blk.{layer_idx}.ffn_gate_exps.weight", dims=(2, 2, 4)),
        up=_tensor(f"blk.{layer_idx}.ffn_up_exps.weight", dims=(2, 2, 4)),
        down=_tensor(f"blk.{layer_idx}.ffn_down_exps.weight", dims=(2, 2, 4)),
    )


def test_gpu_weight_stager_rejects_negative_staging_budget() -> None:
    store = _FakeStagingStore()

    with pytest.raises(ValueError, match="max staged bytes"):
        DeepSeekV4FlashGPUWeightStager(
            store,
            device="cuda",
            max_staged_bytes=-1,
        )


def test_gpu_weight_stager_records_cache_hits_and_misses_without_gpu() -> None:
    store = _FakeStagingStore()
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    stager.record_cache_miss("dynamic", 16, tensor_name="a")
    stager.record_cache_hit("dynamic", tensor_name="a")
    stager.record_cache_miss("grouped", 32, tensor_name="b")

    assert stager.cache_stats() == {
        "dynamic_hits": 1,
        "dynamic_misses": 1,
        "grouped_hits": 0,
        "grouped_misses": 1,
        "loaded_bytes": 48,
        "lru_evictions": 0,
        "streamed_bytes": 0,
        "batched_payload_stage_calls": 0,
    }


def test_gpu_weight_stager_memory_stats_include_cache_stats() -> None:
    store = _FakeStagingStore()
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    stager.record_cache_miss("dynamic", 8, tensor_name="a")

    stats = stager.memory_stats()

    assert stats["loaded_bytes"] == 8
    assert stats["dynamic_misses"] == 1
    assert "lru_evictions" in stats
    assert "pinned_entries" in stats
    assert "streamed_bytes" in stats


def test_grouped_expert_payload_stage_reuses_grouped_payload_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_cuda_to(monkeypatch)
    store = _FakeGroupedExpertStore()
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("blk.2.ffn_gate_exps.weight", dims=(2, 2, 8)),
        up=_tensor("blk.2.ffn_up_exps.weight", dims=(2, 2, 8)),
        down=_tensor("blk.2.ffn_down_exps.weight", dims=(2, 2, 8)),
    )
    for tensor in (grouped.gate, grouped.up, grouped.down):
        for expert_id in (1, 3):
            store.raw_payloads[(tensor.name, expert_id)] = bytes(
                [expert_id, len(tensor.name) % 251]
            )
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=1 << 20,
    )
    expert_ids = torch.tensor([1, 3], dtype=torch.int64)

    first = stager.stage_grouped_expert_payloads_for_ids(
        grouped,
        expert_ids,
        layer_idx=2,
    )
    second = stager.stage_grouped_expert_payloads_for_ids(
        grouped,
        expert_ids,
        layer_idx=2,
    )

    assert second is not first
    assert first[0][1].payload.data_ptr() == second[0][1].payload.data_ptr()
    assert first[0][2].payload.data_ptr() == second[0][2].payload.data_ptr()
    assert first[0][3].payload.data_ptr() == second[0][3].payload.data_ptr()
    assert store.raw_payload_read_count == 6
    stats = stager.cache_stats()
    assert stats["batched_payload_stage_calls"] == 2
    assert stats["grouped_misses"] == 6
    assert stats["grouped_hits"] == 6
    assert "selected_payload_cache_hits" not in stats


def test_grouped_expert_payload_stage_does_not_cache_streamed_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_cuda_to(monkeypatch)
    store = _FakeGroupedExpertStore()
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("blk.2.ffn_gate_exps.weight", dims=(2, 2, 8)),
        up=_tensor("blk.2.ffn_up_exps.weight", dims=(2, 2, 8)),
        down=_tensor("blk.2.ffn_down_exps.weight", dims=(2, 2, 8)),
    )
    for tensor in (grouped.gate, grouped.up, grouped.down):
        store.raw_payloads[(tensor.name, 3)] = bytes([3, len(tensor.name) % 251])
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        cache_admission_policy=DeepSeekV4FlashCacheAdmissionPolicy(
            stream_experts=frozenset({(2, 3)}),
        ),
    )
    expert_ids = torch.tensor([3], dtype=torch.int64)

    first = stager.stage_grouped_expert_payloads_for_ids(
        grouped,
        expert_ids,
        layer_idx=2,
    )
    second = stager.stage_grouped_expert_payloads_for_ids(
        grouped,
        expert_ids,
        layer_idx=2,
    )

    assert second is not first
    assert store.raw_payload_read_count == 6
    assert stager.staged_bytes == 0
    assert first[0][1].payload.data_ptr() != second[0][1].payload.data_ptr()
    stats = stager.cache_stats()
    assert stats["batched_payload_stage_calls"] == 2
    assert "selected_payload_cache_hits" not in stats
    assert "selected_payload_cache_misses" not in stats
    assert stats["streamed_bytes"] == 12


def test_grouped_expert_payload_stage_rereads_evicted_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_cuda_to(monkeypatch)
    store = _FakeGroupedExpertStore()
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("blk.2.ffn_gate_exps.weight", dims=(2, 2, 8)),
        up=_tensor("blk.2.ffn_up_exps.weight", dims=(2, 2, 8)),
        down=_tensor("blk.2.ffn_down_exps.weight", dims=(2, 2, 8)),
    )
    for tensor in (grouped.gate, grouped.up, grouped.down):
        for expert_id in (1, 3):
            store.raw_payloads[(tensor.name, expert_id)] = bytes(
                [expert_id, len(tensor.name) % 251]
            )
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=6,
    )
    expert_ids = torch.tensor([1, 3], dtype=torch.int64)

    first = stager.stage_grouped_expert_payloads_for_ids(
        grouped,
        expert_ids,
        layer_idx=2,
    )
    second = stager.stage_grouped_expert_payloads_for_ids(
        grouped,
        expert_ids,
        layer_idx=2,
    )

    assert second is not first
    assert store.raw_payload_read_count == 12
    assert "selected_payload_cache_hits" not in stager.cache_stats()


def test_full_resident_grouped_payloads_are_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_cuda_to(monkeypatch)
    store = _FakeGroupedExpertStore()
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("blk.2.ffn_gate_exps.weight", dims=(2, 2, 8)),
        up=_tensor("blk.2.ffn_up_exps.weight", dims=(2, 2, 8)),
        down=_tensor("blk.2.ffn_down_exps.weight", dims=(2, 2, 8)),
    )
    for tensor in (grouped.gate, grouped.up, grouped.down):
        for expert_id in (1, 3):
            store.raw_payloads[(tensor.name, expert_id)] = bytes(
                [expert_id, len(tensor.name) % 251]
            )
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=6,
    )

    stager.enable_full_resident_mode()
    first = stager.stage_grouped_expert_payloads_for_ids(
        grouped,
        torch.tensor([1, 3], dtype=torch.int64),
        layer_idx=2,
    )
    second = stager.stage_grouped_expert_payloads_for_ids(
        grouped,
        torch.tensor([1, 3], dtype=torch.int64),
        layer_idx=2,
    )

    assert stager.full_resident_enabled is True
    assert first[0][1].payload.data_ptr() == second[0][1].payload.data_ptr()
    assert store.raw_payload_read_count == 6
    stats = stager.memory_stats()
    assert stats["grouped_entries"] == 6
    assert stats["pinned_entries"] == 6
    assert stats["streamed_bytes"] == 0
    assert stats["lru_evictions"] == 0


def test_grouped_expert_payload_hits_refresh_grouped_lru_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_cuda_to(monkeypatch)
    store = _FakeGroupedExpertStore()
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("blk.2.ffn_gate_exps.weight", dims=(2, 2, 8)),
        up=_tensor("blk.2.ffn_up_exps.weight", dims=(2, 2, 8)),
        down=_tensor("blk.2.ffn_down_exps.weight", dims=(2, 2, 8)),
    )
    for tensor in (grouped.gate, grouped.up, grouped.down):
        for expert_id in (1, 2, 3):
            store.raw_payloads[(tensor.name, expert_id)] = bytes(
                [expert_id, len(tensor.name) % 251]
            )
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=12,
    )

    for expert_id in (1, 2, 1, 3, 1):
        stager.stage_grouped_expert_payloads_for_ids(
            grouped,
            torch.tensor([expert_id], dtype=torch.int64),
            layer_idx=2,
        )

    assert store.raw_payload_read_count == 9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_warm_static_decode_weights_makes_output_q8_payload_resident() -> None:
    store = _FakeStagingStore()
    tensor = _tensor("output.weight", dims=(4, 4))
    store.tensor_payloads[tensor.name] = b"\x01\x02\x03\x04"
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    stager.warm_static_decode_weights(output_weight=tensor)
    before = stager.memory_stats()
    staged = stager.stage_q8_raw_payload(tensor)
    after = stager.memory_stats()

    assert staged.device.type == "cuda"
    assert after["dynamic_hits"] == before["dynamic_hits"] + 1
    assert store.tensor_payload_read_count == 1


def test_model_warm_decode_static_weights_uses_output_head_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_head = _tensor("output.weight", dims=(4, 4))
    model = DeepSeekV4FlashForCausalLM(
        weight_store=type(
            "FakeStore",
            (),
            {"bindings": type("FakeBindings", (), {"output_head": output_head})()},
        )()
    )

    class FakeStager:
        def __init__(self) -> None:
            self.output_weights: list[DeepSeekV4FlashTensor] = []

        def warm_static_decode_weights(
            self,
            *,
            output_weight: DeepSeekV4FlashTensor,
        ) -> None:
            self.output_weights.append(output_weight)

    stager = FakeStager()
    monkeypatch.setattr(model, "_get_gpu_weight_stager", lambda device: stager)

    model.warm_decode_static_weights(torch.device("cuda"))

    assert stager.output_weights == [output_head]


def test_model_hot_expert_preparation_prefers_pinned_experts_and_skips_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeHotStore()
    model = DeepSeekV4FlashForCausalLM(weight_store=store)

    class FakeStager:
        hot_expert_policy = DeepSeekV4FlashHotExpertPolicy(
            pinned_experts=frozenset({(3, 2), (1, 3), (1, 1)})
        )

        def __init__(self) -> None:
            self.payload_calls: list[tuple[str, int, int | None]] = []

        def stage_grouped_expert_payload(
            self,
            tensor: DeepSeekV4FlashTensor,
            expert_id: int,
            *,
            layer_idx: int | None = None,
        ) -> object:
            if tensor.name == "blk.1.ffn_up_exps.weight" and expert_id == 3:
                raise KeyError(tensor.name)
            self.payload_calls.append((tensor.name, expert_id, layer_idx))
            return object()

        def memory_stats(self) -> dict[str, int | None]:
            return {"grouped_entries": len(self.payload_calls)}

    stager = FakeStager()
    monkeypatch.setattr(model, "_get_gpu_weight_stager", lambda device: stager)

    stats = model.prepare_deepseek_hot_experts(
        device=torch.device("cuda"),
        max_layers=1,
        experts_per_layer=2,
    )

    assert stats == {"grouped_entries": 5}
    assert stager.payload_calls == [
        ("blk.1.ffn_gate_exps.weight", 1, 1),
        ("blk.1.ffn_up_exps.weight", 1, 1),
        ("blk.1.ffn_down_exps.weight", 1, 1),
        ("blk.1.ffn_gate_exps.weight", 3, 1),
        ("blk.1.ffn_down_exps.weight", 3, 1),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_caches_decoded_grouped_expert_matrix() -> None:
    store = _FakeGroupedExpertStore()
    tensor = _tensor("blk.1.ffn_gate_exps.weight")
    store.matrices[(tensor.name, 0)] = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    first = stager.stage_grouped_expert_matrix(tensor, 0)
    second = stager.stage_grouped_expert_matrix(tensor, 0)

    assert first.device.type == "cuda"
    assert first.data_ptr() == second.data_ptr()
    assert store.decode_count == 1
    torch.testing.assert_close(first.cpu(), store.matrices[(tensor.name, 0)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_streams_grouped_expert_when_budget_is_exceeded() -> None:
    store = _FakeGroupedExpertStore()
    tensor = _tensor("blk.1.ffn_down_exps.weight")
    store.matrices[(tensor.name, 0)] = torch.eye(2, dtype=torch.float32)
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=1,
    )

    staged = stager.stage_grouped_expert_matrix(tensor, 0)

    assert staged.device.type == "cuda"
    assert stager.staged_bytes == 0
    stats = stager.cache_stats()
    assert stats["grouped_misses"] == 0
    assert stats["loaded_bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_streams_grouped_expert_refused_by_admission() -> None:
    store = _FakeGroupedExpertStore()
    tensor = _tensor("blk.1.ffn_down_exps.weight")
    store.matrices[(tensor.name, 2)] = torch.eye(2, dtype=torch.float32)
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        cache_admission_policy=DeepSeekV4FlashCacheAdmissionPolicy(
            stream_experts=frozenset({(1, 2)}),
        ),
    )

    first = stager.stage_grouped_expert_matrix(tensor, 2, layer_idx=1)
    second = stager.stage_grouped_expert_matrix(tensor, 2, layer_idx=1)

    assert first.device.type == "cuda"
    assert second.device.type == "cuda"
    assert first.data_ptr() != second.data_ptr()
    assert store.decode_count == 2
    assert stager.staged_bytes == 0
    stats = stager.memory_stats()
    assert stats["streamed_bytes"] == 32
    assert stats["grouped_misses"] == 0
    assert stats["loaded_bytes"] == 0
    assert stats["grouped_entries"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_streams_raw_payload_refused_by_admission() -> None:
    store = _FakeGroupedExpertStore()
    tensor = _tensor("blk.1.ffn_gate_exps.weight")
    store.raw_payloads[(tensor.name, 2)] = b"\x01\x02\x03\x04"
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        cache_admission_policy=DeepSeekV4FlashCacheAdmissionPolicy(
            stream_experts=frozenset({(1, 2)}),
        ),
    )

    first = stager.stage_grouped_expert_payload(tensor, 2, layer_idx=1)
    second = stager.stage_grouped_expert_payload(tensor, 2, layer_idx=1)

    assert first.payload.device.type == "cuda"
    assert second.payload.device.type == "cuda"
    assert first.payload.data_ptr() != second.payload.data_ptr()
    assert store.raw_payload_read_count == 2
    assert stager.staged_bytes == 0
    stats = stager.memory_stats()
    assert stats["streamed_bytes"] == 8
    assert stats["grouped_misses"] == 0
    assert stats["loaded_bytes"] == 0
    assert stats["grouped_entries"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_tracks_staged_bytes_once_per_cache_key() -> None:
    store = _FakeStagingStore()
    matrix_tensor = _tensor("blk.1.ffn_gate_inp.weight", dims=(2, 2))
    vector_tensor = _tensor("blk.1.attn_norm.weight", dims=(4,))
    store.generic_matrices[matrix_tensor.name] = torch.eye(2, dtype=torch.float32)
    store.vectors[(vector_tensor.name, torch.float16)] = torch.ones(
        4,
        dtype=torch.float16,
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    stager.stage_matrix(matrix_tensor)
    stager.stage_matrix(matrix_tensor)
    stager.stage_vector(vector_tensor, dtype=torch.float16)
    stager.stage_vector(vector_tensor, dtype=torch.float16)

    assert stager.staged_bytes == 24
    assert stager.memory_stats() == {
        "staged_bytes": 24,
        "max_staged_bytes": None,
        "full_resident_enabled": 0,
        "dynamic_entries": 2,
        "grouped_entries": 0,
        "dynamic_hits": 2,
        "dynamic_misses": 2,
        "grouped_hits": 0,
        "grouped_misses": 0,
        "loaded_bytes": 24,
        "lru_evictions": 0,
        "pinned_entries": 0,
        "streamed_bytes": 0,
        "batched_payload_stage_calls": 0,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_lru_eviction_removes_oldest_non_pinned_entry() -> None:
    store = _FakeStagingStore()
    first_tensor = _tensor("blk.1.first.weight", dims=(2, 2))
    second_tensor = _tensor("blk.1.second.weight", dims=(2, 2))
    third_tensor = _tensor("blk.1.third.weight", dims=(2, 2))
    store.generic_matrices[first_tensor.name] = torch.eye(2, dtype=torch.float32)
    store.generic_matrices[second_tensor.name] = torch.ones((2, 2), dtype=torch.float32)
    store.generic_matrices[third_tensor.name] = torch.full((2, 2), 3.0)
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=32,
    )

    first = stager.stage_matrix(first_tensor)
    stager.stage_matrix(second_tensor)
    stager.stage_matrix(third_tensor)
    reloaded_first = stager.stage_matrix(first_tensor)

    assert first.data_ptr() != reloaded_first.data_ptr()
    assert store.matrix_decode_count == 4
    assert stager.memory_stats()["lru_evictions"] == 2
    assert stager.staged_bytes == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_pinned_grouped_entries_survive_lru_eviction() -> None:
    store = _FakeStagingStore()
    expert_tensor = _tensor("blk.2.ffn_gate_exps.weight")
    matrix_tensor = _tensor("blk.2.router.weight", dims=(2, 2))
    new_tensor = _tensor("blk.2.output.weight", dims=(2, 2))
    store.matrices[(expert_tensor.name, 7)] = torch.eye(2, dtype=torch.float32)
    store.generic_matrices[matrix_tensor.name] = torch.ones((2, 2), dtype=torch.float32)
    store.generic_matrices[new_tensor.name] = torch.full((2, 2), 2.0)
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=32,
        hot_expert_policy=DeepSeekV4FlashHotExpertPolicy(
            pinned_experts=frozenset({(2, 7)})
        ),
    )

    pinned = stager.stage_grouped_expert_matrix(expert_tensor, 7, layer_idx=2)
    stager.stage_matrix(matrix_tensor)
    stager.stage_matrix(new_tensor)
    cached_pinned = stager.stage_grouped_expert_matrix(expert_tensor, 7, layer_idx=2)

    assert pinned.data_ptr() == cached_pinned.data_ptr()
    assert store.decode_count == 1
    assert stager.memory_stats()["pinned_entries"] == 1
    assert stager.memory_stats()["lru_evictions"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_pinned_raw_payload_entries_survive_lru_eviction() -> None:
    store = _FakeStagingStore()
    gate_tensor = _tensor("blk.2.ffn_gate_exps.weight")
    up_tensor = _tensor("blk.2.ffn_up_exps.weight")
    down_tensor = _tensor("blk.2.ffn_down_exps.weight")
    matrix_tensor = _tensor("blk.2.router.weight", dims=(2, 2))
    new_tensor = _tensor("blk.2.output.weight", dims=(2, 2))
    store.raw_payloads[(gate_tensor.name, 7)] = b"\x01\x02\x03\x04"
    store.raw_payloads[(up_tensor.name, 7)] = b"\x05\x06\x07\x08"
    store.raw_payloads[(down_tensor.name, 7)] = b"\x09\x0a\x0b\x0c"
    store.generic_matrices[matrix_tensor.name] = torch.ones((2, 2), dtype=torch.float32)
    store.generic_matrices[new_tensor.name] = torch.full((2, 2), 2.0)
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=43,
        hot_expert_policy=DeepSeekV4FlashHotExpertPolicy(
            pinned_experts=frozenset({(2, 7)})
        ),
    )

    first_gate = stager.stage_grouped_expert_payload(gate_tensor, 7, layer_idx=2)
    first_up = stager.stage_grouped_expert_payload(up_tensor, 7, layer_idx=2)
    first_down = stager.stage_grouped_expert_payload(down_tensor, 7, layer_idx=2)
    stager.stage_matrix(matrix_tensor)
    stager.stage_matrix(new_tensor)
    cached_gate = stager.stage_grouped_expert_payload(gate_tensor, 7, layer_idx=2)
    cached_up = stager.stage_grouped_expert_payload(up_tensor, 7, layer_idx=2)
    cached_down = stager.stage_grouped_expert_payload(down_tensor, 7, layer_idx=2)

    assert first_gate.payload.data_ptr() == cached_gate.payload.data_ptr()
    assert first_up.payload.data_ptr() == cached_up.payload.data_ptr()
    assert first_down.payload.data_ptr() == cached_down.payload.data_ptr()
    assert store.raw_payload_read_count == 3
    stats = stager.memory_stats()
    assert stats["pinned_entries"] == 3
    assert stats["grouped_hits"] == 3
    assert stats["lru_evictions"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_manual_pinned_grouped_entry_survives_eviction() -> None:
    store = _FakeStagingStore()
    expert_tensor = _tensor("blk.4.ffn_gate_exps.weight")
    first_tensor = _tensor("blk.4.router.weight", dims=(2, 2))
    second_tensor = _tensor("blk.4.output.weight", dims=(2, 2))
    store.matrices[(expert_tensor.name, 5)] = torch.eye(2, dtype=torch.float32)
    store.generic_matrices[first_tensor.name] = torch.ones((2, 2), dtype=torch.float32)
    store.generic_matrices[second_tensor.name] = torch.full((2, 2), 2.0)
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=32,
    )

    pinned = stager.stage_grouped_expert_matrix(expert_tensor, 5, layer_idx=4)
    stager.pin_grouped_expert(4, 5)
    stager.stage_matrix(first_tensor)
    stager.stage_matrix(second_tensor)
    cached_pinned = stager.stage_grouped_expert_matrix(expert_tensor, 5, layer_idx=4)

    assert pinned.data_ptr() == cached_pinned.data_ptr()
    assert stager.memory_stats()["pinned_entries"] == 1
    assert stager.memory_stats()["lru_evictions"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_all_pinned_overflow_streams_new_tensor() -> None:
    store = _FakeStagingStore()
    first_expert = _tensor("blk.3.ffn_gate_exps.weight")
    second_expert = _tensor("blk.3.ffn_up_exps.weight")
    new_tensor = _tensor("blk.3.router.weight", dims=(2, 2))
    store.matrices[(first_expert.name, 1)] = torch.eye(2, dtype=torch.float32)
    store.matrices[(second_expert.name, 2)] = torch.ones((2, 2), dtype=torch.float32)
    store.generic_matrices[new_tensor.name] = torch.full((2, 2), 4.0)
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=32,
        hot_expert_policy=DeepSeekV4FlashHotExpertPolicy(
            pinned_experts=frozenset({(3, 1), (3, 2)})
        ),
    )

    first = stager.stage_grouped_expert_matrix(first_expert, 1, layer_idx=3)
    second = stager.stage_grouped_expert_matrix(second_expert, 2, layer_idx=3)
    streamed = stager.stage_matrix(new_tensor)
    cached_first = stager.stage_grouped_expert_matrix(first_expert, 1, layer_idx=3)
    cached_second = stager.stage_grouped_expert_matrix(second_expert, 2, layer_idx=3)

    assert streamed.device.type == "cuda"
    assert first.data_ptr() == cached_first.data_ptr()
    assert second.data_ptr() == cached_second.data_ptr()
    stats = stager.memory_stats()
    assert stats["streamed_bytes"] == 16
    assert stats["lru_evictions"] == 0
    assert stats["pinned_entries"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_streams_new_tensor_when_staging_budget_is_exceeded() -> None:
    store = _FakeStagingStore()
    matrix_tensor = _tensor("blk.1.ffn_gate_inp.weight", dims=(2, 2))
    store.generic_matrices[matrix_tensor.name] = torch.eye(2, dtype=torch.float32)
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=15,
    )

    staged = stager.stage_matrix(matrix_tensor)

    assert staged.device.type == "cuda"
    assert stager.staged_bytes == 0
    stats = stager.cache_stats()
    assert stats["dynamic_misses"] == 0
    assert stats["loaded_bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_caches_decoded_matrix_by_cache_key() -> None:
    store = _FakeStagingStore()
    tensor = _tensor("blk.1.ffn_gate_inp.weight", dims=(2, 2))
    store.generic_matrices[tensor.name] = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    first = stager.stage_matrix(tensor)
    second = stager.stage_matrix(tensor)

    assert first.device.type == "cuda"
    assert first.data_ptr() == second.data_ptr()
    assert store.matrix_decode_count == 1
    torch.testing.assert_close(first.cpu(), store.generic_matrices[tensor.name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_caches_vector_by_cache_key() -> None:
    store = _FakeStagingStore()
    tensor = _tensor("blk.1.attn_norm.weight", dims=(4,))
    store.vectors[(tensor.name, torch.float32)] = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float32,
    )
    store.vectors[(tensor.name, torch.float16)] = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float16,
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    first = stager.stage_vector(tensor)
    second = stager.stage_vector(tensor)
    half = stager.stage_vector(tensor, dtype=torch.float16)
    cached_half = stager.stage_vector(tensor, dtype=torch.float16)

    assert first.device.type == "cuda"
    assert first.data_ptr() == second.data_ptr()
    assert half.device.type == "cuda"
    assert half.data_ptr() == cached_half.data_ptr()
    assert first.data_ptr() != half.data_ptr()
    assert store.vector_read_count == 2
    torch.testing.assert_close(first.cpu(), store.vectors[(tensor.name, torch.float32)])
    torch.testing.assert_close(half.cpu(), store.vectors[(tensor.name, torch.float16)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_profiles_matrix_and_vector_stage() -> None:
    store = _FakeStagingStore()
    matrix_tensor = _tensor("blk.1.attn_q.weight", dims=(2, 2))
    vector_tensor = _tensor("blk.1.attn_norm.weight", dims=(2,))
    store.generic_matrices[matrix_tensor.name] = torch.eye(2, dtype=torch.float32)
    store.vectors[(vector_tensor.name, torch.float32)] = torch.ones(
        2,
        dtype=torch.float32,
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    stager.profiler = DeepSeekV4FlashProfiler(enabled=True)

    stager.stage_matrix(matrix_tensor)
    stager.stage_vector(vector_tensor)

    events = stager.profiler.to_dict()["events"]
    assert [event["name"] for event in events] == [
        "stage_matrix",
        "stage_vector",
    ]
    assert events[0]["metadata"]["tensor"] == matrix_tensor.name
    assert events[0]["metadata"]["cache"] == "miss"
    assert events[1]["metadata"]["tensor"] == vector_tensor.name
    assert events[1]["metadata"]["cache"] == "miss"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_caches_output_q8_chunks_by_row_range() -> None:
    store = _FakeStagingStore()
    tensor = _tensor("output.weight", dims=(4, 4))
    values = torch.arange(16, dtype=torch.int8).reshape(4, 4)
    scales = torch.ones((4, 1), dtype=torch.float32)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    first_values, first_scales = stager.stage_output_q8_chunk(
        tensor,
        row_start=0,
        row_end=4,
        values=values,
        scales=scales,
    )
    second_values, second_scales = stager.stage_output_q8_chunk(
        tensor,
        row_start=0,
        row_end=4,
        values=values,
        scales=scales,
    )

    assert first_values.device.type == "cuda"
    assert first_scales.device.type == "cuda"
    assert first_values.data_ptr() == second_values.data_ptr()
    assert first_scales.data_ptr() == second_scales.data_ptr()
    stats = stager.cache_stats()
    assert stats["dynamic_misses"] == 1
    assert stats["dynamic_hits"] == 1


def test_gpu_weight_stager_rejects_output_q8_row_end_past_tensor_rows() -> None:
    store = _FakeStagingStore()
    tensor = _tensor("output.weight", dims=(128, 4))
    values = torch.arange(128, dtype=torch.int8).reshape(1, 128)
    scales = torch.ones((1, 4), dtype=torch.float32)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    with pytest.raises(ValueError, match="row range"):
        stager.stage_output_q8_chunk(
            tensor,
            row_start=3,
            row_end=5,
            values=values,
            scales=scales,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_streams_output_q8_when_budget_is_exceeded() -> None:
    store = _FakeStagingStore()
    tensor = _tensor("output.weight", dims=(4, 4))
    values = torch.arange(16, dtype=torch.int8).reshape(4, 4)
    scales = torch.ones((4, 1), dtype=torch.float32)
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=1,
    )

    staged_values, staged_scales = stager.stage_output_q8_chunk(
        tensor,
        row_start=0,
        row_end=4,
        values=values,
        scales=scales,
    )

    assert staged_values.device.type == "cuda"
    assert staged_scales.device.type == "cuda"
    assert stager.get_output_q8_chunk(tensor, row_start=0, row_end=4) is None
    assert stager.staged_bytes == 0
    stats = stager.cache_stats()
    assert stats["dynamic_misses"] == 0
    assert stats["loaded_bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_keeps_matrix_and_vector_cache_keys_distinct() -> None:
    store = _FakeStagingStore()
    tensor = _tensor("shared.name.weight", dims=(2, 2))
    store.generic_matrices[tensor.name] = torch.eye(2, dtype=torch.float32)
    store.vectors[(tensor.name, torch.float32)] = torch.ones(2, dtype=torch.float32)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    matrix = stager.stage_matrix(tensor)
    vector = stager.stage_vector(tensor)
    cached_matrix = stager.stage_matrix(tensor)
    cached_vector = stager.stage_vector(tensor)

    assert matrix.device.type == "cuda"
    assert vector.device.type == "cuda"
    assert matrix.data_ptr() == cached_matrix.data_ptr()
    assert vector.data_ptr() == cached_vector.data_ptr()
    assert matrix.data_ptr() != vector.data_ptr()
    assert store.matrix_decode_count == 1
    assert store.vector_read_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_clear_dynamic_cache_preserves_grouped_expert_cache() -> None:
    store = _FakeStagingStore()
    matrix_tensor = _tensor("blk.1.ffn_gate_inp.weight", dims=(2, 2))
    expert_tensor = _tensor("blk.1.ffn_gate_exps.weight")
    store.generic_matrices[matrix_tensor.name] = torch.eye(2, dtype=torch.float32)
    store.matrices[(expert_tensor.name, 0)] = torch.eye(2, dtype=torch.float32)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    first_matrix = stager.stage_matrix(matrix_tensor)
    first_expert = stager.stage_grouped_expert_matrix(expert_tensor, 0)
    assert stager.staged_bytes == 32
    stager.clear_dynamic_cache()
    assert stager.staged_bytes == 16
    second_matrix = stager.stage_matrix(matrix_tensor)
    second_expert = stager.stage_grouped_expert_matrix(expert_tensor, 0)

    assert first_matrix.data_ptr() != second_matrix.data_ptr()
    assert first_expert.data_ptr() == second_expert.data_ptr()
    assert store.matrix_decode_count == 2
    assert store.decode_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_backend_runs_staged_q2_iq2_expert_gemm_on_device() -> None:
    device = torch.device("cuda")
    hidden = torch.tensor([1.0, -2.0], device=device)
    gate = torch.tensor([[0.5, 1.0], [1.5, -0.5]], device=device)
    up = torch.tensor([[2.0, -1.0], [0.25, 0.75]], device=device)
    down = torch.tensor([[1.0, 0.5], [-0.25, 2.0]], device=device)
    backend = DeepSeekV4FlashGPUBackend()

    output = backend.routed_expert_gemm(
        hidden=hidden,
        gate_weight=gate,
        up_weight=up,
        down_weight=down,
    )

    expected = grouped_expert_reference(
        hidden.cpu(),
        gate.cpu(),
        up.cpu(),
        down.cpu(),
    ).to(device)
    assert output.device.type == "cuda"
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.skipif(
    os.environ.get("RUN_DEEPSEEK_REAL_GGUF_STAGING") != "1",
    reason="real GGUF staging is opt-in",
)
@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target GGUF not downloaded")
def test_real_gguf_grouped_expert_stages_to_gpu() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        layer = next(
            layer
            for layer in store.bindings.layers
            if layer.grouped_experts is not None
        )
        assert layer.grouped_experts is not None
        stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

        staged = stager.stage_grouped_expert(
            DeepSeekV4FlashGroupedExpertTensors(
                gate=layer.grouped_experts.gate,
                up=layer.grouped_experts.up,
                down=layer.grouped_experts.down,
            ),
            expert_id=0,
        )
        router_layer = next(layer for layer in store.bindings.layers if layer.router)
        assert router_layer.router is not None
        attention_norm_layer = next(
            layer for layer in store.bindings.layers if layer.attention_norm
        )
        assert attention_norm_layer.attention_norm is not None

        router = stager.stage_matrix(router_layer.router)
        attention_norm = stager.stage_vector(attention_norm_layer.attention_norm)
        output_norm = (
            stager.stage_vector(store.bindings.output_norm)
            if store.bindings.output_norm is not None
            else None
        )

    assert staged.gate.device.type == "cuda"
    assert staged.up.device.type == "cuda"
    assert staged.down.device.type == "cuda"
    assert router.device.type == "cuda"
    assert attention_norm.device.type == "cuda"
    if output_norm is not None:
        assert output_norm.device.type == "cuda"
    assert staged.gate.ndim == 2
    assert staged.up.ndim == 2
    assert staged.down.ndim == 2
    assert router.ndim == 2
    assert attention_norm.ndim == 1
    if output_norm is not None:
        assert output_norm.ndim == 1
    assert torch.isfinite(staged.gate).all()
    assert torch.isfinite(router).all()
    assert torch.isfinite(attention_norm).all()
    if output_norm is not None:
        assert torch.isfinite(output_norm).all()
