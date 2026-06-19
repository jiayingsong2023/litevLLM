from __future__ import annotations

import pytest
import torch
from fixtures import GGML_TYPE_Q2_K

from vllm.model_executor.models.deepseek_v4_flash.expert_cache import (
    DeepSeekV4FlashCacheAdmissionPolicy,
    DeepSeekV4FlashExpertPrefetchRequest,
)
from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    DeepSeekV4FlashTensor,
    ggml_tensor_nbytes,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_weight_staging import (
    DeepSeekV4FlashGPUWeightStager,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashGroupedExpertTensors,
    DeepSeekV4FlashQuantizedExpertPayload,
)


def _grouped_tensor(name: str) -> DeepSeekV4FlashTensor:
    dims = (256, 1, 2)
    return DeepSeekV4FlashTensor(
        name=name,
        dims=dims,
        tensor_type=GGML_TYPE_Q2_K,
        offset=0,
        nbytes=ggml_tensor_nbytes(dims, GGML_TYPE_Q2_K),
    )


def _grouped_expert_tensors() -> DeepSeekV4FlashGroupedExpertTensors:
    return DeepSeekV4FlashGroupedExpertTensors(
        gate=_grouped_tensor("blk.3.ffn_gate_exps.weight"),
        up=_grouped_tensor("blk.3.ffn_up_exps.weight"),
        down=_grouped_tensor("blk.3.ffn_down_exps.weight"),
    )


class _FakeRawPayloadStore:
    def __init__(self) -> None:
        self.raw_reads = 0
        self.payloads: dict[tuple[str, int], bytes] = {}

    def raw_grouped_expert_payload(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> DeepSeekV4FlashQuantizedExpertPayload:
        self.raw_reads += 1
        payload = self.payloads[(tensor.name, expert_id)]
        input_size, output_size, _expert_count = tensor.dims
        return DeepSeekV4FlashQuantizedExpertPayload(
            tensor_name=tensor.name,
            expert_id=expert_id,
            ggml_type=tensor.tensor_type,
            rows=output_size,
            columns=input_size,
            payload=memoryview(bytearray(payload)),
        )

    def decode_grouped_expert_matrix(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> torch.Tensor:
        raise AssertionError(f"unexpected grouped decode for {tensor.name}:{expert_id}")

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        raise AssertionError(f"unexpected matrix decode for {tensor.name}")

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise AssertionError(f"unexpected vector read for {tensor.name} as {dtype}")


class _FakeDecodedOnlyStore:
    def __init__(self) -> None:
        self.decode_reads = 0
        self.matrices: dict[tuple[str, int], torch.Tensor] = {}

    def raw_grouped_expert_payload(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> DeepSeekV4FlashQuantizedExpertPayload:
        raise NotImplementedError("raw payload unavailable")

    def decode_grouped_expert_matrix(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> torch.Tensor:
        self.decode_reads += 1
        return self.matrices[(tensor.name, expert_id)].clone()

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        raise AssertionError(f"unexpected matrix decode for {tensor.name}")

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise AssertionError(f"unexpected vector read for {tensor.name} as {dtype}")


class _FailingPrefetchStager:
    def prefetch_grouped_experts(
        self,
        tensors: DeepSeekV4FlashGroupedExpertTensors,
        request: DeepSeekV4FlashExpertPrefetchRequest,
    ) -> None:
        del tensors, request
        raise RuntimeError("prefetch failed")


class _RecordingPrefetchStager:
    def __init__(self) -> None:
        self.cache_admission_policy = DeepSeekV4FlashCacheAdmissionPolicy(
            stream_experts=frozenset({(3, 1)})
        )
        self.requests: list[DeepSeekV4FlashExpertPrefetchRequest] = []

    def prefetch_grouped_experts(
        self,
        tensors: DeepSeekV4FlashGroupedExpertTensors,
        request: DeepSeekV4FlashExpertPrefetchRequest,
    ) -> None:
        del tensors
        self.requests.append(request)


def test_expert_prefetch_request_rejects_negative_values() -> None:
    request = DeepSeekV4FlashExpertPrefetchRequest(layer_idx=2, expert_ids=(0, 1))

    assert request.layer_idx == 2
    assert request.expert_ids == (0, 1)
    with pytest.raises(ValueError, match="layer_idx"):
        DeepSeekV4FlashExpertPrefetchRequest(layer_idx=-1, expert_ids=(0,))
    with pytest.raises(ValueError, match="expert_ids"):
        DeepSeekV4FlashExpertPrefetchRequest(layer_idx=1, expert_ids=(0, -1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_prefetch_stages_raw_payloads_and_repeated_prefetch_records_hits() -> None:
    tensors = _grouped_expert_tensors()
    store = _FakeRawPayloadStore()
    payload = bytes(range(ggml_tensor_nbytes((256, 1), GGML_TYPE_Q2_K)))
    for tensor in (tensors.gate, tensors.up, tensors.down):
        store.payloads[(tensor.name, 0)] = payload
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    request = DeepSeekV4FlashExpertPrefetchRequest(layer_idx=3, expert_ids=(0,))

    stager.prefetch_grouped_experts(tensors, request)
    stager.prefetch_grouped_experts(tensors, request)

    stats = stager.cache_stats()
    assert store.raw_reads == 3
    assert stats["grouped_misses"] == 3
    assert stats["grouped_hits"] == 3
    assert stats["prefetch_misses"] == 3
    assert stats["prefetch_hits"] == 3
    assert stats["prefetch_payload_misses"] == 3
    assert stats["prefetch_payload_hits"] == 3
    assert stats["prefetch_payload_streamed_bytes"] == 0
    assert stats["prefetch_failures"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_demand_staging_reuses_payloads_loaded_by_prefetch() -> None:
    tensors = _grouped_expert_tensors()
    store = _FakeRawPayloadStore()
    payload = bytes(range(ggml_tensor_nbytes((256, 1), GGML_TYPE_Q2_K)))
    for tensor in (tensors.gate, tensors.up, tensors.down):
        store.payloads[(tensor.name, 0)] = payload
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    stager.prefetch_grouped_experts(
        tensors,
        DeepSeekV4FlashExpertPrefetchRequest(layer_idx=3, expert_ids=(0,)),
    )
    for tensor in (tensors.gate, tensors.up, tensors.down):
        stager.stage_grouped_expert_payload(tensor, 0, layer_idx=3)

    stats = stager.cache_stats()
    assert store.raw_reads == 3
    assert stats["grouped_misses"] == 3
    assert stats["grouped_hits"] == 3
    assert stats["prefetch_misses"] == 3
    assert stats["prefetch_payload_misses"] == 3
    assert stats["prefetch_payload_hits"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_prefetch_uses_injected_cuda_stream() -> None:
    tensors = _grouped_expert_tensors()
    store = _FakeRawPayloadStore()
    payload = bytes(range(ggml_tensor_nbytes((256, 1), GGML_TYPE_Q2_K)))
    for tensor in (tensors.gate, tensors.up, tensors.down):
        store.payloads[(tensor.name, 0)] = payload
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    stream = torch.cuda.Stream()

    stager.prefetch_grouped_experts(
        tensors,
        DeepSeekV4FlashExpertPrefetchRequest(layer_idx=3, expert_ids=(0,)),
        stream=stream,
    )
    stream.synchronize()

    stats = stager.cache_stats()
    assert stats["prefetch_misses"] == 3
    assert stats["prefetch_payload_misses"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_prefetch_falls_back_to_decoded_staging_without_raw_payload() -> None:
    tensors = _grouped_expert_tensors()
    store = _FakeDecodedOnlyStore()
    for tensor in (tensors.gate, tensors.up, tensors.down):
        store.matrices[(tensor.name, 1)] = torch.ones((1, 256), dtype=torch.float32)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    stager.prefetch_grouped_experts(
        tensors,
        DeepSeekV4FlashExpertPrefetchRequest(layer_idx=3, expert_ids=(1,)),
    )

    stats = stager.cache_stats()
    assert store.decode_reads == 3
    assert stats["grouped_misses"] == 3
    assert stats["prefetch_misses"] == 3
    assert stats["prefetch_payload_misses"] == 3


def test_model_best_effort_prefetch_records_failures() -> None:
    model = DeepSeekV4FlashForCausalLM()
    model.enable_deepseek_profile()

    model._prefetch_grouped_experts_best_effort(
        _FailingPrefetchStager(),
        _grouped_expert_tensors(),
        (0, 1),
        layer_idx=3,
    )

    profile = model.deepseek_profile()
    assert profile["counters"]["deepseek_prefetch_failures"] == 1


def test_model_best_effort_prefetch_skips_stream_only_experts() -> None:
    model = DeepSeekV4FlashForCausalLM()
    stager = _RecordingPrefetchStager()

    model._prefetch_grouped_experts_best_effort(
        stager,
        _grouped_expert_tensors(),
        (0, 1),
        layer_idx=3,
    )
    model._prefetch_grouped_experts_best_effort(
        stager,
        _grouped_expert_tensors(),
        (1,),
        layer_idx=3,
    )

    assert stager.requests == [
        DeepSeekV4FlashExpertPrefetchRequest(layer_idx=3, expert_ids=(0,))
    ]
