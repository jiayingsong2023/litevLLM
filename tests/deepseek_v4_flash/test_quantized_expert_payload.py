from __future__ import annotations

import pytest
import torch
from fixtures import (
    GGML_TYPE_F16,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_Q2_K,
    write_minimal_deepseek_v4_flash_gguf,
)

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    DeepSeekV4FlashTensor,
    ggml_tensor_nbytes,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_weight_staging import (
    DeepSeekV4FlashGPUWeightStager,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashQuantizedExpertPayload,
    open_deepseek_v4_flash_weight_store,
)


def _grouped_tensor(
    name: str,
    *,
    ggml_type: int,
    dims: tuple[int, int, int] = (256, 1, 2),
) -> DeepSeekV4FlashTensor:
    return DeepSeekV4FlashTensor(
        name=name,
        dims=dims,
        tensor_type=ggml_type,
        offset=0,
        nbytes=ggml_tensor_nbytes(dims, ggml_type),
    )


@pytest.mark.parametrize("ggml_type", (GGML_TYPE_Q2_K, GGML_TYPE_IQ2_XXS))
def test_weight_store_raw_grouped_expert_payload_describes_one_expert_slice(
    tmp_path,
    ggml_type: int,
) -> None:
    path = tmp_path / "deepseek-v4-flash.gguf"
    tensor_name = "blk.0.ffn_gate_exps.weight"
    input_size = 256
    output_size = 1
    expert_count = 2
    expert_nbytes = ggml_tensor_nbytes((input_size, output_size), ggml_type)
    first_expert = bytes([0x11]) * expert_nbytes
    second_expert = bytes([0x22]) * expert_nbytes
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight", tensor_name),
        tensor_types=(GGML_TYPE_F16, GGML_TYPE_F16, ggml_type),
        tensor_dims={
            "token_embd.weight": (2,),
            "blk.0.attn_q.weight": (2,),
            tensor_name: (input_size, output_size, expert_count),
        },
        tensor_payloads={
            "token_embd.weight": b"\x00" * 4,
            "blk.0.attn_q.weight": b"\x00" * 4,
            tensor_name: first_expert + second_expert,
        },
    )

    with open_deepseek_v4_flash_weight_store(path) as store:
        tensor = store.model.tensors[tensor_name]
        payload = store.raw_grouped_expert_payload(tensor, expert_id=1)
        try:
            assert payload == DeepSeekV4FlashQuantizedExpertPayload(
                tensor_name=tensor_name,
                expert_id=1,
                ggml_type=ggml_type,
                rows=output_size,
                columns=input_size,
                payload=payload.payload,
            )
            assert payload.payload.tobytes() == second_expert
        finally:
            payload.payload.release()


def test_weight_store_raw_grouped_expert_payload_rejects_invalid_expert_id(
    tmp_path,
) -> None:
    path = tmp_path / "deepseek-v4-flash.gguf"
    tensor_name = "blk.0.ffn_gate_exps.weight"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight", tensor_name),
        tensor_types=(GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_Q2_K),
        tensor_dims={
            "token_embd.weight": (2,),
            "blk.0.attn_q.weight": (2,),
            tensor_name: (256, 1, 2),
        },
    )

    with open_deepseek_v4_flash_weight_store(path) as store:
        tensor = store.model.tensors[tensor_name]
        with pytest.raises(ValueError, match="expert id out of range"):
            store.raw_grouped_expert_payload(tensor, expert_id=2)


class _FakeRawPayloadStore:
    def __init__(self) -> None:
        self.read_count = 0
        self.payloads: dict[tuple[str, int], bytes] = {}

    def raw_grouped_expert_payload(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> DeepSeekV4FlashQuantizedExpertPayload:
        self.read_count += 1
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("ggml_type", (GGML_TYPE_Q2_K, GGML_TYPE_IQ2_XXS))
def test_gpu_weight_stager_caches_raw_grouped_expert_payload(
    ggml_type: int,
) -> None:
    store = _FakeRawPayloadStore()
    tensor = _grouped_tensor("blk.1.ffn_gate_exps.weight", ggml_type=ggml_type)
    payload_bytes = bytes(range(ggml_tensor_nbytes((256, 1), ggml_type)))
    store.payloads[(tensor.name, 0)] = payload_bytes
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    first = stager.stage_grouped_expert_payload(tensor, 0, layer_idx=1)
    second = stager.stage_grouped_expert_payload(tensor, 0, layer_idx=1)

    assert first.tensor_name == tensor.name
    assert first.expert_id == 0
    assert first.ggml_type == ggml_type
    assert first.rows == 1
    assert first.columns == 256
    assert first.payload.dtype == torch.uint8
    assert first.payload.device.type == "cuda"
    assert first.payload.data_ptr() == second.payload.data_ptr()
    assert store.read_count == 1
    assert first.payload.cpu().tolist() == list(payload_bytes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_streams_raw_payload_when_budget_exceeded() -> None:
    store = _FakeRawPayloadStore()
    tensor = _grouped_tensor("blk.1.ffn_up_exps.weight", ggml_type=GGML_TYPE_Q2_K)
    payload_bytes = bytes(range(ggml_tensor_nbytes((256, 1), GGML_TYPE_Q2_K)))
    store.payloads[(tensor.name, 0)] = payload_bytes
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=1,
    )

    first = stager.stage_grouped_expert_payload(tensor, 0)
    second = stager.stage_grouped_expert_payload(tensor, 0)

    assert first.payload.device.type == "cuda"
    assert second.payload.device.type == "cuda"
    assert stager.staged_bytes == 0
    stats = stager.memory_stats()
    assert stats["grouped_misses"] == 0
    assert stats["loaded_bytes"] == 0
    assert stats["streamed_bytes"] == 2 * len(payload_bytes)
    assert stats["grouped_entries"] == 0
