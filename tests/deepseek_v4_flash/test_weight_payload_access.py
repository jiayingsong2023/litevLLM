from __future__ import annotations

import struct

import pytest
import torch
from fixtures import (
    GGML_TYPE_F16,
    GGML_TYPE_IQ2_XXS,
    write_minimal_deepseek_v4_flash_gguf,
)

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    ggml_tensor_nbytes,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    grouped_expert_payload_offset,
    open_deepseek_v4_flash_weight_store,
)


def test_tensor_to_torch_reads_f16_payload_and_survives_store_close(tmp_path) -> None:
    path = tmp_path / "deepseek-v4-flash.gguf"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight"),
        tensor_types=(GGML_TYPE_F16, GGML_TYPE_F16),
        tensor_dims={
            "token_embd.weight": (2,),
            "blk.0.attn_q.weight": (2,),
        },
        tensor_payloads={
            "token_embd.weight": struct.pack("<ee", 1.0, 2.0),
            "blk.0.attn_q.weight": struct.pack("<ee", 3.0, 4.0),
        },
    )

    store = open_deepseek_v4_flash_weight_store(path)
    tensor = store.tensor_to_torch(
        store.bindings.token_embedding,
        dtype=torch.float16,
    )
    store.close()

    assert tensor.tolist() == [1.0, 2.0]


def test_tensor_to_torch_accepts_shape_override(tmp_path) -> None:
    path = tmp_path / "deepseek-v4-flash.gguf"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight"),
        tensor_types=(GGML_TYPE_F16, GGML_TYPE_F16),
        tensor_dims={
            "token_embd.weight": (4,),
            "blk.0.attn_q.weight": (2,),
        },
        tensor_payloads={
            "token_embd.weight": struct.pack("<eeee", 1.0, 2.0, 3.0, 4.0),
            "blk.0.attn_q.weight": struct.pack("<ee", 5.0, 6.0),
        },
    )

    with open_deepseek_v4_flash_weight_store(path) as store:
        tensor = store.tensor_to_torch(
            store.bindings.token_embedding,
            dtype=torch.float16,
            shape=(2, 2),
        )

    assert tensor.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_decode_matrix_reads_f16_matrix_payload(tmp_path) -> None:
    path = tmp_path / "deepseek-v4-flash.gguf"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight"),
        tensor_types=(GGML_TYPE_F16, GGML_TYPE_F16),
        tensor_dims={
            "token_embd.weight": (2, 2),
            "blk.0.attn_q.weight": (2,),
        },
        tensor_payloads={
            "token_embd.weight": struct.pack("<eeee", 1.0, 2.0, 3.0, 4.0),
            "blk.0.attn_q.weight": struct.pack("<ee", 5.0, 6.0),
        },
    )

    with open_deepseek_v4_flash_weight_store(path) as store:
        matrix = store.decode_matrix(store.bindings.token_embedding)

    assert matrix.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_grouped_expert_payload_offset_uses_expert_major_slice_size() -> None:
    offset = grouped_expert_payload_offset(
        expert_id=2,
        projection_dims=(4096, 2048, 256),
        projection_type=GGML_TYPE_IQ2_XXS,
    )

    assert offset == 2 * ggml_tensor_nbytes((4096, 2048), GGML_TYPE_IQ2_XXS)


@pytest.mark.parametrize("expert_id", [-1, 256])
def test_grouped_expert_payload_offset_rejects_invalid_expert_id(
    expert_id: int,
) -> None:
    with pytest.raises(ValueError, match="expert id out of range"):
        grouped_expert_payload_offset(
            expert_id=expert_id,
            projection_dims=(4096, 2048, 256),
            projection_type=GGML_TYPE_IQ2_XXS,
        )


def test_grouped_expert_payload_offset_rejects_invalid_dims() -> None:
    with pytest.raises(ValueError, match="projection_dims must have exactly 3"):
        grouped_expert_payload_offset(
            expert_id=0,
            projection_dims=(4096, 2048),
            projection_type=GGML_TYPE_IQ2_XXS,
        )
