from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from fixtures import (
    GGML_TYPE_F16,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_Q2_K,
    write_minimal_deepseek_v4_flash_gguf,
)

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import ggml_tensor_nbytes
from vllm.model_executor.models.deepseek_v4_flash.moe import (
    grouped_expert_reference,
    hash_routed_expert_ids_reference,
    routed_moe_reference,
    topk_router_reference,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashWeightStoreError,
    open_deepseek_v4_flash_weight_store,
)

TARGET_GGUF = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


def test_grouped_expert_reference_matches_silu_gate_down_projection() -> None:
    hidden = torch.tensor([1.0, 2.0])
    gate = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    up = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    down = torch.eye(2)

    out = grouped_expert_reference(hidden, gate, up, down)

    expected = down.matmul(F.silu(gate.matmul(hidden)) * up.matmul(hidden))
    torch.testing.assert_close(out, expected)


def test_decode_grouped_expert_matrix_decodes_one_q2_k_slice(tmp_path) -> None:
    path = tmp_path / "deepseek-v4-flash.gguf"
    tensor_name = "blk.0.ffn_gate_exps.weight"
    input_size = 256
    output_size = 1
    expert_count = 2
    expert_nbytes = ggml_tensor_nbytes((input_size, output_size), GGML_TYPE_Q2_K)
    first_expert = bytes(expert_nbytes)
    second_expert = bytes([0x11]) * expert_nbytes
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight", tensor_name),
        tensor_types=(GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_Q2_K),
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
        decoded = store.decode_grouped_expert_matrix(tensor, expert_id=0)

    assert decoded.shape == (output_size, input_size)
    torch.testing.assert_close(decoded, torch.zeros((output_size, input_size)))


@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_grouped_expert_decode_reads_one_expert_slice() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        layer_0 = store.bindings.layers[0]
        assert layer_0.grouped_experts is not None

        gate = store.decode_grouped_expert_matrix(
            layer_0.grouped_experts.gate,
            expert_id=0,
        )

    assert gate.shape == (2048, 4096)


@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_layer0_hash_routing_reads_six_experts_for_token() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        token_to_experts = store.bindings.layers[0].expert_token_to_expert_ids
        assert token_to_experts is not None
        table = store.tensor_to_torch(token_to_experts, dtype=torch.int32)

    expert_ids = hash_routed_expert_ids_reference(table, token_id=1)

    assert expert_ids.shape == (6,)
    assert torch.all(expert_ids >= 0)
    assert torch.all(expert_ids < 256)


def test_decode_grouped_expert_matrix_rejects_invalid_expert_id(tmp_path) -> None:
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
            store.decode_grouped_expert_matrix(tensor, expert_id=2)


def test_decode_grouped_expert_matrix_rejects_unsupported_tensor_type(
    tmp_path,
) -> None:
    path = tmp_path / "deepseek-v4-flash.gguf"
    tensor_name = "blk.0.ffn_gate_exps.weight"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight", tensor_name),
        tensor_types=(GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F16),
        tensor_dims={
            "token_embd.weight": (2,),
            "blk.0.attn_q.weight": (2,),
            tensor_name: (256, 1, 2),
        },
    )

    with open_deepseek_v4_flash_weight_store(path) as store:
        tensor = store.model.tensors[tensor_name]
        with pytest.raises(
            DeepSeekV4FlashWeightStoreError,
            match="unsupported grouped expert tensor type",
        ):
            store.decode_grouped_expert_matrix(tensor, expert_id=0)


def test_decode_grouped_expert_matrix_rejects_non_grouped_tensor(tmp_path) -> None:
    path = tmp_path / "deepseek-v4-flash.gguf"
    tensor_name = "blk.0.ffn_gate_exps.weight"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight", tensor_name),
        tensor_types=(GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_IQ2_XXS),
        tensor_dims={
            "token_embd.weight": (2,),
            "blk.0.attn_q.weight": (2,),
            tensor_name: (256, 1),
        },
    )

    with open_deepseek_v4_flash_weight_store(path) as store:
        tensor = store.model.tensors[tensor_name]
        with pytest.raises(
            DeepSeekV4FlashWeightStoreError,
            match="must have dims",
        ):
            store.decode_grouped_expert_matrix(tensor, expert_id=0)


def test_routed_moe_reference_combines_topk_expert_outputs() -> None:
    hidden = torch.tensor([1.0, 2.0])
    router_weight = torch.tensor([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
    expert_outputs = {
        0: torch.tensor([1.0, 0.0]),
        1: torch.tensor([0.0, 2.0]),
        2: torch.tensor([3.0, 3.0]),
    }

    out = routed_moe_reference(
        hidden,
        router_weight,
        lambda expert_id, _hidden: expert_outputs[expert_id],
        top_k=2,
        correction_bias=torch.tensor([0.0, -10.0, 10.0]),
    )

    expert_ids, weights = topk_router_reference(
        hidden,
        router_weight,
        top_k=2,
        correction_bias=torch.tensor([0.0, -10.0, 10.0]),
    )
    expected = torch.zeros_like(hidden, dtype=torch.float32)
    for expert_id, weight in zip(expert_ids.tolist(), weights, strict=True):
        expected = expected + weight * expert_outputs[expert_id].to(torch.float32)
    torch.testing.assert_close(out, expected)
    assert expert_ids.tolist() == [2, 0]


def test_routed_moe_reference_rejects_empty_topk() -> None:
    with pytest.raises(ValueError, match="top_k must be > 0"):
        routed_moe_reference(
            torch.ones(2),
            torch.ones((2, 2)),
            lambda _expert_id, hidden: hidden,
            top_k=0,
        )
