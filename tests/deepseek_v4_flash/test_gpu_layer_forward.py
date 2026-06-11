from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_Q8_0,
    DeepSeekV4FlashTensor,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_layers import (
    deepseek_v4_flash_router_topk,
    deepseek_v4_flash_sliding_layer_forward,
    deepseek_v4_flash_staged_matrix_projection,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_weight_staging import (
    DeepSeekV4FlashGPUWeightStager,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashGroupedExpertTensors,
    DeepSeekV4FlashHyperConnectionTensors,
    DeepSeekV4FlashLayerSemanticBindings,
    open_deepseek_v4_flash_weight_store,
)

TARGET_GGUF = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


class _FakeLayerStore:
    def __init__(self) -> None:
        self.matrices: dict[str, torch.Tensor] = {}
        self.vectors: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self.expert_matrices: dict[tuple[str, int], torch.Tensor] = {}

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        return self.matrices[tensor.name].clone()

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.vectors[(tensor.name, dtype)].clone()

    def decode_grouped_expert_matrix(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> torch.Tensor:
        return self.expert_matrices[(tensor.name, expert_id)].clone()


class _RecordingBackend:
    def __init__(self) -> None:
        self.sliding_attention_calls = 0
        self.routed_expert_ids: list[int] = []
        self.query_widths: list[int] = []

    def sliding_attention(
        self,
        *,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        attn_sinks: torch.Tensor | None,
        token_idx: int,
    ) -> torch.Tensor:
        assert query.is_cuda
        assert kv_rows.is_cuda
        assert attn_sinks is None or attn_sinks.is_cuda
        assert token_idx == 3
        self.sliding_attention_calls += 1
        self.query_widths.append(query.numel())
        return query

    def routed_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        assert hidden.is_cuda
        assert gate_weight.is_cuda
        assert up_weight.is_cuda
        assert down_weight.is_cuda
        self.routed_expert_ids.append(int(gate_weight[0, 0].item()))
        return torch.ones_like(hidden, dtype=torch.float32)


class _MarkerBackend(_RecordingBackend):
    def sliding_attention(
        self,
        *,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        attn_sinks: torch.Tensor | None,
        token_idx: int,
    ) -> torch.Tensor:
        self.sliding_attention_calls += 1
        self.query_widths.append(query.numel())
        return torch.zeros_like(query)

    def routed_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        marker = float(gate_weight[0, 0].item())
        self.routed_expert_ids.append(int(marker))
        return torch.full_like(hidden, marker, dtype=torch.float32)


def _tensor(name: str, dims: tuple[int, ...]) -> DeepSeekV4FlashTensor:
    return DeepSeekV4FlashTensor(
        name=name,
        dims=dims,
        tensor_type=GGML_TYPE_Q8_0,
        offset=0,
        nbytes=0,
    )


def _add_identity_layer_weights(
    store: _FakeLayerStore,
    tensors: dict[str, DeepSeekV4FlashTensor],
    *,
    hidden_size: int,
    num_experts: int,
) -> None:
    store.vectors[(tensors["attn_norm"].name, torch.float32)] = torch.ones(hidden_size)
    store.vectors[(tensors["ffn_norm"].name, torch.float32)] = torch.ones(hidden_size)
    store.matrices[tensors["attn_q"].name] = torch.eye(hidden_size)
    store.matrices[tensors["attn_out"].name] = torch.eye(hidden_size)
    store.matrices[tensors["router"].name] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    for expert_id in range(num_experts):
        gate = torch.eye(hidden_size)
        gate[0, 0] = float(expert_id)
        store.expert_matrices[(tensors["gate"].name, expert_id)] = gate
        store.expert_matrices[(tensors["up"].name, expert_id)] = torch.eye(hidden_size)
        store.expert_matrices[(tensors["down"].name, expert_id)] = torch.eye(
            hidden_size
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_sliding_layer_forward_uses_staged_attention_and_selected_experts() -> None:
    hidden_size = 4
    num_experts = 3
    store = _FakeLayerStore()
    tensors = {
        "attn_norm": _tensor("blk.0.attn_norm.weight", (hidden_size,)),
        "attn_q": _tensor("blk.0.attn_q.weight", (hidden_size, hidden_size)),
        "attn_out": _tensor("blk.0.attn_output.weight", (hidden_size, hidden_size)),
        "ffn_norm": _tensor("blk.0.ffn_norm.weight", (hidden_size,)),
        "router": _tensor("blk.0.ffn_gate_inp.weight", (hidden_size, num_experts)),
        "gate": _tensor("blk.0.ffn_gate_exps.weight", (hidden_size, hidden_size, 3)),
        "up": _tensor("blk.0.ffn_up_exps.weight", (hidden_size, hidden_size, 3)),
        "down": _tensor("blk.0.ffn_down_exps.weight", (hidden_size, hidden_size, 3)),
    }
    _add_identity_layer_weights(
        store,
        tensors,
        hidden_size=hidden_size,
        num_experts=num_experts,
    )
    layer = DeepSeekV4FlashLayerSemanticBindings(
        layer_index=0,
        attention_norm=tensors["attn_norm"],
        attention_query=tensors["attn_q"],
        attention_output=tensors["attn_out"],
        ffn_norm=tensors["ffn_norm"],
        router=tensors["router"],
        grouped_experts=DeepSeekV4FlashGroupedExpertTensors(
            gate=tensors["gate"],
            up=tensors["up"],
            down=tensors["down"],
        ),
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    backend = _RecordingBackend()
    hidden = torch.tensor([1.0, 2.0, 0.0, 0.0], device="cuda")

    output = deepseek_v4_flash_sliding_layer_forward(
        hidden,
        layer=layer,
        stager=stager,
        backend=backend,
        token_idx=3,
        router_top_k=2,
    )

    assert output.device.type == "cuda"
    assert output.shape == hidden.shape
    assert backend.sliding_attention_calls == 1
    assert backend.routed_expert_ids == [1, 0]
    torch.testing.assert_close(
        deepseek_v4_flash_staged_matrix_projection(
            torch.tensor([1.0, 2.0], device="cuda"),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda"),
        ),
        torch.tensor([5.0, 11.0], device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_sliding_layer_forward_uses_two_stage_attention_output_projection() -> None:
    hidden_size = 4
    num_experts = 3
    store = _FakeLayerStore()
    tensors = {
        "attn_norm": _tensor("blk.0.attn_norm.weight", (hidden_size,)),
        "attn_q": _tensor("blk.0.attn_q.weight", (hidden_size, hidden_size)),
        "attn_q_b": _tensor("blk.0.attn_q_b.weight", (hidden_size, 8)),
        "attn_out_a": _tensor("blk.0.attn_output_a.weight", (1, 8)),
        "attn_out_b": _tensor("blk.0.attn_output_b.weight", (8, hidden_size)),
        "ffn_norm": _tensor("blk.0.ffn_norm.weight", (hidden_size,)),
        "router": _tensor("blk.0.ffn_gate_inp.weight", (hidden_size, num_experts)),
        "gate": _tensor("blk.0.ffn_gate_exps.weight", (hidden_size, hidden_size, 3)),
        "up": _tensor("blk.0.ffn_up_exps.weight", (hidden_size, hidden_size, 3)),
        "down": _tensor("blk.0.ffn_down_exps.weight", (hidden_size, hidden_size, 3)),
    }
    _add_identity_layer_weights(
        store,
        {
            **tensors,
            "attn_out": tensors["attn_out_a"],
        },
        hidden_size=hidden_size,
        num_experts=num_experts,
    )
    store.matrices[tensors["attn_q_b"].name] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    store.matrices[tensors["attn_out_a"].name] = torch.ones((8, 1))
    store.matrices[tensors["attn_out_b"].name] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    layer = DeepSeekV4FlashLayerSemanticBindings(
        layer_index=0,
        attention_norm=tensors["attn_norm"],
        attention_query=tensors["attn_q"],
        attention_query_b=tensors["attn_q_b"],
        attention_output=tensors["attn_out_a"],
        attention_output_a=tensors["attn_out_a"],
        attention_output_b=tensors["attn_out_b"],
        ffn_norm=tensors["ffn_norm"],
        router=tensors["router"],
        grouped_experts=DeepSeekV4FlashGroupedExpertTensors(
            gate=tensors["gate"],
            up=tensors["up"],
            down=tensors["down"],
        ),
    )
    backend = _MarkerBackend()

    output = deepseek_v4_flash_sliding_layer_forward(
        torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda"),
        layer=layer,
        stager=DeepSeekV4FlashGPUWeightStager(store, device="cuda"),
        backend=backend,
        token_idx=3,
        router_top_k=1,
    )

    assert output.shape == (hidden_size,)
    assert backend.query_widths == [8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_router_topk_applies_reference_routed_scaling_factor() -> None:
    _expert_ids, weights = deepseek_v4_flash_router_topk(
        torch.tensor([1.0, 2.0], device="cuda"),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda"),
        top_k=2,
    )

    torch.testing.assert_close(weights.sum(), torch.tensor(1.5, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_sliding_layer_forward_hash_routes_and_adds_shared_expert() -> None:
    hidden_size = 4
    store = _FakeLayerStore()
    tensors = {
        "attn_norm": _tensor("blk.0.attn_norm.weight", (hidden_size,)),
        "attn_q": _tensor("blk.0.attn_q.weight", (hidden_size, hidden_size)),
        "attn_out": _tensor("blk.0.attn_output.weight", (hidden_size, hidden_size)),
        "ffn_norm": _tensor("blk.0.ffn_norm.weight", (hidden_size,)),
        "router": _tensor("blk.0.ffn_gate_inp.weight", (hidden_size, 3)),
        "gate": _tensor("blk.0.ffn_gate_exps.weight", (hidden_size, hidden_size, 3)),
        "up": _tensor("blk.0.ffn_up_exps.weight", (hidden_size, hidden_size, 3)),
        "down": _tensor("blk.0.ffn_down_exps.weight", (hidden_size, hidden_size, 3)),
        "shared_gate": _tensor(
            "blk.0.ffn_gate_shexp.weight",
            (hidden_size, hidden_size),
        ),
        "shared_up": _tensor("blk.0.ffn_up_shexp.weight", (hidden_size, hidden_size)),
        "shared_down": _tensor(
            "blk.0.ffn_down_shexp.weight",
            (hidden_size, hidden_size),
        ),
        "token_to_experts": _tensor("blk.0.ffn_gate_tid2eid.weight", (2, 5)),
    }
    _add_identity_layer_weights(store, tensors, hidden_size=hidden_size, num_experts=3)
    for name, marker in (
        ("shared_gate", 10.0),
        ("shared_up", 1.0),
        ("shared_down", 1.0),
    ):
        matrix = torch.eye(hidden_size)
        matrix[0, 0] = marker
        store.matrices[tensors[name].name] = matrix
    store.vectors[(tensors["token_to_experts"].name, torch.int32)] = torch.tensor(
        [[0, 1, 2, 0, 2], [1, 2, 0, 1, 0]],
        dtype=torch.int32,
    )
    layer = DeepSeekV4FlashLayerSemanticBindings(
        layer_index=0,
        attention_norm=tensors["attn_norm"],
        attention_query=tensors["attn_q"],
        attention_output=tensors["attn_out"],
        ffn_norm=tensors["ffn_norm"],
        router=tensors["router"],
        grouped_experts=DeepSeekV4FlashGroupedExpertTensors(
            gate=tensors["gate"],
            up=tensors["up"],
            down=tensors["down"],
        ),
        shared_experts=DeepSeekV4FlashGroupedExpertTensors(
            gate=tensors["shared_gate"],
            up=tensors["shared_up"],
            down=tensors["shared_down"],
        ),
        expert_token_to_expert_ids=tensors["token_to_experts"],
    )
    backend = _MarkerBackend()

    output = deepseek_v4_flash_sliding_layer_forward(
        torch.zeros(hidden_size, device="cuda"),
        layer=layer,
        stager=DeepSeekV4FlashGPUWeightStager(store, device="cuda"),
        backend=backend,
        token_idx=3,
        token_id=4,
    )

    assert backend.routed_expert_ids == [2, 0, 10]
    torch.testing.assert_close(output, torch.full((hidden_size,), 11.5, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_sliding_layer_forward_expands_1d_hyper_connection_streams() -> None:
    hidden_size = 4
    store = _FakeLayerStore()
    tensors = {
        "attn_norm": _tensor("blk.0.attn_norm.weight", (hidden_size,)),
        "attn_q": _tensor("blk.0.attn_q.weight", (hidden_size, hidden_size)),
        "attn_out": _tensor("blk.0.attn_output.weight", (hidden_size, hidden_size)),
        "ffn_norm": _tensor("blk.0.ffn_norm.weight", (hidden_size,)),
        "router": _tensor("blk.0.ffn_gate_inp.weight", (hidden_size, 3)),
        "gate": _tensor("blk.0.ffn_gate_exps.weight", (hidden_size, hidden_size, 3)),
        "up": _tensor("blk.0.ffn_up_exps.weight", (hidden_size, hidden_size, 3)),
        "down": _tensor("blk.0.ffn_down_exps.weight", (hidden_size, hidden_size, 3)),
        "hc_fn": _tensor("blk.0.hc_attn_fn.weight", (16, 24)),
        "hc_base": _tensor("blk.0.hc_attn_base.weight", (24,)),
        "hc_scale": _tensor("blk.0.hc_attn_scale.weight", (3,)),
    }
    _add_identity_layer_weights(store, tensors, hidden_size=hidden_size, num_experts=3)
    store.matrices[tensors["hc_fn"].name] = torch.zeros(24, 16)
    store.vectors[(tensors["hc_base"].name, torch.float32)] = torch.zeros(24)
    store.vectors[(tensors["hc_scale"].name, torch.float32)] = torch.ones(3)
    layer = DeepSeekV4FlashLayerSemanticBindings(
        layer_index=0,
        attention_norm=tensors["attn_norm"],
        attention_query=tensors["attn_q"],
        attention_output=tensors["attn_out"],
        ffn_norm=tensors["ffn_norm"],
        router=tensors["router"],
        grouped_experts=DeepSeekV4FlashGroupedExpertTensors(
            gate=tensors["gate"],
            up=tensors["up"],
            down=tensors["down"],
        ),
        attention_hyper_connection=DeepSeekV4FlashHyperConnectionTensors(
            fn=tensors["hc_fn"],
            base=tensors["hc_base"],
            scale=tensors["hc_scale"],
        ),
    )

    output = deepseek_v4_flash_sliding_layer_forward(
        torch.zeros(hidden_size, device="cuda"),
        layer=layer,
        stager=DeepSeekV4FlashGPUWeightStager(store, device="cuda"),
        backend=_MarkerBackend(),
        token_idx=3,
        router_top_k=1,
    )

    assert output.device.type == "cuda"
    assert output.shape == (4, hidden_size)
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.skipif(
    os.environ.get("RUN_DEEPSEEK_REAL_GGUF_LAYER") != "1",
    reason="real GGUF layer smoke is opt-in",
)
@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target GGUF not downloaded")
def test_real_gguf_layer0_sliding_forward_smoke() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        layer = replace(
            store.bindings.layers[0],
            attention_hyper_connection=None,
            ffn_hyper_connection=None,
        )
        stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
        hidden = torch.zeros(4096, dtype=torch.float32, device="cuda")
        hidden[0] = 1.0

        output = deepseek_v4_flash_sliding_layer_forward(
            hidden,
            layer=layer,
            stager=stager,
            backend=DeepSeekV4FlashGPUBackend(),
            token_idx=0,
        )

    assert output.device.type == "cuda"
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.skipif(
    os.environ.get("RUN_DEEPSEEK_REAL_GGUF_LAYER") != "1",
    reason="real GGUF layer smoke is opt-in",
)
@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target GGUF not downloaded")
def test_real_gguf_layer0_reports_unsupported_1d_hyper_connection() -> None:
    with (
        open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store,
        pytest.raises(ValueError, match="1-D hyper-connection"),
    ):
        deepseek_v4_flash_sliding_layer_forward(
            torch.zeros(4096, dtype=torch.float32, device="cuda"),
            layer=store.bindings.layers[0],
            stager=DeepSeekV4FlashGPUWeightStager(store, device="cuda"),
            backend=DeepSeekV4FlashGPUBackend(),
            token_idx=0,
        )
