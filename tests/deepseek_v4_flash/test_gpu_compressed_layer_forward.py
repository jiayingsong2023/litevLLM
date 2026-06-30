from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.attention import (
    apply_deepseek_layer_rope_to_tail_reference,
)
from vllm.model_executor.models.deepseek_v4_flash.config import (
    DEEPSEEK_V4_FLASH_SHAPE,
)
from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_F32,
    GGML_TYPE_Q8_0,
    DeepSeekV4FlashTensor,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_layers import (
    _select_compressed_rows_with_indexer,
    _update_compressor_state,
    deepseek_v4_flash_compressed_layer_forward,
    deepseek_v4_flash_layer_forward,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_runtime import (
    DeepSeekV4FlashGPUCacheConfig,
    DeepSeekV4FlashGPURequestState,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_weight_staging import (
    DeepSeekV4FlashGPUWeightStager,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashCompressorTensors,
    DeepSeekV4FlashGroupedExpertTensors,
    DeepSeekV4FlashHyperConnectionTensors,
    DeepSeekV4FlashIndexerTensors,
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
        self.payloads: dict[str, bytes] = {}
        self.payload_read_count = 0

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

    def tensor_payload(self, tensor: DeepSeekV4FlashTensor) -> memoryview:
        self.payload_read_count += 1
        return memoryview(self.payloads[tensor.name])


class _RecordingCompressedBackend:
    def __init__(self) -> None:
        self.compressed_selected_rows: torch.Tensor | None = None
        self.compressed_rows_shape: tuple[int, ...] | None = None
        self.sliding_attention_calls = 0
        self.routed_expert_calls = 0

    def compressed_attention(
        self,
        *,
        query: torch.Tensor,
        compressed_rows: torch.Tensor,
        selected_rows: torch.Tensor,
    ) -> torch.Tensor:
        assert query.is_cuda
        assert compressed_rows.is_cuda
        assert selected_rows.is_cuda
        self.compressed_selected_rows = selected_rows.detach().clone()
        self.compressed_rows_shape = tuple(compressed_rows.shape)
        return query

    def sliding_attention(
        self,
        *,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        attn_sinks: torch.Tensor | None,
        token_idx: int,
    ) -> torch.Tensor:
        self.sliding_attention_calls += 1
        return query

    def routed_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        self.routed_expert_calls += 1
        return torch.zeros_like(hidden, dtype=torch.float32)


def _tensor(name: str, dims: tuple[int, ...]) -> DeepSeekV4FlashTensor:
    return DeepSeekV4FlashTensor(
        name=name,
        dims=dims,
        tensor_type=GGML_TYPE_Q8_0,
        offset=0,
        nbytes=0,
    )


def _f32_tensor(name: str, dims: tuple[int, ...]) -> DeepSeekV4FlashTensor:
    return DeepSeekV4FlashTensor(
        name=name,
        dims=dims,
        tensor_type=GGML_TYPE_F32,
        offset=0,
        nbytes=0,
    )


def _compressed_layer_fixture(
    *,
    doubled_width: bool = False,
    q8_attention_output: bool = False,
) -> tuple[
    _FakeLayerStore,
    DeepSeekV4FlashLayerSemanticBindings,
]:
    hidden_size = 4
    kv_width = 4
    attention_width = 32 if q8_attention_output else kv_width
    compressor_width = 8 if doubled_width else kv_width
    indexer_width = 128
    indexer_compressor_width = 256 if doubled_width else indexer_width
    tensors = {
        "attn_norm": _tensor("blk.2.attn_norm.weight", (hidden_size,)),
        "attn_q": _tensor("blk.2.attn_q.weight", (hidden_size, attention_width)),
        "attn_out": _tensor(
            "blk.2.attn_output.weight",
            (attention_width, hidden_size),
        ),
        "comp_kv": _tensor(
            "blk.2.attn_compressor_kv.weight",
            (hidden_size, compressor_width),
        ),
        "comp_gate": _tensor(
            "blk.2.attn_compressor_gate.weight",
            (hidden_size, compressor_width),
        ),
        "comp_ape": _tensor(
            "blk.2.attn_compressor_ape.weight",
            (compressor_width, 4),
        ),
        "comp_norm": _tensor("blk.2.attn_compressor_norm.weight", (kv_width,)),
        "index_q_b": _tensor(
            "blk.2.indexer.attn_q_b.weight",
            (hidden_size, indexer_width),
        ),
        "index_proj": _tensor("blk.2.indexer.proj.weight", (hidden_size, 1)),
        "index_comp_kv": _tensor(
            "blk.2.indexer_compressor_kv.weight",
            (hidden_size, indexer_compressor_width),
        ),
        "index_comp_gate": _tensor(
            "blk.2.indexer_compressor_gate.weight",
            (hidden_size, indexer_compressor_width),
        ),
        "index_comp_ape": _tensor(
            "blk.2.indexer_compressor_ape.weight",
            (indexer_compressor_width, 4),
        ),
        "index_comp_norm": _tensor(
            "blk.2.indexer_compressor_norm.weight",
            (indexer_width,),
        ),
        "ffn_norm": _tensor("blk.2.ffn_norm.weight", (hidden_size,)),
        "router": _tensor("blk.2.ffn_gate_inp.weight", (hidden_size, 1)),
        "gate": _tensor("blk.2.ffn_gate_exps.weight", (hidden_size, hidden_size, 1)),
        "up": _tensor("blk.2.ffn_up_exps.weight", (hidden_size, hidden_size, 1)),
        "down": _tensor("blk.2.ffn_down_exps.weight", (hidden_size, hidden_size, 1)),
        "hc_attn_fn": _tensor("blk.2.hc_attn_fn.weight", (hidden_size * 2, 8)),
        "hc_attn_base": _tensor("blk.2.hc_attn_base.weight", (8,)),
        "hc_attn_scale": _tensor("blk.2.hc_attn_scale.weight", (3,)),
        "hc_ffn_fn": _tensor("blk.2.hc_ffn_fn.weight", (hidden_size * 2, 8)),
        "hc_ffn_base": _tensor("blk.2.hc_ffn_base.weight", (8,)),
        "hc_ffn_scale": _tensor("blk.2.hc_ffn_scale.weight", (3,)),
    }
    store = _FakeLayerStore()
    for name in ("attn_norm", "comp_norm", "ffn_norm"):
        width = kv_width if name == "comp_norm" else hidden_size
        store.vectors[(tensors[name].name, torch.float32)] = torch.ones(width)
    store.vectors[(tensors["index_comp_norm"].name, torch.float32)] = torch.ones(
        indexer_width
    )
    comp_identity = torch.zeros((compressor_width, hidden_size), dtype=torch.float32)
    comp_identity[:hidden_size] = torch.eye(hidden_size)
    if doubled_width:
        comp_identity[hidden_size:] = 2.0 * torch.eye(hidden_size)
    index_identity = torch.zeros(
        (indexer_compressor_width, hidden_size),
        dtype=torch.float32,
    )
    index_identity[:hidden_size] = torch.eye(hidden_size)
    attention_query = torch.zeros((attention_width, hidden_size))
    attention_query[:hidden_size] = torch.eye(hidden_size)
    store.matrices[tensors["attn_q"].name] = attention_query
    if q8_attention_output:
        store.payloads[tensors["attn_out"].name] = b"".join(
            struct.pack("<e", 1.0)
            + bytes(1 if column == row else 0 for column in range(attention_width))
            for row in range(hidden_size)
        )
    else:
        store.matrices[tensors["attn_out"].name] = torch.eye(hidden_size)
    store.matrices[tensors["comp_kv"].name] = comp_identity
    comp_gate = torch.zeros((compressor_width, hidden_size), dtype=torch.float32)
    comp_gate[:, 0] = 10.0
    store.matrices[tensors["comp_gate"].name] = comp_gate
    store.matrices[tensors["comp_ape"].name] = torch.zeros((compressor_width, 4))
    store.matrices[tensors["index_q_b"].name] = torch.cat(
        [torch.eye(hidden_size), torch.zeros(hidden_size, indexer_width - hidden_size)],
        dim=1,
    )
    store.matrices[tensors["index_proj"].name] = torch.ones((hidden_size, 1))
    store.matrices[tensors["index_comp_kv"].name] = index_identity
    store.matrices[tensors["index_comp_gate"].name] = store.matrices[
        tensors["index_comp_kv"].name
    ].clone()
    store.matrices[tensors["index_comp_ape"].name] = torch.zeros(
        (indexer_compressor_width, 4)
    )
    store.matrices[tensors["router"].name] = torch.ones((1, hidden_size))
    store.matrices[tensors["hc_attn_fn"].name] = torch.zeros((hidden_size * 2, 8))
    store.matrices[tensors["hc_ffn_fn"].name] = torch.zeros((hidden_size * 2, 8))
    for name in (
        "hc_attn_base",
        "hc_ffn_base",
    ):
        store.vectors[(tensors[name].name, torch.float32)] = torch.zeros(8)
    for name in (
        "hc_attn_scale",
        "hc_ffn_scale",
    ):
        store.vectors[(tensors[name].name, torch.float32)] = torch.ones(3)
    for expert_tensor in ("gate", "up", "down"):
        store.expert_matrices[(tensors[expert_tensor].name, 0)] = torch.eye(hidden_size)
    layer = DeepSeekV4FlashLayerSemanticBindings(
        layer_index=2,
        attention_norm=tensors["attn_norm"],
        attention_query=tensors["attn_q"],
        attention_output=tensors["attn_out"],
        attention_compressor=DeepSeekV4FlashCompressorTensors(
            ape=tensors["comp_ape"],
            gate=tensors["comp_gate"],
            kv=tensors["comp_kv"],
            norm=tensors["comp_norm"],
        ),
        indexer=DeepSeekV4FlashIndexerTensors(
            query_b=tensors["index_q_b"],
            projection=tensors["index_proj"],
            compressor=DeepSeekV4FlashCompressorTensors(
                ape=tensors["index_comp_ape"],
                gate=tensors["index_comp_gate"],
                kv=tensors["index_comp_kv"],
                norm=tensors["index_comp_norm"],
            ),
        ),
        ffn_norm=tensors["ffn_norm"],
        router=tensors["router"],
        grouped_experts=DeepSeekV4FlashGroupedExpertTensors(
            gate=tensors["gate"],
            up=tensors["up"],
            down=tensors["down"],
        ),
        attention_hyper_connection=(
            DeepSeekV4FlashHyperConnectionTensors(
                fn=tensors["hc_attn_fn"],
                base=tensors["hc_attn_base"],
                scale=tensors["hc_attn_scale"],
            )
            if doubled_width
            else None
        ),
        ffn_hyper_connection=(
            DeepSeekV4FlashHyperConnectionTensors(
                fn=tensors["hc_ffn_fn"],
                base=tensors["hc_ffn_base"],
                scale=tensors["hc_ffn_scale"],
            )
            if doubled_width
            else None
        ),
    )
    return store, layer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_attention_decode_reuses_static_q8_projection_payload() -> None:
    store, layer = _compressed_layer_fixture(q8_attention_output=True)
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=16,
            hidden_size=4,
            kv_width=4,
            dtype=torch.float32,
            device="cuda",
        )
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    backend = _RecordingCompressedBackend()
    hidden = torch.ones((4,), dtype=torch.float32, device="cuda")

    first_output = deepseek_v4_flash_compressed_layer_forward(
        hidden,
        layer=layer,
        stager=stager,
        backend=backend,
        state=state,
        token_idx=0,
        router_top_k=1,
    )
    first_read_count = store.payload_read_count

    second_hidden = torch.tensor([0.0, 1.0, 0.0, 0.0], device="cuda")
    second_output = deepseek_v4_flash_compressed_layer_forward(
        second_hidden,
        layer=layer,
        stager=stager,
        backend=backend,
        state=state,
        token_idx=1,
        router_top_k=1,
    )

    assert first_read_count == 1
    assert store.payload_read_count == first_read_count
    torch.testing.assert_close(
        first_output,
        torch.tensor([2.0, 2.0, 2.0, 2.0], device="cuda"),
    )
    torch.testing.assert_close(
        second_output,
        torch.tensor([0.0, 3.0, 0.0, 0.0], device="cuda"),
    )


def _run_four_ratio4_tokens(
    *,
    state: DeepSeekV4FlashGPURequestState,
    layer: DeepSeekV4FlashLayerSemanticBindings,
    stager: DeepSeekV4FlashGPUWeightStager,
    backend: _RecordingCompressedBackend,
    token_offset: int = 0,
    hidden_by_token: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | None = None,
) -> None:
    if hidden_by_token is None:
        hidden_by_token = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"),
            torch.tensor([0.0, 1.0, 0.0, 0.0], device="cuda"),
            torch.tensor([0.0, 0.0, 1.0, 0.0], device="cuda"),
            torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
        )
    for offset, hidden in enumerate(hidden_by_token):
        deepseek_v4_flash_compressed_layer_forward(
            hidden,
            layer=layer,
            stager=stager,
            backend=backend,
            state=state,
            token_idx=token_offset + offset,
            router_top_k=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_compressor_state_emits_only_on_ratio_boundary_and_uses_gate_scores() -> None:
    store, layer = _compressed_layer_fixture()
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=16,
            hidden_size=4,
            kv_width=4,
            dtype=torch.float32,
            device="cuda",
        )
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    backend = _RecordingCompressedBackend()
    hidden_by_token = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"),
        torch.tensor([0.0, 1.0, 0.0, 0.0], device="cuda"),
        torch.tensor([0.0, 0.0, 1.0, 0.0], device="cuda"),
        torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
    )

    for token_idx, hidden in enumerate(hidden_by_token[:3]):
        output = deepseek_v4_flash_compressed_layer_forward(
            hidden,
            layer=layer,
            stager=stager,
            backend=backend,
            state=state,
            token_idx=token_idx,
            router_top_k=1,
        )
        assert output.device.type == "cuda"
        assert int(state.compressed_kv_cache._compressed_counts[2].item()) == 0

    output = deepseek_v4_flash_compressed_layer_forward(
        hidden_by_token[3],
        layer=layer,
        stager=stager,
        backend=backend,
        state=state,
        token_idx=3,
        router_top_k=1,
    )

    assert output.device.type == "cuda"
    assert state.token_position == 0
    assert int(state.compressed_kv_cache._compressed_counts[2].item()) == 1
    emitted = state.compressed_kv_cache.compressed_rows[2, 0]
    assert emitted[0] > emitted[3]
    assert emitted[0] > 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_ratio4_compressor_accepts_real_doubled_width_contract() -> None:
    store, layer = _compressed_layer_fixture(doubled_width=True)
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=16,
            hidden_size=4,
            kv_width=4,
            dtype=torch.float32,
            device="cuda",
        )
    )

    _run_four_ratio4_tokens(
        state=state,
        layer=layer,
        stager=DeepSeekV4FlashGPUWeightStager(store, device="cuda"),
        backend=_RecordingCompressedBackend(),
    )

    assert int(state.compressed_kv_cache._compressed_counts[2].item()) == 1
    assert state.compressed_kv_cache.compressed_rows[2, 0].shape == (4,)
    assert state.compressed_kv_cache.indexer_rows[2, 0].shape == (128,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_indexer_selection_skips_projection_when_under_top_k() -> None:
    store, layer = _compressed_layer_fixture(doubled_width=True)

    class FailingStager(DeepSeekV4FlashGPUWeightStager):
        def stage_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
            raise AssertionError(f"unexpected projection for {tensor.name}")

        def stage_vector(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
            raise AssertionError(f"unexpected vector staging for {tensor.name}")

    selected = _select_compressed_rows_with_indexer(
        torch.ones((4,), dtype=torch.float32, device="cuda"),
        layer=layer,
        stager=FailingStager(store, device="cuda"),
        indexer_rows=torch.ones((2, 128), dtype=torch.float32, device="cuda"),
        token_idx=3,
    )

    torch.testing.assert_close(
        selected,
        torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_compressed_layer_mhc_pre_post_returns_streams() -> None:
    store, layer = _compressed_layer_fixture(doubled_width=True)
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=16,
            hidden_size=4,
            kv_width=4,
            dtype=torch.float32,
            device="cuda",
        )
    )
    streams = torch.stack(
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"),
            torch.tensor([0.0, 1.0, 0.0, 0.0], device="cuda"),
        )
    )

    output = deepseek_v4_flash_compressed_layer_forward(
        streams,
        layer=layer,
        stager=DeepSeekV4FlashGPUWeightStager(store, device="cuda"),
        backend=_RecordingCompressedBackend(),
        state=state,
        token_idx=0,
        router_top_k=1,
    )

    assert output.device.type == "cuda"
    assert output.shape == streams.shape
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_ratio4_compressor_second_boundary_uses_carry_lane() -> None:
    store, layer = _compressed_layer_fixture(doubled_width=True)
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=16,
            hidden_size=4,
            kv_width=4,
            dtype=torch.float32,
            device="cuda",
        )
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    backend = _RecordingCompressedBackend()

    _run_four_ratio4_tokens(
        state=state,
        layer=layer,
        stager=stager,
        backend=backend,
    )
    first = state.compressed_kv_cache.compressed_rows[2, 0].detach().clone()
    _run_four_ratio4_tokens(
        state=state,
        layer=layer,
        stager=stager,
        backend=backend,
        token_offset=4,
        hidden_by_token=(
            torch.tensor([0.0, 1.0, 0.0, 0.0], device="cuda"),
            torch.tensor([0.0, 1.0, 0.0, 0.0], device="cuda"),
            torch.tensor([0.0, 1.0, 0.0, 0.0], device="cuda"),
            torch.tensor([0.0, 1.0, 0.0, 0.0], device="cuda"),
        ),
    )

    assert int(state.compressed_kv_cache._compressed_counts[2].item()) == 2
    torch.testing.assert_close(state.compressed_kv_cache.compressed_rows[2, 1], first)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_runtime_reset_clears_compressor_state() -> None:
    store, layer = _compressed_layer_fixture()
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=16,
            hidden_size=4,
            kv_width=4,
            dtype=torch.float32,
            device="cuda",
        )
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    backend = _RecordingCompressedBackend()
    for token_idx, hidden in enumerate(
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"),
            torch.tensor([0.0, 1.0, 0.0, 0.0], device="cuda"),
            torch.tensor([0.0, 0.0, 1.0, 0.0], device="cuda"),
        )
    ):
        deepseek_v4_flash_compressed_layer_forward(
            hidden,
            layer=layer,
            stager=stager,
            backend=backend,
            state=state,
            token_idx=token_idx,
            router_top_k=1,
        )

    state.reset()
    deepseek_v4_flash_compressed_layer_forward(
        torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
        layer=layer,
        stager=stager,
        backend=backend,
        state=state,
        token_idx=3,
        router_top_k=1,
    )

    assert state.token_position == 0
    assert int(state.compressed_kv_cache._compressed_counts[2].item()) == 0

    state.reset()
    _run_four_ratio4_tokens(
        state=state,
        layer=layer,
        stager=stager,
        backend=backend,
    )

    assert int(state.compressed_kv_cache._compressed_counts[2].item()) == 1
    emitted = state.compressed_kv_cache.compressed_rows[2, 0]
    assert emitted[0] > emitted[3]
    assert emitted[0] > 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_boundary_rows_are_written_and_selected_rows_are_bounded() -> None:
    store, layer = _compressed_layer_fixture()
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=16,
            hidden_size=4,
            kv_width=4,
            dtype=torch.float32,
            device="cuda",
        )
    )
    existing_row = torch.tensor([10.0, 0.0, 0.0, 0.0], device="cuda")
    existing_indexer = torch.zeros(128, device="cuda")
    existing_indexer[0] = 10.0
    state.compressed_kv_cache.append_compressed(
        layer_idx=2,
        token_idx=0,
        row=existing_row,
        indexer_row=existing_indexer,
    )
    backend = _RecordingCompressedBackend()

    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    for token_idx, hidden in enumerate(
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"),
            torch.tensor([0.0, 1.0, 0.0, 0.0], device="cuda"),
            torch.tensor([0.0, 0.0, 1.0, 0.0], device="cuda"),
            torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda"),
        )
    ):
        output = deepseek_v4_flash_compressed_layer_forward(
            hidden,
            layer=layer,
            stager=stager,
            backend=backend,
            state=state,
            token_idx=token_idx,
            router_top_k=1,
        )

    assert output.device.type == "cuda"
    assert state.token_position == 0
    assert int(state.compressed_kv_cache._compressed_counts[2].item()) == 2
    expected_row = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")
    expected_row = expected_row * torch.rsqrt(expected_row.pow(2).mean() + 1e-6)
    torch.testing.assert_close(
        state.compressed_kv_cache.compressed_rows[2, 1],
        expected_row,
        rtol=5e-4,
        atol=5e-4,
    )
    assert backend.compressed_selected_rows is not None
    assert backend.compressed_selected_rows.device.type == "cuda"
    assert torch.all(backend.compressed_selected_rows >= 0)
    assert torch.all(backend.compressed_selected_rows < 2)
    assert backend.compressed_rows_shape == (2, 4)
    assert backend.sliding_attention_calls == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_compressor_emitted_row_uses_boundary_rope_position() -> None:
    hidden_size = 64
    store = _FakeLayerStore()
    kv = DeepSeekV4FlashTensor(
        "compressor.kv",
        (hidden_size, hidden_size),
        GGML_TYPE_F32,
        0,
        0,
    )
    gate = DeepSeekV4FlashTensor(
        "compressor.gate", (hidden_size, hidden_size), GGML_TYPE_F32, 0, 0
    )
    ape = DeepSeekV4FlashTensor("compressor.ape", (hidden_size, 4), GGML_TYPE_F32, 0, 0)
    norm = DeepSeekV4FlashTensor("compressor.norm", (hidden_size,), GGML_TYPE_F32, 0, 0)
    compressor = DeepSeekV4FlashCompressorTensors(
        kv=kv,
        gate=gate,
        ape=ape,
        norm=norm,
    )
    store.matrices[compressor.kv.name] = torch.eye(hidden_size)
    store.matrices[compressor.gate.name] = torch.zeros(hidden_size, hidden_size)
    store.matrices[compressor.ape.name] = torch.zeros(hidden_size, 4)
    store.vectors[(compressor.norm.name, torch.float32)] = torch.ones(hidden_size)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=16,
            hidden_size=hidden_size,
            kv_width=hidden_size,
            dtype=torch.float32,
            device="cuda",
        )
    )
    hidden = torch.linspace(-1.0, 1.0, hidden_size, device="cuda")

    emitted = None
    for token_idx in range(8):
        _candidate, emitted = _update_compressor_state(
            hidden,
            state=state,
            layer_idx=2,
            state_name="attention",
            compressor=compressor,
            stager=stager,
            token_idx=token_idx,
            ratio=4,
        )

    assert emitted is not None
    unrotated = hidden * torch.rsqrt(hidden.pow(2).mean() + 1e-6)
    expected = apply_deepseek_layer_rope_to_tail_reference(
        unrotated,
        token_idx=4,
        layer_idx=2,
    )
    torch.testing.assert_close(emitted, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_layer_dispatch_only_falls_back_to_sliding_for_sliding_layers() -> None:
    store, compressed_layer = _compressed_layer_fixture()
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=16,
            hidden_size=4,
            kv_width=4,
            dtype=torch.float32,
            device="cuda",
        )
    )

    output = deepseek_v4_flash_layer_forward(
        torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"),
        layer=compressed_layer,
        stager=DeepSeekV4FlashGPUWeightStager(store, device="cuda"),
        backend=_RecordingCompressedBackend(),
        state=state,
        token_idx=3,
        router_top_k=1,
    )

    assert output.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.skipif(
    os.environ.get("RUN_DEEPSEEK_REAL_GGUF_LAYER") != "1",
    reason="real GGUF layer smoke is opt-in",
)
def test_real_gguf_layer2_compressed_forward_smoke() -> None:
    if not TARGET_GGUF.exists():
        pytest.fail(f"target GGUF not downloaded: {TARGET_GGUF}")
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        layer = store.bindings.layers[2]
        state = DeepSeekV4FlashGPURequestState(
            DeepSeekV4FlashGPUCacheConfig(
                context_length=4096,
                hidden_size=4096,
                dtype=torch.float32,
                device="cuda",
            )
        )
        hidden = torch.zeros(4096, dtype=torch.float32, device="cuda")
        hidden[0] = 1.0

        output = deepseek_v4_flash_compressed_layer_forward(
            hidden,
            layer=layer,
            stager=DeepSeekV4FlashGPUWeightStager(store, device="cuda"),
            backend=DeepSeekV4FlashGPUBackend(),
            state=state,
            token_idx=3,
        )

    assert output.device.type == "cuda"
    assert output.shape in (hidden.shape, (4, hidden.numel()))
    assert torch.isfinite(output).all()


def _make_real_compressed_layer() -> tuple[
    _FakeLayerStore,
    DeepSeekV4FlashLayerSemanticBindings,
]:
    """Build a compressed layer (layer 2) that uses real sliding attention tensors."""
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    head_dim = DEEPSEEK_V4_FLASH_SHAPE.head_dim
    num_heads = DEEPSEEK_V4_FLASH_SHAPE.num_attention_heads
    q_lora_rank = DEEPSEEK_V4_FLASH_SHAPE.q_lora_rank
    o_lora_rank = DEEPSEEK_V4_FLASH_SHAPE.o_lora_rank
    output_groups = DEEPSEEK_V4_FLASH_SHAPE.output_groups
    indexer_head_dim = DEEPSEEK_V4_FLASH_SHAPE.indexer_head_dim
    indexer_heads = DEEPSEEK_V4_FLASH_SHAPE.indexer_heads
    kv_width = head_dim
    indexer_width = indexer_head_dim
    num_experts = 3

    torch.manual_seed(42)
    store = _FakeLayerStore()
    tensors: dict[str, DeepSeekV4FlashTensor] = {
        "attn_norm": _f32_tensor("blk.2.attn_norm.weight", (hidden_size,)),
        "attn_q_a": _f32_tensor("blk.2.attn_q_a.weight", (q_lora_rank, hidden_size)),
        "attn_q_a_norm": _f32_tensor("blk.2.attn_q_a_norm.weight", (q_lora_rank,)),
        "attn_q_b": _f32_tensor(
            "blk.2.attn_q_b.weight",
            (num_heads * head_dim, q_lora_rank),
        ),
        "attn_kv": _f32_tensor("blk.2.attn_kv_a_mqa.weight", (head_dim, hidden_size)),
        "attn_kv_norm": _f32_tensor("blk.2.attn_kv_a_norm.weight", (head_dim,)),
        "attn_sinks": _f32_tensor("blk.2.attn_sinks.weight", (num_heads,)),
        "attn_out_a": _f32_tensor(
            "blk.2.attn_output_a.weight",
            (num_heads * head_dim // output_groups, o_lora_rank),
        ),
        "attn_out_b": _f32_tensor(
            "blk.2.attn_output_b.weight",
            (o_lora_rank, hidden_size),
        ),
        "comp_kv": _f32_tensor(
            "blk.2.attn_compressor_kv.weight",
            (hidden_size, kv_width),
        ),
        "comp_gate": _f32_tensor(
            "blk.2.attn_compressor_gate.weight",
            (hidden_size, kv_width),
        ),
        "comp_ape": _f32_tensor(
            "blk.2.attn_compressor_ape.weight",
            (kv_width, 4),
        ),
        "comp_norm": _f32_tensor("blk.2.attn_compressor_norm.weight", (kv_width,)),
        "index_q_b": _f32_tensor(
            "blk.2.indexer.attn_q_b.weight",
            (hidden_size, indexer_heads * indexer_head_dim),
        ),
        "index_proj": _f32_tensor(
            "blk.2.indexer.proj.weight",
            (hidden_size, indexer_heads),
        ),
        "index_comp_kv": _f32_tensor(
            "blk.2.indexer_compressor_kv.weight",
            (hidden_size, indexer_width),
        ),
        "index_comp_gate": _f32_tensor(
            "blk.2.indexer_compressor_gate.weight",
            (hidden_size, indexer_width),
        ),
        "index_comp_ape": _f32_tensor(
            "blk.2.indexer_compressor_ape.weight",
            (indexer_width, 4),
        ),
        "index_comp_norm": _f32_tensor(
            "blk.2.indexer_compressor_norm.weight",
            (indexer_width,),
        ),
        "ffn_norm": _f32_tensor("blk.2.ffn_norm.weight", (hidden_size,)),
        "router": _f32_tensor("blk.2.ffn_gate_inp.weight", (hidden_size, num_experts)),
        "gate": _f32_tensor(
            "blk.2.ffn_gate_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
        "up": _f32_tensor(
            "blk.2.ffn_up_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
        "down": _f32_tensor(
            "blk.2.ffn_down_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
    }

    store.vectors[(tensors["attn_norm"].name, torch.float32)] = torch.ones(hidden_size)
    store.vectors[(tensors["attn_q_a_norm"].name, torch.float32)] = torch.ones(
        q_lora_rank
    )
    store.vectors[(tensors["attn_kv_norm"].name, torch.float32)] = torch.ones(head_dim)
    store.vectors[(tensors["attn_sinks"].name, torch.float32)] = torch.zeros(num_heads)
    store.vectors[(tensors["comp_norm"].name, torch.float32)] = torch.ones(kv_width)
    store.vectors[(tensors["index_comp_norm"].name, torch.float32)] = torch.ones(
        indexer_width
    )
    store.vectors[(tensors["ffn_norm"].name, torch.float32)] = torch.ones(hidden_size)

    store.matrices[tensors["attn_q_a"].name] = (
        torch.randn(q_lora_rank, hidden_size, dtype=torch.float32) * 0.01
    )
    store.matrices[tensors["attn_q_b"].name] = (
        torch.randn(num_heads * head_dim, q_lora_rank, dtype=torch.float32) * 0.01
    )
    store.matrices[tensors["attn_kv"].name] = (
        torch.randn(head_dim, hidden_size, dtype=torch.float32) * 0.01
    )
    store.matrices[tensors["attn_out_a"].name] = (
        torch.randn(
            num_heads * head_dim // output_groups,
            o_lora_rank,
            dtype=torch.float32,
        )
        * 0.01
    )
    store.matrices[tensors["attn_out_b"].name] = (
        torch.randn(o_lora_rank, hidden_size, dtype=torch.float32) * 0.01
    )

    # Identity compressor/indexer so emitted rows are well-defined.
    store.matrices[tensors["comp_kv"].name] = torch.eye(kv_width, hidden_size)
    store.matrices[tensors["comp_gate"].name] = torch.zeros(hidden_size, kv_width)
    store.matrices[tensors["comp_ape"].name] = torch.zeros(kv_width, 4)
    store.matrices[tensors["index_q_b"].name] = torch.cat(
        [
            torch.eye(hidden_size),
            torch.zeros(hidden_size, indexer_heads * indexer_head_dim - hidden_size),
        ],
        dim=1,
    )
    store.matrices[tensors["index_proj"].name] = torch.ones(hidden_size, indexer_heads)
    store.matrices[tensors["index_comp_kv"].name] = torch.eye(
        indexer_width, hidden_size
    )
    store.matrices[tensors["index_comp_gate"].name] = store.matrices[
        tensors["index_comp_kv"].name
    ].clone()
    store.matrices[tensors["index_comp_ape"].name] = torch.zeros(indexer_width, 4)

    # Router that strongly selects expert 0.
    router = torch.full((num_experts, hidden_size), -10.0, dtype=torch.float32)
    router[0, :] = 10.0
    store.matrices[tensors["router"].name] = router

    for tensor_name in ("gate", "up", "down"):
        for expert_id in range(num_experts):
            store.expert_matrices[(tensors[tensor_name].name, expert_id)] = torch.eye(
                hidden_size, dtype=torch.float32
            )

    layer = DeepSeekV4FlashLayerSemanticBindings(
        layer_index=2,
        attention_norm=tensors["attn_norm"],
        attention_query_a=tensors["attn_q_a"],
        attention_query_a_norm=tensors["attn_q_a_norm"],
        attention_query_b=tensors["attn_q_b"],
        attention_key_value=tensors["attn_kv"],
        attention_key_value_a_norm=tensors["attn_kv_norm"],
        attention_sinks=tensors["attn_sinks"],
        attention_output_a=tensors["attn_out_a"],
        attention_output_b=tensors["attn_out_b"],
        attention_compressor=DeepSeekV4FlashCompressorTensors(
            ape=tensors["comp_ape"],
            gate=tensors["comp_gate"],
            kv=tensors["comp_kv"],
            norm=tensors["comp_norm"],
        ),
        indexer=DeepSeekV4FlashIndexerTensors(
            query_b=tensors["index_q_b"],
            projection=tensors["index_proj"],
            compressor=DeepSeekV4FlashCompressorTensors(
                ape=tensors["index_comp_ape"],
                gate=tensors["index_comp_gate"],
                kv=tensors["index_comp_kv"],
                norm=tensors["index_comp_norm"],
            ),
        ),
        ffn_norm=tensors["ffn_norm"],
        router=tensors["router"],
        grouped_experts=DeepSeekV4FlashGroupedExpertTensors(
            gate=tensors["gate"],
            up=tensors["up"],
            down=tensors["down"],
        ),
    )
    return store, layer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_select_compressed_rows_with_indexer_triton_matches_reference() -> None:
    device = torch.device("cuda")
    heads = 64
    row_width = 128
    n_rows = 1024

    index_query = torch.randn(heads, row_width, device=device, dtype=torch.float32)
    indexer_rows = torch.randn(n_rows, row_width, device=device, dtype=torch.float32)
    index_weights = torch.randn(heads, device=device, dtype=torch.float32)

    os.environ["FASTINFERENCE_DEEPSEEK_V4_FLASH_INDEXER_SELECT_FALLBACK"] = "1"
    # The function also needs layer/stager/state; construct minimal fakes or
    # recompute reference manually:
    per_head_scores = index_query.matmul(indexer_rows.T)
    per_head_scores = torch.clamp_min(per_head_scores, 0.0)
    scale = 1.0 / float(heads * row_width) ** 0.5
    expected_scores = (index_weights.reshape(heads, 1) * per_head_scores).sum(
        dim=0
    ) * scale

    os.environ["FASTINFERENCE_DEEPSEEK_V4_FLASH_INDEXER_SELECT_FALLBACK"] = "0"
    from vllm.kernels.triton.deepseek_v4_flash.compressed_indexer_select import (
        deepseek_v4_indexer_select_scores,
    )

    got_scores = deepseek_v4_indexer_select_scores(
        index_query, indexer_rows, index_weights
    )
    torch.testing.assert_close(got_scores, expected_scores, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_compressed_layer_forward_explicit_kv_rows_matches_state_read() -> None:
    store, layer = _make_real_compressed_layer()
    backend = _RecordingCompressedBackend()
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    hidden = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    layer_idx = layer.layer_index
    token_idx = 0

    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    def _make_state() -> DeepSeekV4FlashGPURequestState:
        return DeepSeekV4FlashGPURequestState(
            DeepSeekV4FlashGPUCacheConfig(
                context_length=128,
                hidden_size=hidden_size,
                dtype=torch.float32,
                device=hidden.device,
            )
        )

    state_a = _make_state()
    output_state_read = deepseek_v4_flash_compressed_layer_forward(
        hidden,
        layer=layer,
        stager=stager,
        backend=backend,
        state=state_a,
        token_idx=token_idx,
        router_top_k=1,
    )

    state_b = _make_state()
    # Seed the cache so raw_kv_window can materialize the current token's row.
    _ = deepseek_v4_flash_compressed_layer_forward(
        hidden,
        layer=layer,
        stager=stager,
        backend=backend,
        state=state_b,
        token_idx=token_idx,
        router_top_k=1,
    )
    explicit_kv_rows = state_b.raw_kv_window(
        layer_idx,
        token_idx,
        DEEPSEEK_V4_FLASH_SHAPE.sliding_window,
    )
    output_explicit_rows = deepseek_v4_flash_compressed_layer_forward(
        hidden,
        layer=layer,
        stager=stager,
        backend=backend,
        state=state_b,
        token_idx=token_idx,
        kv_rows=explicit_kv_rows,
        router_top_k=1,
    )

    assert output_state_read.device.type == "cuda"
    assert output_explicit_rows.device.type == "cuda"
    torch.testing.assert_close(output_state_read, output_explicit_rows)
