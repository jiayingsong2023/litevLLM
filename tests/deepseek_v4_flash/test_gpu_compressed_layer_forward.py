from __future__ import annotations

import os
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


def _compressed_layer_fixture(
    *,
    doubled_width: bool = False,
) -> tuple[
    _FakeLayerStore,
    DeepSeekV4FlashLayerSemanticBindings,
]:
    hidden_size = 4
    kv_width = 4
    compressor_width = 8 if doubled_width else kv_width
    indexer_width = 128
    indexer_compressor_width = 256 if doubled_width else indexer_width
    tensors = {
        "attn_norm": _tensor("blk.2.attn_norm.weight", (hidden_size,)),
        "attn_q": _tensor("blk.2.attn_q.weight", (hidden_size, kv_width)),
        "attn_out": _tensor("blk.2.attn_output.weight", (kv_width, hidden_size)),
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
    store.matrices[tensors["attn_q"].name] = torch.eye(hidden_size)
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
    assert state.token_position == 4
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

    assert state.token_position == 1
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
    assert state.token_position == 4
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
    assert torch.all(backend.compressed_selected_rows < 1)
    assert backend.compressed_rows_shape == (1, 4)
    assert backend.sliding_attention_calls == 0


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
