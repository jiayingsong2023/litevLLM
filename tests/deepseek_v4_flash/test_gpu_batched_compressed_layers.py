from __future__ import annotations

import dataclasses

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.config import (
    DEEPSEEK_V4_FLASH_SHAPE,
    layer_compress_ratio,
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
    deepseek_v4_flash_compressed_layer_forward,
    deepseek_v4_flash_compressed_layer_forward_batched,
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


class _RecordingCompressedBackend(DeepSeekV4FlashGPUBackend):
    """Backend that returns deterministic outputs for compressed attention and MoE."""

    def __init__(self) -> None:
        self.compressed_selected_rows: torch.Tensor | None = None
        self.compressed_rows_shape: tuple[int, ...] | None = None
        self.compressed_attention_calls = 0
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
        self.compressed_attention_calls += 1
        return query

    def fused_sliding_window_attention(
        self,
        *,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        attn_sinks: torch.Tensor | None,
        token_idx: int,
    ) -> torch.Tensor | None:
        return None

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


def _make_request_state(
    *,
    context_length: int = 128,
    dtype: torch.dtype = torch.float32,
) -> DeepSeekV4FlashGPURequestState:
    return DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=context_length,
            hidden_size=DEEPSEEK_V4_FLASH_SHAPE.hidden_size,
            dtype=dtype,
            device=torch.device("cuda"),
        )
    )


def _make_real_compressed_layer_binding(
    layer_index: int = 2,
    *,
    with_indexer: bool = True,
) -> tuple[DeepSeekV4FlashLayerSemanticBindings, _FakeLayerStore]:
    """Build a compressed layer binding that exercises the real attention path."""
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    head_dim = DEEPSEEK_V4_FLASH_SHAPE.head_dim
    num_heads = DEEPSEEK_V4_FLASH_SHAPE.num_attention_heads
    q_lora_rank = DEEPSEEK_V4_FLASH_SHAPE.q_lora_rank
    o_lora_rank = DEEPSEEK_V4_FLASH_SHAPE.o_lora_rank
    output_groups = DEEPSEEK_V4_FLASH_SHAPE.output_groups
    indexer_head_dim = DEEPSEEK_V4_FLASH_SHAPE.indexer_head_dim
    indexer_heads = DEEPSEEK_V4_FLASH_SHAPE.indexer_heads
    num_experts = 3

    ratio = layer_compress_ratio(layer_index)
    if ratio == 0:
        raise ValueError(f"layer {layer_index} is not a compressed layer")

    torch.manual_seed(42 + layer_index)
    store = _FakeLayerStore()
    tensors: dict[str, DeepSeekV4FlashTensor] = {
        "attn_norm": _f32_tensor(f"blk.{layer_index}.attn_norm.weight", (hidden_size,)),
        "attn_q_a": _f32_tensor(
            f"blk.{layer_index}.attn_q_a.weight",
            (q_lora_rank, hidden_size),
        ),
        "attn_q_a_norm": _f32_tensor(
            f"blk.{layer_index}.attn_q_a_norm.weight",
            (q_lora_rank,),
        ),
        "attn_q_b": _f32_tensor(
            f"blk.{layer_index}.attn_q_b.weight",
            (num_heads * head_dim, q_lora_rank),
        ),
        "attn_kv": _f32_tensor(
            f"blk.{layer_index}.attn_kv_a_mqa.weight",
            (head_dim, hidden_size),
        ),
        "attn_kv_norm": _f32_tensor(
            f"blk.{layer_index}.attn_kv_a_norm.weight",
            (head_dim,),
        ),
        "attn_sinks": _f32_tensor(
            f"blk.{layer_index}.attn_sinks.weight",
            (num_heads,),
        ),
        "attn_out_a": _f32_tensor(
            f"blk.{layer_index}.attn_output_a.weight",
            (num_heads * head_dim // output_groups, o_lora_rank),
        ),
        "attn_out_b": _f32_tensor(
            f"blk.{layer_index}.attn_output_b.weight",
            (o_lora_rank, hidden_size),
        ),
        "comp_kv": _f32_tensor(
            f"blk.{layer_index}.attn_compressor_kv.weight",
            (hidden_size, head_dim),
        ),
        "comp_gate": _f32_tensor(
            f"blk.{layer_index}.attn_compressor_gate.weight",
            (hidden_size, head_dim),
        ),
        "comp_ape": _f32_tensor(
            f"blk.{layer_index}.attn_compressor_ape.weight",
            (head_dim, ratio),
        ),
        "comp_norm": _f32_tensor(
            f"blk.{layer_index}.attn_compressor_norm.weight",
            (head_dim,),
        ),
        "ffn_norm": _f32_tensor(f"blk.{layer_index}.ffn_norm.weight", (hidden_size,)),
        "router": _f32_tensor(
            f"blk.{layer_index}.ffn_gate_inp.weight",
            (hidden_size, num_experts),
        ),
        "gate": _f32_tensor(
            f"blk.{layer_index}.ffn_gate_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
        "up": _f32_tensor(
            f"blk.{layer_index}.ffn_up_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
        "down": _f32_tensor(
            f"blk.{layer_index}.ffn_down_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
    }

    store.vectors[(tensors["attn_norm"].name, torch.float32)] = torch.ones(hidden_size)
    store.vectors[(tensors["attn_q_a_norm"].name, torch.float32)] = torch.ones(
        q_lora_rank
    )
    store.vectors[(tensors["attn_kv_norm"].name, torch.float32)] = torch.ones(head_dim)
    store.vectors[(tensors["attn_sinks"].name, torch.float32)] = torch.zeros(num_heads)
    store.vectors[(tensors["comp_norm"].name, torch.float32)] = torch.ones(head_dim)
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

    # Identity compressor so emitted rows are well-defined.
    store.matrices[tensors["comp_kv"].name] = torch.eye(head_dim, hidden_size)
    store.matrices[tensors["comp_gate"].name] = torch.zeros(hidden_size, head_dim)
    store.matrices[tensors["comp_ape"].name] = torch.zeros(head_dim, ratio)

    # Router that strongly selects expert 0.
    router = torch.full((num_experts, hidden_size), -10.0, dtype=torch.float32)
    router[0, :] = 10.0
    store.matrices[tensors["router"].name] = router

    for tensor_name in ("gate", "up", "down"):
        for expert_id in range(num_experts):
            store.expert_matrices[(tensors[tensor_name].name, expert_id)] = torch.eye(
                hidden_size, dtype=torch.float32
            )

    indexer: DeepSeekV4FlashIndexerTensors | None = None
    if with_indexer:
        tensors["index_q_b"] = _f32_tensor(
            f"blk.{layer_index}.indexer.attn_q_b.weight",
            (hidden_size, indexer_heads * indexer_head_dim),
        )
        tensors["index_proj"] = _f32_tensor(
            f"blk.{layer_index}.indexer.proj.weight",
            (hidden_size, indexer_heads),
        )
        tensors["index_comp_kv"] = _f32_tensor(
            f"blk.{layer_index}.indexer_compressor_kv.weight",
            (hidden_size, indexer_head_dim),
        )
        tensors["index_comp_gate"] = _f32_tensor(
            f"blk.{layer_index}.indexer_compressor_gate.weight",
            (hidden_size, indexer_head_dim),
        )
        tensors["index_comp_ape"] = _f32_tensor(
            f"blk.{layer_index}.indexer_compressor_ape.weight",
            (indexer_head_dim, 4),
        )
        tensors["index_comp_norm"] = _f32_tensor(
            f"blk.{layer_index}.indexer_compressor_norm.weight",
            (indexer_head_dim,),
        )
        store.matrices[tensors["index_q_b"].name] = torch.cat(
            [
                torch.eye(hidden_size),
                torch.zeros(
                    hidden_size,
                    indexer_heads * indexer_head_dim - hidden_size,
                ),
            ],
            dim=1,
        )
        store.matrices[tensors["index_proj"].name] = torch.ones(
            hidden_size, indexer_heads
        )
        store.matrices[tensors["index_comp_kv"].name] = torch.eye(
            indexer_head_dim, hidden_size
        )
        store.matrices[tensors["index_comp_gate"].name] = store.matrices[
            tensors["index_comp_kv"].name
        ].clone()
        store.matrices[tensors["index_comp_ape"].name] = torch.zeros(
            indexer_head_dim, 4
        )
        store.vectors[(tensors["index_comp_norm"].name, torch.float32)] = torch.ones(
            indexer_head_dim
        )
        indexer = DeepSeekV4FlashIndexerTensors(
            query_b=tensors["index_q_b"],
            projection=tensors["index_proj"],
            compressor=DeepSeekV4FlashCompressorTensors(
                ape=tensors["index_comp_ape"],
                gate=tensors["index_comp_gate"],
                kv=tensors["index_comp_kv"],
                norm=tensors["index_comp_norm"],
            ),
        )

    layer = DeepSeekV4FlashLayerSemanticBindings(
        layer_index=layer_index,
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
        indexer=indexer,
        ffn_norm=tensors["ffn_norm"],
        router=tensors["router"],
        grouped_experts=DeepSeekV4FlashGroupedExpertTensors(
            gate=tensors["gate"],
            up=tensors["up"],
            down=tensors["down"],
        ),
    )
    return layer, store


def _make_small_compressed_layer_binding(
    layer_index: int = 2,
    *,
    with_indexer: bool = True,
) -> tuple[DeepSeekV4FlashLayerSemanticBindings, _FakeLayerStore]:
    """Build a small compressed layer for fast validation/hash-routing tests."""
    hidden_size = 4
    kv_width = 4
    attention_width = kv_width
    indexer_width = 128
    num_experts = 3

    ratio = layer_compress_ratio(layer_index)
    if ratio == 0:
        raise ValueError(f"layer {layer_index} is not a compressed layer")

    torch.manual_seed(42 + layer_index)
    store = _FakeLayerStore()
    tensors: dict[str, DeepSeekV4FlashTensor] = {
        "attn_norm": _tensor(f"blk.{layer_index}.attn_norm.weight", (hidden_size,)),
        "attn_q": _tensor(
            f"blk.{layer_index}.attn_q.weight",
            (hidden_size, attention_width),
        ),
        "attn_out": _tensor(
            f"blk.{layer_index}.attn_output.weight",
            (attention_width, hidden_size),
        ),
        "comp_kv": _tensor(
            f"blk.{layer_index}.attn_compressor_kv.weight",
            (hidden_size, kv_width),
        ),
        "comp_gate": _tensor(
            f"blk.{layer_index}.attn_compressor_gate.weight",
            (hidden_size, kv_width),
        ),
        "comp_ape": _tensor(
            f"blk.{layer_index}.attn_compressor_ape.weight",
            (kv_width, ratio),
        ),
        "comp_norm": _tensor(
            f"blk.{layer_index}.attn_compressor_norm.weight",
            (kv_width,),
        ),
        "ffn_norm": _tensor(f"blk.{layer_index}.ffn_norm.weight", (hidden_size,)),
        "router": _tensor(
            f"blk.{layer_index}.ffn_gate_inp.weight",
            (hidden_size, num_experts),
        ),
        "gate": _tensor(
            f"blk.{layer_index}.ffn_gate_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
        "up": _tensor(
            f"blk.{layer_index}.ffn_up_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
        "down": _tensor(
            f"blk.{layer_index}.ffn_down_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
    }

    for name in ("attn_norm", "comp_norm", "ffn_norm"):
        width = kv_width if name == "comp_norm" else hidden_size
        store.vectors[(tensors[name].name, torch.float32)] = torch.ones(width)

    attention_query = torch.zeros((attention_width, hidden_size))
    attention_query[:hidden_size] = torch.eye(hidden_size)
    store.matrices[tensors["attn_q"].name] = attention_query
    store.matrices[tensors["attn_out"].name] = torch.eye(hidden_size)

    comp_identity = torch.eye(kv_width, hidden_size)
    store.matrices[tensors["comp_kv"].name] = comp_identity
    comp_gate = torch.zeros((kv_width, hidden_size), dtype=torch.float32)
    comp_gate[:, 0] = 10.0
    store.matrices[tensors["comp_gate"].name] = comp_gate
    store.matrices[tensors["comp_ape"].name] = torch.zeros(kv_width, ratio)

    router = torch.full((num_experts, hidden_size), -10.0, dtype=torch.float32)
    router[0, :] = 10.0
    store.matrices[tensors["router"].name] = router

    for tensor_name in ("gate", "up", "down"):
        for expert_id in range(num_experts):
            store.expert_matrices[(tensors[tensor_name].name, expert_id)] = torch.eye(
                hidden_size, dtype=torch.float32
            )

    indexer: DeepSeekV4FlashIndexerTensors | None = None
    if with_indexer:
        tensors["index_q_b"] = _tensor(
            f"blk.{layer_index}.indexer.attn_q_b.weight",
            (hidden_size, indexer_width),
        )
        tensors["index_proj"] = _tensor(
            f"blk.{layer_index}.indexer.proj.weight",
            (hidden_size, 1),
        )
        tensors["index_comp_kv"] = _tensor(
            f"blk.{layer_index}.indexer_compressor_kv.weight",
            (hidden_size, indexer_width),
        )
        tensors["index_comp_gate"] = _tensor(
            f"blk.{layer_index}.indexer_compressor_gate.weight",
            (hidden_size, indexer_width),
        )
        tensors["index_comp_ape"] = _tensor(
            f"blk.{layer_index}.indexer_compressor_ape.weight",
            (indexer_width, 4),
        )
        tensors["index_comp_norm"] = _tensor(
            f"blk.{layer_index}.indexer_compressor_norm.weight",
            (indexer_width,),
        )
        store.vectors[(tensors["index_comp_norm"].name, torch.float32)] = torch.ones(
            indexer_width
        )
        store.matrices[tensors["index_q_b"].name] = torch.cat(
            [
                torch.eye(hidden_size),
                torch.zeros(hidden_size, indexer_width - hidden_size),
            ],
            dim=1,
        )
        store.matrices[tensors["index_proj"].name] = torch.ones((hidden_size, 1))
        store.matrices[tensors["index_comp_kv"].name] = torch.eye(
            indexer_width, hidden_size
        )
        store.matrices[tensors["index_comp_gate"].name] = store.matrices[
            tensors["index_comp_kv"].name
        ].clone()
        store.matrices[tensors["index_comp_ape"].name] = torch.zeros(indexer_width, 4)
        indexer = DeepSeekV4FlashIndexerTensors(
            query_b=tensors["index_q_b"],
            projection=tensors["index_proj"],
            compressor=DeepSeekV4FlashCompressorTensors(
                ape=tensors["index_comp_ape"],
                gate=tensors["index_comp_gate"],
                kv=tensors["index_comp_kv"],
                norm=tensors["index_comp_norm"],
            ),
        )

    layer = DeepSeekV4FlashLayerSemanticBindings(
        layer_index=layer_index,
        attention_norm=tensors["attn_norm"],
        attention_query=tensors["attn_q"],
        attention_output=tensors["attn_out"],
        attention_compressor=DeepSeekV4FlashCompressorTensors(
            ape=tensors["comp_ape"],
            gate=tensors["comp_gate"],
            kv=tensors["comp_kv"],
            norm=tensors["comp_norm"],
        ),
        indexer=indexer,
        ffn_norm=tensors["ffn_norm"],
        router=tensors["router"],
        grouped_experts=DeepSeekV4FlashGroupedExpertTensors(
            gate=tensors["gate"],
            up=tensors["up"],
            down=tensors["down"],
        ),
    )
    return layer, store


def _make_hash_routed_compressed_layer_binding() -> tuple[
    DeepSeekV4FlashLayerSemanticBindings,
    _FakeLayerStore,
]:
    """Return a small compressed layer augmented with hash-routed experts."""
    layer, store = _make_small_compressed_layer_binding(layer_index=2)
    vocab_size = 128
    hash_tensor = _tensor(
        f"blk.{layer.layer_index}.expert_token_to_expert_ids.weight",
        (DEEPSEEK_V4_FLASH_SHAPE.num_experts_per_tok, vocab_size),
    )
    store.vectors[(hash_tensor.name, torch.int32)] = torch.zeros(
        DEEPSEEK_V4_FLASH_SHAPE.num_experts_per_tok,
        vocab_size,
        dtype=torch.int32,
    )
    return dataclasses.replace(layer, expert_token_to_expert_ids=hash_tensor), store


def _make_hyper_connection_compressed_layer_binding() -> tuple[
    DeepSeekV4FlashLayerSemanticBindings,
    _FakeLayerStore,
]:
    """Return a small compressed layer with attention hyper-connection tensors."""
    layer, store = _make_small_compressed_layer_binding(layer_index=2)
    hidden_size = 4
    hc_tensors = {
        "hc_attn_fn": _tensor(
            f"blk.{layer.layer_index}.hc_attn_fn.weight",
            (hidden_size * 2, 8),
        ),
        "hc_attn_base": _tensor(
            f"blk.{layer.layer_index}.hc_attn_base.weight",
            (8,),
        ),
        "hc_attn_scale": _tensor(
            f"blk.{layer.layer_index}.hc_attn_scale.weight",
            (3,),
        ),
    }
    store.matrices[hc_tensors["hc_attn_fn"].name] = torch.zeros((hidden_size * 2, 8))
    store.vectors[(hc_tensors["hc_attn_base"].name, torch.float32)] = torch.zeros(8)
    store.vectors[(hc_tensors["hc_attn_scale"].name, torch.float32)] = torch.ones(3)
    return dataclasses.replace(
        layer,
        attention_hyper_connection=DeepSeekV4FlashHyperConnectionTensors(
            fn=hc_tensors["hc_attn_fn"],
            base=hc_tensors["hc_attn_base"],
            scale=hc_tensors["hc_attn_scale"],
        ),
    ), store


def _add_hyper_connections_to_compressed_layer(
    store: _FakeLayerStore,
    layer: DeepSeekV4FlashLayerSemanticBindings,
    *,
    hc_mult: int = 2,
) -> DeepSeekV4FlashLayerSemanticBindings:
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    mix_count = 2 * hc_mult + hc_mult * hc_mult
    flat_size = hc_mult * hidden_size
    layer_index = layer.layer_index
    fn_t = _f32_tensor(
        f"blk.{layer_index}.hc_attn_fn.weight",
        (flat_size, mix_count),
    )
    base_t = _f32_tensor(
        f"blk.{layer_index}.hc_attn_base.weight",
        (mix_count,),
    )
    scale_t = _f32_tensor(
        f"blk.{layer_index}.hc_attn_scale.weight",
        (3,),
    )
    ffn_fn_t = _f32_tensor(
        f"blk.{layer_index}.hc_ffn_fn.weight",
        (flat_size, mix_count),
    )
    ffn_base_t = _f32_tensor(
        f"blk.{layer_index}.hc_ffn_base.weight",
        (mix_count,),
    )
    ffn_scale_t = _f32_tensor(
        f"blk.{layer_index}.hc_ffn_scale.weight",
        (3,),
    )

    torch.manual_seed(44 + layer_index)
    store.matrices[fn_t.name] = (
        torch.randn(flat_size, mix_count, dtype=torch.float32, device="cuda") * 0.01
    )
    store.vectors[(base_t.name, torch.float32)] = torch.randn(
        mix_count, dtype=torch.float32, device="cuda"
    )
    store.vectors[(scale_t.name, torch.float32)] = torch.randn(
        3, dtype=torch.float32, device="cuda"
    )
    store.matrices[ffn_fn_t.name] = (
        torch.randn(flat_size, mix_count, dtype=torch.float32, device="cuda") * 0.01
    )
    store.vectors[(ffn_base_t.name, torch.float32)] = torch.randn(
        mix_count, dtype=torch.float32, device="cuda"
    )
    store.vectors[(ffn_scale_t.name, torch.float32)] = torch.randn(
        3, dtype=torch.float32, device="cuda"
    )

    return dataclasses.replace(
        layer,
        attention_hyper_connection=DeepSeekV4FlashHyperConnectionTensors(
            fn=fn_t, base=base_t, scale=scale_t
        ),
        ffn_hyper_connection=DeepSeekV4FlashHyperConnectionTensors(
            fn=ffn_fn_t, base=ffn_base_t, scale=ffn_scale_t
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_compressed_layer_matches_single_slot_real_attention() -> None:
    layer, store = _make_real_compressed_layer_binding(layer_index=2)
    backend = _RecordingCompressedBackend()
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden_a = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    hidden_b = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    hidden_batched = torch.stack([hidden_a, hidden_b])

    state_a = _make_request_state()
    state_b = _make_request_state()

    output_a = deepseek_v4_flash_compressed_layer_forward(
        hidden_a,
        layer=layer,
        stager=stager,
        backend=backend,
        state=state_a,
        token_idx=3,
        router_top_k=1,
    )
    output_b = deepseek_v4_flash_compressed_layer_forward(
        hidden_b,
        layer=layer,
        stager=stager,
        backend=backend,
        state=state_b,
        token_idx=3,
        router_top_k=1,
    )

    output_batched = deepseek_v4_flash_compressed_layer_forward_batched(
        hidden_batched,
        layer=layer,
        stager=stager,
        backend=backend,
        states=[state_a, state_b],
        token_indices=[3, 3],
        router_top_k=1,
    )

    assert output_batched.shape == (2, hidden_size)
    assert output_batched.device.type == "cuda"
    torch.testing.assert_close(output_batched[0], output_a)
    torch.testing.assert_close(output_batched[1], output_b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_compressed_layer_matches_single_slot_ratio128() -> None:
    layer, store = _make_real_compressed_layer_binding(
        layer_index=3,
        with_indexer=False,
    )
    backend = _RecordingCompressedBackend()
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden_a = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    hidden_b = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    hidden_batched = torch.stack([hidden_a, hidden_b])

    state_a = _make_request_state()
    state_b = _make_request_state()

    output_a = deepseek_v4_flash_compressed_layer_forward(
        hidden_a,
        layer=layer,
        stager=stager,
        backend=backend,
        state=state_a,
        token_idx=127,
        router_top_k=1,
    )
    output_b = deepseek_v4_flash_compressed_layer_forward(
        hidden_b,
        layer=layer,
        stager=stager,
        backend=backend,
        state=state_b,
        token_idx=127,
        router_top_k=1,
    )

    output_batched = deepseek_v4_flash_compressed_layer_forward_batched(
        hidden_batched,
        layer=layer,
        stager=stager,
        backend=backend,
        states=[state_a, state_b],
        token_indices=[127, 127],
        router_top_k=1,
    )

    assert output_batched.shape == (2, hidden_size)
    assert output_batched.device.type == "cuda"
    torch.testing.assert_close(output_batched[0], output_a)
    torch.testing.assert_close(output_batched[1], output_b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_compressed_layer_rejects_mismatched_batch() -> None:
    layer, store = _make_small_compressed_layer_binding(layer_index=2)
    backend = _RecordingCompressedBackend()
    hidden_size = 4
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden = torch.randn(2, hidden_size, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="states and token_indices must match"):
        deepseek_v4_flash_compressed_layer_forward_batched(
            hidden,
            layer=layer,
            stager=stager,
            backend=backend,
            states=[_make_request_state()],
            token_indices=[0, 0],
            router_top_k=1,
        )

    with pytest.raises(
        ValueError,
        match="batched compressed layer expects 2-D hidden",
    ):
        deepseek_v4_flash_compressed_layer_forward_batched(
            hidden[0],
            layer=layer,
            stager=stager,
            backend=backend,
            states=[_make_request_state()],
            token_indices=[0],
            router_top_k=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_compressed_layer_rejects_bad_token_id_tensors_shape() -> None:
    layer, store = _make_small_compressed_layer_binding(layer_index=2)
    backend = _RecordingCompressedBackend()
    hidden_size = 4
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden = torch.randn(2, hidden_size, dtype=torch.float32, device="cuda")
    bad_token_ids = torch.tensor([0, 1, 2], dtype=torch.long, device="cuda")

    with pytest.raises(ValueError, match="token_id_tensors must have shape"):
        deepseek_v4_flash_compressed_layer_forward_batched(
            hidden,
            layer=layer,
            stager=stager,
            backend=backend,
            states=[_make_request_state(), _make_request_state()],
            token_indices=[0, 0],
            token_id_tensors=bad_token_ids,
            router_top_k=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_compressed_layer_hash_routing_requires_token_ids() -> None:
    layer, store = _make_hash_routed_compressed_layer_binding()
    backend = _RecordingCompressedBackend()
    hidden_size = 4
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden = torch.randn(2, hidden_size, dtype=torch.float32, device="cuda")

    with pytest.raises(
        ValueError,
        match="hash-routed compressed layer requires token_id_tensors or token_ids",
    ):
        deepseek_v4_flash_compressed_layer_forward_batched(
            hidden,
            layer=layer,
            stager=stager,
            backend=backend,
            states=[_make_request_state(), _make_request_state()],
            token_indices=[0, 0],
            router_top_k=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_compressed_layer_hash_routing_uses_token_ids() -> None:
    layer, store = _make_hash_routed_compressed_layer_binding()
    backend = _RecordingCompressedBackend()
    hidden_size = 4
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden = torch.randn(2, hidden_size, dtype=torch.float32, device="cuda")

    output = deepseek_v4_flash_compressed_layer_forward_batched(
        hidden,
        layer=layer,
        stager=stager,
        backend=backend,
        states=[_make_request_state(), _make_request_state()],
        token_indices=[0, 0],
        token_ids=[0, 1],
        router_top_k=1,
    )

    assert output.shape == (2, hidden_size)
    assert output.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_compressed_layer_with_hyper_connections_matches_single_slot() -> None:
    layer, store = _make_real_compressed_layer_binding(layer_index=2)
    layer = _add_hyper_connections_to_compressed_layer(store, layer, hc_mult=2)
    backend = _RecordingCompressedBackend()
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden_a = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    hidden_b = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    hidden_batched = torch.stack([hidden_a, hidden_b])

    output_a = deepseek_v4_flash_compressed_layer_forward(
        hidden_a,
        layer=layer,
        stager=stager,
        backend=backend,
        state=_make_request_state(),
        token_idx=0,
        router_top_k=1,
    )
    output_b = deepseek_v4_flash_compressed_layer_forward(
        hidden_b,
        layer=layer,
        stager=stager,
        backend=backend,
        state=_make_request_state(),
        token_idx=0,
        router_top_k=1,
    )

    output_batched = deepseek_v4_flash_compressed_layer_forward_batched(
        hidden_batched,
        layer=layer,
        stager=stager,
        backend=backend,
        states=[_make_request_state(), _make_request_state()],
        token_indices=[0, 0],
        router_top_k=1,
    )

    assert output_batched.shape == (2, 2, hidden_size)
    assert output_batched.device.type == "cuda"
    torch.testing.assert_close(output_batched[0], output_a)
    torch.testing.assert_close(output_batched[1], output_b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_compressed_layer_accepts_attention_hyper_connection() -> None:
    layer, store = _make_hyper_connection_compressed_layer_binding()
    backend = _RecordingCompressedBackend()
    hidden_size = 4
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden = torch.randn(2, hidden_size, dtype=torch.float32, device="cuda")

    output = deepseek_v4_flash_compressed_layer_forward_batched(
        hidden,
        layer=layer,
        stager=stager,
        backend=backend,
        states=[_make_request_state(), _make_request_state()],
        token_indices=[0, 0],
        router_top_k=1,
    )

    assert output.shape == (2, 2, hidden_size)
    assert output.device.type == "cuda"
