from __future__ import annotations

import dataclasses

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.config import (
    DEEPSEEK_V4_FLASH_SHAPE,
)
from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_F16,
    DeepSeekV4FlashTensor,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_layers import (
    deepseek_v4_flash_sliding_layer_forward,
    deepseek_v4_flash_sliding_layer_forward_batched,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_runtime import (
    DeepSeekV4FlashGPUCacheConfig,
    DeepSeekV4FlashGPURequestState,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_weight_staging import (
    DeepSeekV4FlashGPUWeightStager,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashGroupedExpertTensors,
    DeepSeekV4FlashLayerSemanticBindings,
)


class _FakeLayerStore:
    def __init__(self) -> None:
        self.matrices: dict[str, torch.Tensor] = {}
        self.vectors: dict[tuple[str, torch.dtype], torch.Tensor] = {}

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
        return self.matrices[tensor.name][..., expert_id].clone()

    def stage_grouped_expert(
        self,
        tensors: DeepSeekV4FlashGroupedExpertTensors,
        expert_id: int,
        *,
        layer_idx: int | None = None,
    ) -> object:
        del tensors, expert_id, layer_idx
        hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
        eye = torch.eye(hidden_size, device="cuda", dtype=torch.float32)
        return type(
            "_StagedExpert",
            (),
            {"gate": eye, "up": eye, "down": eye},
        )()


class _NullAttentionBackend(DeepSeekV4FlashGPUBackend):
    """Backend that forces reference attention and zero expert outputs."""

    def fused_sliding_window_attention(
        self,
        *,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        attn_sinks: torch.Tensor | None,
        token_idx: int,
    ) -> torch.Tensor | None:
        return None

    def sliding_attention(
        self,
        *,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        attn_sinks: torch.Tensor | None,
        token_idx: int,
    ) -> torch.Tensor:
        return torch.zeros_like(query, dtype=torch.float32)

    def routed_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(hidden, dtype=torch.float32)


def _tensor(
    name: str,
    dims: tuple[int, ...],
    tensor_type: int = GGML_TYPE_F16,
) -> DeepSeekV4FlashTensor:
    return DeepSeekV4FlashTensor(
        name=name,
        dims=dims,
        tensor_type=tensor_type,
        offset=0,
        nbytes=0,
    )


def _make_real_sliding_layer() -> tuple[
    DeepSeekV4FlashLayerSemanticBindings,
    _FakeLayerStore,
]:
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    q_lora_rank = 8
    o_lora_rank = 8
    head_dim = DEEPSEEK_V4_FLASH_SHAPE.head_dim
    num_heads = DEEPSEEK_V4_FLASH_SHAPE.num_attention_heads
    num_experts = 3

    torch.manual_seed(42)
    store = _FakeLayerStore()
    tensors: dict[str, DeepSeekV4FlashTensor] = {
        "attn_norm": _tensor("blk.0.attn_norm.weight", (hidden_size,)),
        "attn_q_a": _tensor("blk.0.attn_q_a.weight", (q_lora_rank, hidden_size)),
        "attn_q_a_norm": _tensor("blk.0.attn_q_a_norm.weight", (q_lora_rank,)),
        "attn_q_b": _tensor(
            "blk.0.attn_q_b.weight",
            (num_heads * head_dim, q_lora_rank),
        ),
        "attn_kv": _tensor("blk.0.attn_kv_a_mqa.weight", (head_dim, hidden_size)),
        "attn_kv_norm": _tensor("blk.0.attn_kv_a_norm.weight", (head_dim,)),
        "attn_sinks": _tensor(
            "blk.0.attn_sinks.weight",
            (num_heads,),
        ),
        "attn_out_a": _tensor(
            "blk.0.attn_output_a.weight",
            (
                num_heads * head_dim // DEEPSEEK_V4_FLASH_SHAPE.output_groups,
                o_lora_rank,
            ),
        ),
        "attn_out_b": _tensor(
            "blk.0.attn_output_b.weight",
            (o_lora_rank, hidden_size),
        ),
        "ffn_norm": _tensor("blk.0.ffn_norm.weight", (hidden_size,)),
        "router": _tensor("blk.0.ffn_gate_inp.weight", (num_experts, hidden_size)),
        "gate": _tensor(
            "blk.0.ffn_gate_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
        "up": _tensor(
            "blk.0.ffn_up_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
        "down": _tensor(
            "blk.0.ffn_down_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
    }

    store.vectors[(tensors["attn_norm"].name, torch.float32)] = torch.ones(hidden_size)
    store.vectors[(tensors["attn_q_a_norm"].name, torch.float32)] = torch.ones(
        q_lora_rank
    )
    store.vectors[(tensors["attn_kv_norm"].name, torch.float32)] = torch.ones(head_dim)
    store.vectors[(tensors["attn_sinks"].name, torch.float32)] = torch.zeros(num_heads)
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
            num_heads * head_dim // DEEPSEEK_V4_FLASH_SHAPE.output_groups,
            o_lora_rank,
            dtype=torch.float32,
        )
        * 0.01
    )
    store.matrices[tensors["attn_out_b"].name] = (
        torch.randn(o_lora_rank, hidden_size, dtype=torch.float32) * 0.01
    )

    # Router that strongly selects expert 0.
    router = torch.full((num_experts, hidden_size), -10.0, dtype=torch.float32)
    router[0, :] = 10.0
    store.matrices[tensors["router"].name] = router

    for tensor_name in ("gate", "up", "down"):
        store.matrices[tensors[tensor_name].name] = torch.stack(
            [torch.eye(hidden_size, dtype=torch.float32) for _ in range(num_experts)],
            dim=-1,
        )

    layer = DeepSeekV4FlashLayerSemanticBindings(
        layer_index=0,
        attention_norm=tensors["attn_norm"],
        attention_query_a=tensors["attn_q_a"],
        attention_query_a_norm=tensors["attn_q_a_norm"],
        attention_query_b=tensors["attn_q_b"],
        attention_key_value=tensors["attn_kv"],
        attention_key_value_a_norm=tensors["attn_kv_norm"],
        attention_sinks=tensors["attn_sinks"],
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
    return layer, store


def _make_fake_sliding_layer() -> tuple[
    DeepSeekV4FlashLayerSemanticBindings,
    _FakeLayerStore,
]:
    """Create a sliding layer that uses the non-real attention fallback path."""
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    head_dim = DEEPSEEK_V4_FLASH_SHAPE.head_dim
    num_heads = DEEPSEEK_V4_FLASH_SHAPE.num_attention_heads
    num_experts = 3

    torch.manual_seed(43)
    store = _FakeLayerStore()
    tensors: dict[str, DeepSeekV4FlashTensor] = {
        "attn_norm": _tensor("blk.0.attn_norm.weight", (hidden_size,)),
        "attn_q": _tensor("blk.0.attn_q.weight", (hidden_size, num_heads * head_dim)),
        "attn_out": _tensor(
            "blk.0.attn_output.weight", (num_heads * head_dim, hidden_size)
        ),
        "ffn_norm": _tensor("blk.0.ffn_norm.weight", (hidden_size,)),
        "router": _tensor("blk.0.ffn_gate_inp.weight", (num_experts, hidden_size)),
        "gate": _tensor(
            "blk.0.ffn_gate_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
        "up": _tensor(
            "blk.0.ffn_up_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
        "down": _tensor(
            "blk.0.ffn_down_exps.weight",
            (hidden_size, hidden_size, num_experts),
        ),
    }

    store.vectors[(tensors["attn_norm"].name, torch.float32)] = torch.ones(hidden_size)
    store.vectors[(tensors["ffn_norm"].name, torch.float32)] = torch.ones(hidden_size)

    store.matrices[tensors["attn_q"].name] = (
        torch.randn(hidden_size, num_heads * head_dim, dtype=torch.float32) * 0.01
    )
    store.matrices[tensors["attn_out"].name] = (
        torch.randn(num_heads * head_dim, hidden_size, dtype=torch.float32) * 0.01
    )

    # Router that strongly selects expert 0.
    router = torch.full((num_experts, hidden_size), -10.0, dtype=torch.float32)
    router[0, :] = 10.0
    store.matrices[tensors["router"].name] = router

    for tensor_name in ("gate", "up", "down"):
        store.matrices[tensors[tensor_name].name] = torch.stack(
            [torch.eye(hidden_size, dtype=torch.float32) for _ in range(num_experts)],
            dim=-1,
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
    return layer, store


def _make_state() -> DeepSeekV4FlashGPURequestState:
    return DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=128,
            hidden_size=DEEPSEEK_V4_FLASH_SHAPE.hidden_size,
            device=torch.device("cuda"),
        )
    )


def _make_hash_routed_sliding_layer() -> tuple[
    DeepSeekV4FlashLayerSemanticBindings,
    _FakeLayerStore,
]:
    """Return a real sliding layer augmented with hash-routed experts."""
    layer, store = _make_real_sliding_layer()
    vocab_size = 128
    hash_tensor = _tensor(
        "blk.0.expert_token_to_expert_ids.weight",
        (DEEPSEEK_V4_FLASH_SHAPE.num_experts_per_tok, vocab_size),
    )
    # Every token maps to expert 0 for a deterministic, valid routing.
    store.vectors[(hash_tensor.name, torch.int32)] = torch.zeros(
        DEEPSEEK_V4_FLASH_SHAPE.num_experts_per_tok,
        vocab_size,
        dtype=torch.int32,
    )
    return dataclasses.replace(layer, expert_token_to_expert_ids=hash_tensor), store


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_sliding_layer_matches_single_slot() -> None:
    layer, store = _make_real_sliding_layer()
    backend = _NullAttentionBackend()
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden_a = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    hidden_b = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    hidden_batched = torch.stack([hidden_a, hidden_b])

    token_idx_a = 0
    token_idx_b = 0

    output_a = deepseek_v4_flash_sliding_layer_forward(
        hidden_a,
        layer=layer,
        stager=stager,
        backend=backend,
        state=_make_state(),
        token_idx=token_idx_a,
        router_top_k=1,
    )
    output_b = deepseek_v4_flash_sliding_layer_forward(
        hidden_b,
        layer=layer,
        stager=stager,
        backend=backend,
        state=_make_state(),
        token_idx=token_idx_b,
        router_top_k=1,
    )

    output_batched = deepseek_v4_flash_sliding_layer_forward_batched(
        hidden_batched,
        layer=layer,
        stager=stager,
        backend=backend,
        states=[_make_state(), _make_state()],
        token_indices=[token_idx_a, token_idx_b],
        router_top_k=1,
    )

    assert output_batched.shape == (2, hidden_size)
    assert output_batched.device.type == "cuda"
    torch.testing.assert_close(output_batched[0], output_a)
    torch.testing.assert_close(output_batched[1], output_b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_sliding_layer_rejects_mismatched_batch() -> None:
    layer, store = _make_real_sliding_layer()
    backend = _NullAttentionBackend()
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden = torch.randn(2, hidden_size, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="token_indices must match batch size"):
        deepseek_v4_flash_sliding_layer_forward_batched(
            hidden,
            layer=layer,
            stager=stager,
            backend=backend,
            states=[_make_state()],
            token_indices=[0, 0],
            router_top_k=1,
        )

    with pytest.raises(ValueError, match="batched sliding layer expects 2-D hidden"):
        deepseek_v4_flash_sliding_layer_forward_batched(
            hidden[0],
            layer=layer,
            stager=stager,
            backend=backend,
            states=[_make_state()],
            token_indices=[0],
            router_top_k=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_sliding_layer_non_real_attention_matches_single_slot() -> None:
    """Batched and single-slot paths agree for the non-real attention fallback."""
    layer, store = _make_fake_sliding_layer()
    backend = _NullAttentionBackend()
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden_a = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    hidden_b = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    hidden_batched = torch.stack([hidden_a, hidden_b])

    output_a = deepseek_v4_flash_sliding_layer_forward(
        hidden_a,
        layer=layer,
        stager=stager,
        backend=backend,
        state=None,
        token_idx=0,
        router_top_k=1,
    )
    output_b = deepseek_v4_flash_sliding_layer_forward(
        hidden_b,
        layer=layer,
        stager=stager,
        backend=backend,
        state=None,
        token_idx=1,
        router_top_k=1,
    )

    output_batched = deepseek_v4_flash_sliding_layer_forward_batched(
        hidden_batched,
        layer=layer,
        stager=stager,
        backend=backend,
        states=[_make_state(), _make_state()],
        token_indices=[0, 1],
        router_top_k=1,
    )

    assert output_batched.shape == (2, hidden_size)
    assert output_batched.device.type == "cuda"
    torch.testing.assert_close(output_batched[0], output_a)
    torch.testing.assert_close(output_batched[1], output_b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_sliding_layer_rejects_bad_token_id_tensors_shape() -> None:
    layer, store = _make_real_sliding_layer()
    backend = _NullAttentionBackend()
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden = torch.randn(2, hidden_size, dtype=torch.float32, device="cuda")
    bad_token_ids = torch.tensor([0, 1, 2], dtype=torch.long, device="cuda")

    with pytest.raises(ValueError, match="token_id_tensors must have shape"):
        deepseek_v4_flash_sliding_layer_forward_batched(
            hidden,
            layer=layer,
            stager=stager,
            backend=backend,
            states=[_make_state(), _make_state()],
            token_indices=[0, 0],
            token_id_tensors=bad_token_ids,
            router_top_k=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_sliding_layer_hash_routing_requires_token_ids() -> None:
    layer, store = _make_hash_routed_sliding_layer()
    backend = _NullAttentionBackend()
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden = torch.randn(2, hidden_size, dtype=torch.float32, device="cuda")

    with pytest.raises(
        ValueError,
        match="hash-routed sliding layer requires token_id_tensors or token_ids",
    ):
        deepseek_v4_flash_sliding_layer_forward_batched(
            hidden,
            layer=layer,
            stager=stager,
            backend=backend,
            states=[_make_state(), _make_state()],
            token_indices=[0, 0],
            router_top_k=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_sliding_layer_hash_routing_uses_token_ids() -> None:
    layer, store = _make_hash_routed_sliding_layer()
    backend = _NullAttentionBackend()
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    hidden = torch.randn(2, hidden_size, dtype=torch.float32, device="cuda")

    output = deepseek_v4_flash_sliding_layer_forward_batched(
        hidden,
        layer=layer,
        stager=stager,
        backend=backend,
        states=[_make_state(), _make_state()],
        token_indices=[0, 0],
        token_ids=[0, 1],
        router_top_k=1,
    )

    assert output.shape == (2, hidden_size)
    assert output.device.type == "cuda"
