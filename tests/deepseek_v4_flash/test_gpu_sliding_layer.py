from __future__ import annotations

import math

import pytest
import torch

from vllm.kernels.triton.deepseek_v4_flash.attention import (
    DeepSeekV4AttentionKernelInputs,
    deepseek_v4_attention,
)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_deepseek_gpu_sliding_attention_matches_torch_reference() -> None:
    device = torch.device("cuda")
    query = torch.tensor([1.0, 0.0, 0.5, -0.5], device=device)
    kv_rows = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )
    attn_sinks = torch.tensor([0.25], device=device)

    output = deepseek_v4_attention(
        DeepSeekV4AttentionKernelInputs(
            hidden=query,
            kv_rows=kv_rows,
            token_idx=0,
            attn_sinks=attn_sinks,
        )
    )

    scores = kv_rows.matmul(query) / math.sqrt(float(query.numel()))
    logits = torch.cat([scores, attn_sinks])
    probs = torch.softmax(logits, dim=0)
    expected = probs[:-1].matmul(kv_rows)

    assert output.device.type == "cuda"
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_backend_sliding_attention_requires_gpu_tensors() -> None:
    backend = DeepSeekV4FlashGPUBackend()

    with pytest.raises(ValueError, match="must be CUDA tensors"):
        backend.sliding_attention(
            query=torch.zeros(4),
            kv_rows=torch.zeros(1, 4),
            attn_sinks=torch.zeros(1),
            token_idx=0,
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
        "router": _tensor("blk.0.ffn_gate_inp.weight", (hidden_size, num_experts)),
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_sliding_layer_forward_explicit_kv_rows_matches_state_read() -> None:
    layer, store = _make_real_sliding_layer()
    backend = _NullAttentionBackend()
    hidden_size = DEEPSEEK_V4_FLASH_SHAPE.hidden_size
    hidden = torch.randn(hidden_size, dtype=torch.float32, device="cuda") * 0.01
    layer_idx = layer.layer_index
    token_idx = 0

    # Align the router so expert 0 is always selected for this hidden.
    hidden_cpu = hidden.to("cpu")
    num_experts = layer.router.dims[0]
    router = torch.full(
        (num_experts, hidden_size),
        -100.0,
        dtype=torch.float32,
    )
    router[0] = hidden_cpu * 1000.0
    store.matrices[layer.router.name] = router

    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    def _make_state() -> DeepSeekV4FlashGPURequestState:
        return DeepSeekV4FlashGPURequestState(
            DeepSeekV4FlashGPUCacheConfig(
                context_length=128,
                hidden_size=hidden_size,
                device=hidden.device,
            )
        )

    state_a = _make_state()
    output_state_read = deepseek_v4_flash_sliding_layer_forward(
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
    _ = deepseek_v4_flash_sliding_layer_forward(
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
    output_explicit_rows = deepseek_v4_flash_sliding_layer_forward(
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
