# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import struct
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash import model as model_module
from vllm.model_executor.models.deepseek_v4_flash.config import (
    DeepSeekV4FlashContextEstimate,
    DeepSeekV4FlashRuntimeBudget,
    DeepSeekV4FlashShape,
)
from vllm.model_executor.models.deepseek_v4_flash.decode_graph import (
    DeepSeekV4FlashDecodeGraph,
)
from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_Q8_0,
    DeepSeekV4FlashTensor,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
    DeepSeekV4FlashGPUCapabilities,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_runtime import (
    DeepSeekV4FlashGPUCacheConfig,
    DeepSeekV4FlashGPURequestState,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashLayerSemanticBindings,
)


class _ReadyBackend:
    is_ready = True
    missing_kernels: tuple[str, ...] = ()

    def require_ready(self) -> None:
        return None

    def sliding_attention(
        self,
        *,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        attn_sinks: torch.Tensor | None,
        token_idx: int,
    ) -> torch.Tensor:
        del kv_rows, attn_sinks, token_idx
        assert query.is_cuda
        return query

    def compressed_attention(
        self,
        *,
        query: torch.Tensor,
        compressed_rows: torch.Tensor,
        selected_rows: torch.Tensor,
    ) -> torch.Tensor:
        del compressed_rows, selected_rows
        assert query.is_cuda
        return query

    def routed_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        del gate_weight, up_weight, down_weight
        assert hidden.is_cuda
        return torch.zeros_like(hidden, dtype=torch.float32)

    def output_argmax(
        self,
        **kwargs: Any,
    ) -> torch.Tensor:
        lm_head_values = kwargs["lm_head_values"]
        assert isinstance(lm_head_values, torch.Tensor)
        assert lm_head_values.is_cuda
        return torch.tensor(
            lm_head_values.shape[0] - 1,
            dtype=torch.long,
            device=lm_head_values.device,
        )


class _FakeStore:
    def __init__(self, *, shape: DeepSeekV4FlashShape, layer_count: int = 0) -> None:
        self.shape = shape
        self.token_embedding = DeepSeekV4FlashTensor(
            "token_embd.weight",
            (shape.hidden_size, shape.vocab_size),
            GGML_TYPE_F16,
            0,
            shape.hidden_size * shape.vocab_size * 2,
        )
        self.output_norm = DeepSeekV4FlashTensor(
            "output_norm.weight",
            (shape.hidden_size,),
            GGML_TYPE_F32,
            0,
            shape.hidden_size * 4,
        )
        self.output_head = DeepSeekV4FlashTensor(
            "output.weight",
            (shape.hidden_size, shape.vocab_size),
            GGML_TYPE_Q8_0,
            0,
            shape.vocab_size * (2 + shape.hidden_size),
        )
        self.output_hc = SimpleNamespace(
            fn=DeepSeekV4FlashTensor(
                "output_hc_fn.weight",
                (4 * shape.hidden_size, 4),
                GGML_TYPE_F32,
                0,
                16 * shape.hidden_size * 4,
            ),
            scale=DeepSeekV4FlashTensor(
                "output_hc_scale.weight",
                (1,),
                GGML_TYPE_F32,
                0,
                4,
            ),
            base=DeepSeekV4FlashTensor(
                "output_hc_base.weight",
                (4,),
                GGML_TYPE_F32,
                0,
                16,
            ),
        )
        self.bindings = SimpleNamespace(
            token_embedding=self.token_embedding,
            layers=tuple(
                DeepSeekV4FlashLayerSemanticBindings(layer_index=idx)
                for idx in range(layer_count)
            ),
            output_hyper_connection=self.output_hc,
            output_norm=self.output_norm,
            output_head=self.output_head,
        )
        embedding = torch.arange(
            shape.vocab_size * shape.hidden_size,
            dtype=torch.float16,
        ).reshape(shape.vocab_size, shape.hidden_size)
        self._payloads = {
            self.token_embedding.name: embedding.numpy().tobytes(),
            self.output_head.name: self._q8_head_payload(shape),
        }
        self._matrices = {
            self.output_hc.fn.name: torch.zeros(
                (4, 4 * shape.hidden_size),
                dtype=torch.float32,
            ),
            self.token_embedding.name: embedding,
        }
        self._vectors = {
            self.output_norm.name: torch.ones(shape.hidden_size, dtype=torch.float32),
            self.output_hc.scale.name: torch.ones(1, dtype=torch.float32),
            self.output_hc.base.name: torch.zeros(4, dtype=torch.float32),
        }

    @staticmethod
    def _q8_head_payload(shape: DeepSeekV4FlashShape) -> bytes:
        if shape.hidden_size % 32 != 0:
            raise ValueError("test hidden size must be divisible by Q8_0 block size")
        row = b"".join(
            struct.pack("<e", 1.0) + bytes([1] * 32)
            for _ in range(shape.hidden_size // 32)
        )
        return row * shape.vocab_size

    def tensor_payload(self, tensor: DeepSeekV4FlashTensor) -> memoryview:
        return memoryview(self._payloads[tensor.name])

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        return self._matrices[tensor.name].clone()

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._vectors[tensor.name].to(dtype=dtype).clone()

    def decode_grouped_expert_matrix(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> torch.Tensor:
        del tensor, expert_id
        raise AssertionError("unit test patches layer forward functions")


def _runtime_budget(context_length: int) -> DeepSeekV4FlashRuntimeBudget:
    return DeepSeekV4FlashRuntimeBudget(
        context=DeepSeekV4FlashContextEstimate(
            context_length=context_length,
            raw_kv_bytes=0,
            compressed_kv_bytes=0,
            scratch_bytes=0,
        ),
        model_mmap_bytes=1,
        resident_weight_bytes=0,
        expert_cache_bytes=0,
        uma_budget_bytes=64 * 1024 * 1024,
        min_system_headroom_bytes=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_graph_matches_reference_step(monkeypatch: pytest.MonkeyPatch) -> None:
    shape = DeepSeekV4FlashShape(
        num_layers=1,
        hidden_size=32,
        vocab_size=7,
        head_dim=32,
    )
    backend = DeepSeekV4FlashGPUBackend(
        capabilities=DeepSeekV4FlashGPUCapabilities(
            q8_linear=True,
            attention=True,
            compressed_attention=True,
            cache_update=True,
            moe=True,
            output=True,
        )
    )
    model = DeepSeekV4FlashForCausalLM(
        shape=shape,
        weight_store=_FakeStore(shape=shape, layer_count=1),  # type: ignore[arg-type]
        runtime_budget=_runtime_budget(8),
        gpu_backend=backend,
    )

    def fake_sliding(hidden: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        assert hidden.is_cuda
        kv_rows_by_layer = kwargs.get("kv_rows_by_layer")
        if kv_rows_by_layer is not None:
            assert kwargs["layer"].layer_index in kv_rows_by_layer
        return hidden.to(torch.float32) + 1.0

    monkeypatch.setattr(
        model_module,
        "deepseek_v4_flash_sliding_layer_forward",
        fake_sliding,
        raising=False,
    )

    device = torch.device("cuda")
    state_ref = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=8,
            hidden_size=shape.hidden_size,
            kv_width=shape.head_dim,
            device=device,
        )
    )
    state_cap = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=8,
            hidden_size=shape.hidden_size,
            kv_width=shape.head_dim,
            device=device,
        )
    )

    token_id = 3
    token_id_tensor = torch.tensor(token_id, dtype=torch.long, device=device)

    ref_out = model._forward_kernel_token_step_token_tensor(
        token_id_tensor=token_id_tensor,
        state=state_ref,
        token_idx=state_ref.token_position,
        device=device,
    )
    assert state_ref.token_position == 1

    kv_rows_by_layer: dict[int, torch.Tensor] = {
        0: torch.zeros((1, shape.head_dim), dtype=torch.float32, device=device),
    }
    graph = DeepSeekV4FlashDecodeGraph.capture(
        model,
        state=state_cap,
        token_idx=state_cap.token_position,
        device=device,
        kv_rows_by_layer=kv_rows_by_layer,
    )
    assert state_cap.token_position == 0
    state_cap.advance_token()

    replay_out = graph.replay(token_id_tensor)
    assert replay_out.shape == ()
    assert replay_out.dtype == torch.long
    assert replay_out.item() == ref_out.item()
