from __future__ import annotations

import os
import struct
from pathlib import Path
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
    open_deepseek_v4_flash_weight_store,
)

TARGET_GGUF = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


class _ReadyBackend:
    is_ready = True
    missing_kernels: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.output_calls = 0

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

    def output_logits(self, **kwargs: torch.Tensor | int) -> torch.Tensor:
        tensor_args = [
            value for value in kwargs.values() if isinstance(value, torch.Tensor)
        ]
        assert tensor_args
        assert all(tensor.is_cuda for tensor in tensor_args)
        self.output_calls += 1
        rows = kwargs["lm_head_values"]
        assert isinstance(rows, torch.Tensor)
        return torch.arange(rows.shape[0], dtype=torch.float32, device=rows.device)


class _FakeStore:
    def __init__(self, *, shape: DeepSeekV4FlashShape, layer_count: int = 3) -> None:
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


def test_model_gpu_staging_budget_preserves_required_headroom() -> None:
    budget = DeepSeekV4FlashRuntimeBudget(
        context=DeepSeekV4FlashContextEstimate(
            context_length=4,
            raw_kv_bytes=10,
            compressed_kv_bytes=20,
            scratch_bytes=30,
        ),
        model_mmap_bytes=1,
        resident_weight_bytes=100,
        expert_cache_bytes=200,
        uma_budget_bytes=1000,
        min_system_headroom_bytes=300,
    )
    model = DeepSeekV4FlashForCausalLM(runtime_budget=budget)

    assert model.gpu_staging_memory_stats() == {
        "staged_bytes": 0,
        "max_staged_bytes": 340,
        "dynamic_entries": 0,
        "grouped_entries": 0,
    }


def test_forward_full_kernel_rejects_no_weight_store_with_clear_error() -> None:
    model = DeepSeekV4FlashForCausalLM(gpu_backend=_ReadyBackend())

    with pytest.raises(RuntimeError, match="attached GGUF weight store"):
        model.forward_full(torch.tensor([1], dtype=torch.long), use_kernel=True)


def test_forward_kernel_rejects_cpu_input_ids() -> None:
    shape = DeepSeekV4FlashShape(
        num_layers=1,
        hidden_size=32,
        vocab_size=3,
        head_dim=32,
    )
    model = DeepSeekV4FlashForCausalLM(
        shape=shape,
        weight_store=_FakeStore(shape=shape, layer_count=1),  # type: ignore[arg-type]
        runtime_budget=_runtime_budget(4),
        gpu_backend=_ReadyBackend(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="CUDA input_ids"):
        model.forward_kernel(torch.tensor([1], dtype=torch.long))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_forward_kernel_rejects_non_1d_input_ids() -> None:
    shape = DeepSeekV4FlashShape(
        num_layers=1,
        hidden_size=32,
        vocab_size=3,
        head_dim=32,
    )
    model = DeepSeekV4FlashForCausalLM(
        shape=shape,
        weight_store=_FakeStore(shape=shape, layer_count=1),  # type: ignore[arg-type]
        runtime_budget=_runtime_budget(4),
        gpu_backend=_ReadyBackend(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="1-D batch=1"):
        model.forward_kernel(torch.ones((1, 1), dtype=torch.long, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_forward_kernel_rejects_context_beyond_budget() -> None:
    shape = DeepSeekV4FlashShape(
        num_layers=1,
        hidden_size=32,
        vocab_size=3,
        head_dim=32,
    )
    model = DeepSeekV4FlashForCausalLM(
        shape=shape,
        weight_store=_FakeStore(shape=shape, layer_count=1),  # type: ignore[arg-type]
        runtime_budget=_runtime_budget(1),
        gpu_backend=_ReadyBackend(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="exceeds configured budget"):
        model.forward_kernel(torch.tensor([1, 2], dtype=torch.long, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_forward_kernel_token_step_reuses_supplied_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = DeepSeekV4FlashShape(
        num_layers=3,
        hidden_size=32,
        vocab_size=5,
        head_dim=32,
    )
    model = DeepSeekV4FlashForCausalLM(
        shape=shape,
        weight_store=_FakeStore(shape=shape),  # type: ignore[arg-type]
        runtime_budget=_runtime_budget(4),
        gpu_backend=_ReadyBackend(),  # type: ignore[arg-type]
    )
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=4,
            hidden_size=shape.hidden_size,
            kv_width=shape.head_dim,
            device="cuda",
        )
    )
    seen: list[tuple[int, int]] = []

    def fake_sliding(hidden: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        seen.append((kwargs["layer"].layer_index, kwargs["token_idx"]))
        return hidden

    def fake_compressed(hidden: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        assert kwargs["state"] is state
        assert state.token_position == kwargs["token_idx"]
        seen.append((kwargs["layer"].layer_index, kwargs["token_idx"]))
        return hidden.reshape(1, -1).expand(4, -1).clone()

    monkeypatch.setattr(
        model_module,
        "deepseek_v4_flash_sliding_layer_forward",
        fake_sliding,
        raising=False,
    )
    monkeypatch.setattr(
        model_module,
        "deepseek_v4_flash_compressed_layer_forward",
        fake_compressed,
        raising=False,
    )

    logits0 = model._forward_kernel_token_step(
        token_id=1,
        state=state,
        token_idx=0,
        device=torch.device("cuda"),
    )
    logits1 = model._forward_kernel_token_step(
        token_id=2,
        state=state,
        token_idx=1,
        device=torch.device("cuda"),
    )

    assert logits0.shape == (1, shape.vocab_size)
    assert logits1.shape == (1, shape.vocab_size)
    assert state.token_position == 2
    assert seen == [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_forward_kernel_runs_multiple_fake_layers_and_returns_cuda_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = DeepSeekV4FlashShape(
        num_layers=3,
        hidden_size=32,
        vocab_size=5,
        head_dim=32,
    )
    backend = _ReadyBackend()
    store = _FakeStore(shape=shape)
    model = DeepSeekV4FlashForCausalLM(
        shape=shape,
        weight_store=store,  # type: ignore[arg-type]
        runtime_budget=_runtime_budget(4),
        gpu_backend=backend,  # type: ignore[arg-type]
    )
    calls: list[tuple[str, int, tuple[int, ...]]] = []

    def fake_sliding(hidden: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        layer = kwargs["layer"]
        assert hidden.is_cuda
        calls.append(("sliding", layer.layer_index, tuple(hidden.shape)))
        return hidden.to(torch.float32) + 1.0

    def fake_compressed(hidden: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        layer = kwargs["layer"]
        assert hidden.is_cuda
        calls.append(("compressed", layer.layer_index, tuple(hidden.shape)))
        return hidden.reshape(1, -1).expand(4, -1).clone() + 1.0

    monkeypatch.setattr(
        model_module,
        "deepseek_v4_flash_sliding_layer_forward",
        fake_sliding,
        raising=False,
    )
    monkeypatch.setattr(
        model_module,
        "deepseek_v4_flash_compressed_layer_forward",
        fake_compressed,
        raising=False,
    )

    logits = model.forward_kernel(torch.tensor([2], dtype=torch.long, device="cuda"))

    assert calls == [
        ("sliding", 0, (32,)),
        ("sliding", 1, (32,)),
        ("compressed", 2, (32,)),
    ]
    assert backend.output_calls == 1
    assert logits.device.type == "cuda"
    assert logits.shape == (1, shape.vocab_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_forward_kernel_projects_output_logits_in_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = DeepSeekV4FlashShape(
        num_layers=1,
        hidden_size=32,
        vocab_size=2050,
        head_dim=32,
    )
    backend = _ReadyBackend()
    model = DeepSeekV4FlashForCausalLM(
        shape=shape,
        weight_store=_FakeStore(shape=shape, layer_count=1),  # type: ignore[arg-type]
        runtime_budget=_runtime_budget(4),
        gpu_backend=backend,  # type: ignore[arg-type]
    )
    chunk_rows: list[int] = []

    def output_logits(**kwargs: torch.Tensor | int) -> torch.Tensor:
        rows = kwargs["lm_head_values"]
        assert isinstance(rows, torch.Tensor)
        chunk_rows.append(rows.shape[0])
        return torch.ones(rows.shape[0], dtype=torch.float32, device=rows.device)

    backend.output_logits = output_logits  # type: ignore[method-assign]
    monkeypatch.setattr(
        model_module,
        "deepseek_v4_flash_sliding_layer_forward",
        lambda hidden, **kwargs: hidden,
        raising=False,
    )

    logits = model.forward_kernel(torch.tensor([1], dtype=torch.long, device="cuda"))

    assert logits.shape == (1, shape.vocab_size)
    assert chunk_rows == [1024, 1024, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_forward_kernel_small_fake_model_uses_real_backend_output() -> None:
    shape = DeepSeekV4FlashShape(
        num_layers=0,
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
        weight_store=_FakeStore(shape=shape, layer_count=0),  # type: ignore[arg-type]
        runtime_budget=_runtime_budget(4),
        gpu_backend=backend,
    )

    logits = model.forward_kernel(torch.tensor([1], dtype=torch.long, device="cuda"))

    assert logits.device.type == "cuda"
    assert logits.shape == (1, shape.vocab_size)
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_forward_full_kernel_never_calls_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = DeepSeekV4FlashShape(
        num_layers=1,
        hidden_size=32,
        vocab_size=3,
        head_dim=32,
    )
    model = DeepSeekV4FlashForCausalLM(
        shape=shape,
        weight_store=_FakeStore(shape=shape, layer_count=1),  # type: ignore[arg-type]
        runtime_budget=_runtime_budget(4),
        gpu_backend=_ReadyBackend(),  # type: ignore[arg-type]
    )

    def fail_reference(input_ids: torch.Tensor) -> torch.Tensor:
        del input_ids
        raise AssertionError("forward_full_reference must not run for use_kernel=True")

    monkeypatch.setattr(model, "forward_full_reference", fail_reference)
    monkeypatch.setattr(
        model_module,
        "deepseek_v4_flash_sliding_layer_forward",
        lambda hidden, **kwargs: hidden,
        raising=False,
    )

    logits = model.forward_full(
        torch.tensor([1], dtype=torch.long, device="cuda"),
        use_kernel=True,
    )

    assert logits.device.type == "cuda"
    assert logits.shape == (1, shape.vocab_size)


def test_real_gguf_forward_smoke_is_opt_in() -> None:
    if os.environ.get("RUN_DEEPSEEK_REAL_GGUF_FORWARD") != "1":
        pytest.skip("set RUN_DEEPSEEK_REAL_GGUF_FORWARD=1 to run real GGUF smoke")
    if not TARGET_GGUF.exists():
        pytest.fail(f"RUN_DEEPSEEK_REAL_GGUF_FORWARD=1 but {TARGET_GGUF} is missing")
    if not torch.cuda.is_available():
        pytest.fail("RUN_DEEPSEEK_REAL_GGUF_FORWARD=1 but CUDA is unavailable")

    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        model = DeepSeekV4FlashForCausalLM(
            weight_store=store,
            runtime_budget=_runtime_budget(4096),
            gpu_backend=_ReadyBackend(),  # type: ignore[arg-type]
        )

        logits = model.forward_full(
            torch.tensor([1], dtype=torch.long, device="cuda"),
            use_kernel=True,
        )

    assert logits.device.type == "cuda"
    assert logits.shape == (1, model.shape.vocab_size)
    assert torch.isfinite(logits).all()
