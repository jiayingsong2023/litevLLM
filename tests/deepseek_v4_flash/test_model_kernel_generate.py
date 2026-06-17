from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest
import torch

import vllm.model_executor.models.deepseek_v4_flash.model as model_module
from vllm.model_executor.models.deepseek_v4_flash.config import (
    DeepSeekV4FlashContextEstimate,
    DeepSeekV4FlashRuntimeBudget,
    DeepSeekV4FlashShape,
)
from vllm.model_executor.models.deepseek_v4_flash.expert_cache import (
    DeepSeekV4FlashExpertPrefetchRequest,
)
from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_F16,
    GGML_TYPE_Q8_0,
    DeepSeekV4FlashTensor,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_runtime import (
    DeepSeekV4FlashGPUCacheConfig,
    DeepSeekV4FlashGPURequestState,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashGroupedExpertTensors,
    open_deepseek_v4_flash_weight_store,
)

TARGET_GGUF = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
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

    def output_logits(self, **kwargs: torch.Tensor | int) -> torch.Tensor:
        tensor_args = [
            value for value in kwargs.values() if isinstance(value, torch.Tensor)
        ]
        assert tensor_args
        assert all(tensor.is_cuda for tensor in tensor_args)
        rows = kwargs["lm_head_values"]
        assert isinstance(rows, torch.Tensor)
        return torch.arange(rows.shape[0], dtype=torch.float32, device=rows.device)


class _FakeStore:
    def __init__(self, shape: DeepSeekV4FlashShape) -> None:
        self.bindings = type(
            "Bindings",
            (),
            {
                "token_embedding": DeepSeekV4FlashTensor(
                    "token_embd.weight",
                    (shape.hidden_size, shape.vocab_size),
                    GGML_TYPE_F16,
                    0,
                    shape.hidden_size * shape.vocab_size * 2,
                )
            },
        )()


class _FakeLayer:
    def __init__(
        self,
        layer_index: int,
        grouped_experts: DeepSeekV4FlashGroupedExpertTensors | None,
        expert_token_to_expert_ids: DeepSeekV4FlashTensor | None = None,
    ) -> None:
        self.layer_index = layer_index
        self.grouped_experts = grouped_experts
        self.expert_token_to_expert_ids = expert_token_to_expert_ids


def _grouped_experts(
    layer_idx: int,
    *,
    expert_count: int = 4,
) -> DeepSeekV4FlashGroupedExpertTensors:
    return DeepSeekV4FlashGroupedExpertTensors(
        gate=DeepSeekV4FlashTensor(
            f"blk.{layer_idx}.ffn_gate_exps.weight",
            (2, 2, expert_count),
            GGML_TYPE_Q8_0,
            0,
            0,
        ),
        up=DeepSeekV4FlashTensor(
            f"blk.{layer_idx}.ffn_up_exps.weight",
            (2, 2, expert_count),
            GGML_TYPE_Q8_0,
            0,
            0,
        ),
        down=DeepSeekV4FlashTensor(
            f"blk.{layer_idx}.ffn_down_exps.weight",
            (2, 2, expert_count),
            GGML_TYPE_Q8_0,
            0,
            0,
        ),
    )


def _expert_token_table(layer_idx: int) -> DeepSeekV4FlashTensor:
    return DeepSeekV4FlashTensor(
        f"blk.{layer_idx}.ffn_expert_token_map.weight",
        (2, 16),
        GGML_TYPE_Q8_0,
        0,
        0,
    )


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
        uma_budget_bytes=1,
        min_system_headroom_bytes=0,
    )


def _fake_model(
    *,
    context_length: int = 16,
    vocab_size: int = 11,
) -> DeepSeekV4FlashForCausalLM:
    shape = DeepSeekV4FlashShape(
        num_layers=0,
        hidden_size=32,
        vocab_size=vocab_size,
        head_dim=32,
    )
    return DeepSeekV4FlashForCausalLM(
        shape=shape,
        weight_store=_FakeStore(shape),  # type: ignore[arg-type]
        runtime_budget=_runtime_budget(context_length),
        gpu_backend=_ReadyBackend(),  # type: ignore[arg-type]
    )


def _pack_q8_0_block(scale: float, values: tuple[int, ...]) -> bytes:
    if len(values) != 32:
        raise ValueError("Q8_0 test block must contain 32 values")
    return struct.pack("<e", scale) + struct.pack("<" + "b" * 32, *values)


def test_model_profile_summary_reports_disabled_profiler_by_default() -> None:
    model = _fake_model()

    assert model.deepseek_profile() == {
        "enabled": False,
        "events": [],
        "counters": {},
    }


def test_prepare_for_serving_sets_context_and_preserves_stager() -> None:
    model = _fake_model(context_length=4)

    model.prepare_for_serving(
        context_length=4,
        device=torch.device("cuda"),
    )
    first_stager = model._gpu_weight_stager
    stats = model.prepare_for_serving(
        context_length=4,
        device=torch.device("cuda"),
    )

    assert model._gpu_weight_stager is first_stager
    assert model._get_gpu_weight_stager(torch.device("cuda:0")) is first_stager
    assert stats == model.gpu_staging_memory_stats()
    assert model.gpu_staging_memory_stats()["max_staged_bytes"] is not None


def test_prepare_deepseek_hot_experts_reuses_stager_and_stages_bounded_raw_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_model(context_length=4)
    assert model.weight_store is not None
    model.weight_store.bindings.layers = [
        _FakeLayer(0, _grouped_experts(0)),
        _FakeLayer(1, None),
        _FakeLayer(2, _grouped_experts(2)),
    ]
    created_stagers = 0

    class FakeStager:
        hot_expert_policy = type("Policy", (), {"pinned_experts": frozenset()})()

        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal created_stagers
            del args, kwargs
            created_stagers += 1
            self.payload_calls: list[tuple[str, int, int | None]] = []

        def stage_grouped_expert_payload(
            self,
            tensor: DeepSeekV4FlashTensor,
            expert_id: int,
            *,
            layer_idx: int | None = None,
        ) -> object:
            self.payload_calls.append((tensor.name, expert_id, layer_idx))
            return object()

        def memory_stats(self) -> dict[str, int | None]:
            return {
                "staged_bytes": len(self.payload_calls),
                "max_staged_bytes": 99,
            }

    monkeypatch.setattr(model_module, "DeepSeekV4FlashGPUWeightStager", FakeStager)

    model.prepare_for_serving(context_length=4, device=torch.device("cuda"))
    first_stager = model._gpu_weight_stager
    stats = model.prepare_deepseek_hot_experts(
        device=torch.device("cuda"),
        max_layers=1,
        experts_per_layer=2,
    )
    second_stats = model.prepare_deepseek_hot_experts(
        device=torch.device("cuda:0"),
        max_layers=1,
        experts_per_layer=2,
    )

    assert model.runtime_budget is not None
    assert model.runtime_budget.context.context_length == 4
    assert created_stagers == 1
    assert model._gpu_weight_stager is first_stager
    assert isinstance(first_stager, FakeStager)
    assert stats == {"staged_bytes": 6, "max_staged_bytes": 99}
    assert second_stats == {"staged_bytes": 12, "max_staged_bytes": 99}
    assert first_stager.payload_calls[:6] == [
        ("blk.0.ffn_gate_exps.weight", 0, 0),
        ("blk.0.ffn_up_exps.weight", 0, 0),
        ("blk.0.ffn_down_exps.weight", 0, 0),
        ("blk.0.ffn_gate_exps.weight", 1, 0),
        ("blk.0.ffn_up_exps.weight", 1, 0),
        ("blk.0.ffn_down_exps.weight", 1, 0),
    ]


def test_prepare_for_serving_rejects_context_mismatch() -> None:
    model = _fake_model(context_length=4)

    with pytest.raises(ValueError, match="context_length"):
        model.prepare_for_serving(
            context_length=8,
            device=torch.device("cuda"),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_model_enable_deepseek_profile_records_generate_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_model()
    model.enable_deepseek_profile()

    def fake_token_step_token_id(
        *,
        token_id: int,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        del token_id, token_idx
        state.advance_token()
        return torch.tensor(2, dtype=torch.long, device=device)

    monkeypatch.setattr(
        model,
        "_validate_generate_greedy_kernel_input",
        lambda input_ids, *, max_tokens: input_ids.device,
    )
    monkeypatch.setattr(model.gpu_backend, "require_ready", lambda: None)
    monkeypatch.setattr(
        model,
        "_forward_kernel_token_step_token_id",
        fake_token_step_token_id,
    )

    output = model.generate_greedy_kernel(
        torch.tensor([1], dtype=torch.long, device="cuda"),
        max_tokens=1,
    )

    profile = model.deepseek_profile()
    event_names = [event["name"] for event in profile["events"]]

    assert output.tolist() == [1, 2]
    assert "generate_greedy_kernel" in event_names


def test_model_profile_records_layer_output_sections_and_syncs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_model()
    sync_calls = 0

    class FakeLayer:
        def __init__(self, layer_index: int) -> None:
            self.layer_index = layer_index

    class FakeLogits:
        is_cuda = True

    def fake_sync() -> None:
        nonlocal sync_calls
        sync_calls += 1

    assert model.weight_store is not None
    model.weight_store.bindings.layers = [FakeLayer(0), FakeLayer(2)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", fake_sync)
    monkeypatch.setattr(model, "_get_gpu_weight_stager", lambda device: object())
    monkeypatch.setattr(
        model,
        "_stage_token_embedding_cuda",
        lambda store, token_id, *, device: object(),
    )
    monkeypatch.setattr(
        model_module,
        "deepseek_v4_flash_sliding_layer_forward",
        lambda hidden, **kwargs: hidden,
    )
    monkeypatch.setattr(
        model_module,
        "deepseek_v4_flash_compressed_layer_forward",
        lambda hidden, **kwargs: hidden,
    )
    monkeypatch.setattr(model, "_kernel_output_streams", lambda hidden: object())
    monkeypatch.setattr(
        model,
        "_output_logits_chunked_cuda",
        lambda store, *, stager, streams, device: FakeLogits(),
    )

    model.enable_deepseek_profile()
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=4,
            hidden_size=model.shape.hidden_size,
            kv_width=model.shape.head_dim,
            device="cpu",
        )
    )

    logits = model._forward_kernel_token_step(
        token_id=1,
        state=state,
        token_idx=0,
        device=torch.device("cpu"),
    )

    profile = model.deepseek_profile()
    event_names = [event["name"] for event in profile["events"]]

    assert logits.is_cuda
    assert state.token_position == 1
    assert event_names == [
        "layer_0_sliding",
        "layer_2_compressed",
        "output_projection",
    ]
    assert sync_calls >= 2 * len(event_names)


def test_forward_kernel_schedules_next_layer_prefetch_before_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_model()
    assert model.weight_store is not None
    token_table = _expert_token_table(1)
    model.weight_store.bindings.layers = [
        _FakeLayer(0, _grouped_experts(0)),
        _FakeLayer(1, _grouped_experts(1), token_table),
    ]
    events: list[tuple[str, int, tuple[int, ...] | None]] = []

    class FakeStager:
        def __init__(self) -> None:
            self.store = self

        def tensor_to_torch(
            self,
            tensor: DeepSeekV4FlashTensor,
            *,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            assert tensor is token_table
            assert dtype is torch.int32
            table = torch.full((2, 16), -1, dtype=torch.int32)
            table[:, 5] = torch.tensor([2, 0], dtype=torch.int32)
            return table

        def prefetch_grouped_experts(
            self,
            tensors: DeepSeekV4FlashGroupedExpertTensors,
            request: DeepSeekV4FlashExpertPrefetchRequest,
        ) -> None:
            assert tensors is model.weight_store.bindings.layers[1].grouped_experts
            events.append(("prefetch", request.layer_idx, request.expert_ids))

    def fake_sliding(hidden: object, **kwargs: object) -> object:
        layer = kwargs["layer"]
        assert isinstance(layer, _FakeLayer)
        events.append(("demand", layer.layer_index, None))
        return hidden

    monkeypatch.setattr(model.gpu_backend, "require_ready", lambda: None)
    monkeypatch.setattr(model, "_get_gpu_weight_stager", lambda device: FakeStager())
    monkeypatch.setattr(
        model,
        "_stage_token_embedding_cuda",
        lambda store, token_id, *, device: object(),
    )
    monkeypatch.setattr(
        model_module,
        "deepseek_v4_flash_sliding_layer_forward",
        fake_sliding,
    )

    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=4,
            hidden_size=model.shape.hidden_size,
            kv_width=model.shape.head_dim,
            device="cpu",
        )
    )

    model._forward_kernel_token_hidden(
        token_id=5,
        state=state,
        token_idx=0,
        device=torch.device("cpu"),
    )

    assert events == [
        ("demand", 0, None),
        ("prefetch", 1, (0, 2)),
        ("demand", 1, None),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_generate_greedy_kernel_uses_output_argmax_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_model()
    calls = {"argmax": 0}

    assert model.weight_store is not None
    model.weight_store.bindings.layers = []

    def fake_output_argmax(
        store: object,
        *,
        stager: object,
        streams: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        del store, stager
        assert streams.is_cuda
        calls["argmax"] += 1
        return torch.tensor(2, dtype=torch.long, device=device)

    def fail_logits_step(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("generate_greedy_kernel used logits path")

    monkeypatch.setattr(model, "_get_gpu_weight_stager", lambda device: object())
    monkeypatch.setattr(
        model,
        "_stage_token_embedding_cuda",
        lambda store, token_id, *, device: torch.ones(
            (model.shape.hidden_size,),
            dtype=torch.float32,
            device=device,
        ),
    )
    monkeypatch.setattr(model, "_kernel_output_streams", lambda hidden: hidden)
    monkeypatch.setattr(
        model,
        "_output_token_argmax_chunked_cuda",
        fake_output_argmax,
    )
    monkeypatch.setattr(model, "_forward_kernel_token_step", fail_logits_step)
    monkeypatch.setattr(model, "_output_logits_chunked_cuda", fail_logits_step)

    output = model.generate_greedy_kernel(
        torch.tensor([1], dtype=torch.long, device="cuda"),
        max_tokens=1,
    )

    assert output.tolist() == [1, 2]
    assert calls == {"argmax": 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_generate_greedy_kernel_appends_one_token_and_advances_shared_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_model()
    seen_state_ids: list[int] = []
    seen_positions: list[int] = []

    def fake_token_step_token_id(
        *,
        token_id: int,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        del token_id
        seen_state_ids.append(id(state))
        seen_positions.append(token_idx)
        state.advance_token()
        return torch.tensor(7, dtype=torch.long, device=device)

    monkeypatch.setattr(
        model,
        "_forward_kernel_token_step_token_id",
        fake_token_step_token_id,
    )

    output = model.generate_greedy_kernel(
        torch.tensor([2, 3], dtype=torch.long, device="cuda"),
        max_tokens=1,
    )

    assert output.device.type == "cuda"
    assert output.dtype == torch.long
    assert output.tolist() == [2, 3, 7]
    assert seen_positions == [0, 1]
    assert len(set(seen_state_ids)) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_generate_greedy_kernel_appends_eight_tokens_and_reuses_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_model(context_length=12)
    seen_state_ids: list[int] = []
    seen_positions: list[int] = []
    seen_token_ids: list[int] = []
    seen_tensor_positions: list[int] = []
    seen_tensor_state_ids: list[int] = []

    def fake_token_step_token_id(
        *,
        token_id: int,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        seen_token_ids.append(token_id)
        seen_state_ids.append(id(state))
        seen_positions.append(token_idx)
        next_token = (token_idx + 1) % model.shape.vocab_size
        state.advance_token()
        return torch.tensor(next_token, dtype=torch.long, device=device)

    def fake_token_step_token_tensor(
        *,
        token_id_tensor: torch.Tensor,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        assert token_id_tensor.is_cuda
        assert token_id_tensor.ndim == 0
        seen_tensor_positions.append(token_idx)
        seen_tensor_state_ids.append(id(state))
        next_token = (token_idx + 1) % model.shape.vocab_size
        state.advance_token()
        return torch.tensor(next_token, dtype=torch.long, device=device)

    monkeypatch.setattr(
        model,
        "_forward_kernel_token_step_token_id",
        fake_token_step_token_id,
    )
    monkeypatch.setattr(
        model,
        "_forward_kernel_token_step_token_tensor",
        fake_token_step_token_tensor,
    )

    output = model.generate_greedy_kernel(
        torch.tensor([4, 5], dtype=torch.int32, device="cuda"),
        max_tokens=8,
    )

    assert output.device.type == "cuda"
    assert output.dtype == torch.long
    assert output.tolist() == [4, 5, 2, 3, 4, 5, 6, 7, 8, 9]
    assert seen_positions == [0, 1]
    assert seen_token_ids == [4, 5]
    assert seen_tensor_positions == list(range(2, 9))
    assert len(set(seen_state_ids + seen_tensor_state_ids)) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_generate_greedy_kernel_uses_tensor_token_path_for_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_model(context_length=8)
    prompt_token_ids: list[int] = []
    tensor_positions: list[int] = []
    tensor_state_ids: list[int] = []

    def fake_token_step_token_id(
        *,
        token_id: int,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        prompt_token_ids.append(token_id)
        state.advance_token()
        return torch.tensor((token_idx + 1) % model.shape.vocab_size, device=device)

    def fake_token_step_token_tensor(
        *,
        token_id_tensor: torch.Tensor,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        assert token_id_tensor.is_cuda
        assert token_id_tensor.ndim == 0
        assert token_id_tensor.dtype == torch.long
        torch.testing.assert_close(
            token_id_tensor,
            torch.tensor(token_idx, dtype=torch.long, device=device),
        )
        tensor_positions.append(token_idx)
        tensor_state_ids.append(id(state))
        state.advance_token()
        return torch.tensor((token_idx + 1) % model.shape.vocab_size, device=device)

    monkeypatch.setattr(
        model,
        "_forward_kernel_token_step_token_id",
        fake_token_step_token_id,
    )
    monkeypatch.setattr(
        model,
        "_forward_kernel_token_step_token_tensor",
        fake_token_step_token_tensor,
        raising=False,
    )

    output = model.generate_greedy_kernel(
        torch.tensor([4, 5], dtype=torch.int32, device="cuda"),
        max_tokens=3,
    )

    assert output.device.type == "cuda"
    assert output.dtype == torch.long
    assert output.tolist() == [4, 5, 2, 3, 4]
    assert prompt_token_ids == [4, 5]
    assert tensor_positions == [2, 3]
    assert len(set(tensor_state_ids)) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_generate_greedy_kernel_rejects_context_overflow() -> None:
    model = _fake_model(context_length=3)

    with pytest.raises(ValueError, match="exceeds configured budget"):
        model.generate_greedy_kernel(
            torch.tensor([1, 2], dtype=torch.long, device="cuda"),
            max_tokens=2,
        )


def test_generate_greedy_kernel_rejects_cpu_input_ids() -> None:
    model = _fake_model()

    with pytest.raises(ValueError, match="CUDA input_ids"):
        model.generate_greedy_kernel(torch.tensor([1], dtype=torch.long), max_tokens=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("max_tokens", [0, -1])
def test_generate_greedy_kernel_rejects_non_positive_max_tokens(
    max_tokens: int,
) -> None:
    model = _fake_model()

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        model.generate_greedy_kernel(
            torch.tensor([1], dtype=torch.long, device="cuda"),
            max_tokens=max_tokens,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_generate_greedy_kernel_rejects_non_integer_input_ids() -> None:
    model = _fake_model()

    with pytest.raises(ValueError, match="integer dtype"):
        model.generate_greedy_kernel(
            torch.tensor([1.0], dtype=torch.float32, device="cuda"),
            max_tokens=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_output_token_argmax_chunked_cuda_uses_argmax_for_each_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_rows = model_module._OUTPUT_PROJECTION_CHUNK_ROWS
    vocab_size = chunk_rows + 17
    model = _fake_model(vocab_size=vocab_size)
    assert model.weight_store is not None
    output_head = DeepSeekV4FlashTensor(
        "output.weight",
        (model.shape.hidden_size, model.shape.vocab_size),
        GGML_TYPE_Q8_0,
        0,
        model.shape.vocab_size * (2 + model.shape.hidden_size),
    )
    output_norm = DeepSeekV4FlashTensor(
        "output_norm.weight",
        (model.shape.hidden_size,),
        GGML_TYPE_F16,
        0,
        model.shape.hidden_size * 2,
    )
    model.weight_store.bindings.output_head = output_head
    model.weight_store.bindings.output_norm = output_norm

    class FakeStager:
        def stage_vector(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
            del tensor
            return torch.ones(
                (model.shape.hidden_size,),
                dtype=torch.float32,
                device="cuda",
            )

        def get_output_q8_chunk(
            self,
            tensor: DeepSeekV4FlashTensor,
            *,
            row_start: int,
            row_end: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del tensor
            rows = row_end - row_start
            values = torch.zeros((rows, 1), dtype=torch.int8, device="cuda")
            scales = torch.ones((rows, 1), dtype=torch.float32, device="cuda")
            return values, scales

    class FakeBackend(_ReadyBackend):
        def __init__(self) -> None:
            self.argmax_offsets: list[int] = []

        def output_argmax_with_value(
            self,
            **kwargs: torch.Tensor | int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            row_offset = kwargs["row_offset"]
            assert isinstance(row_offset, int)
            self.argmax_offsets.append(row_offset)
            token = row_offset + (3 if row_offset == 0 else 5)
            value = 1.0 if row_offset == 0 else 7.0
            return (
                torch.tensor(token, dtype=torch.long, device="cuda"),
                torch.tensor(value, dtype=torch.float32, device="cuda"),
            )

        def output_logits(self, **kwargs: torch.Tensor | int) -> torch.Tensor:
            del kwargs
            raise AssertionError("greedy output argmax path materialized logits")

    backend = FakeBackend()
    model.gpu_backend = backend  # type: ignore[assignment]
    monkeypatch.setattr(
        model,
        "_stage_output_hyper_connection_cuda",
        lambda stager, *, stream_elements: (
            torch.zeros((4, stream_elements), dtype=torch.float32, device="cuda"),
            torch.ones((1,), dtype=torch.float32, device="cuda"),
            torch.zeros((4,), dtype=torch.float32, device="cuda"),
        ),
    )

    token = model._output_token_argmax_chunked_cuda(
        model.weight_store,
        stager=FakeStager(),  # type: ignore[arg-type]
        streams=torch.ones((4, 8), dtype=torch.float32, device="cuda"),
        device=torch.device("cuda"),
    )

    assert token.item() == chunk_rows + 5
    assert backend.argmax_offsets == [0, chunk_rows]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_output_token_argmax_chunked_cuda_uses_raw_q8_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_python_unpack(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("output argmax must not split Q8 payload in Python")

    monkeypatch.setattr(
        model_module,
        "q8_0_matrix_from_gguf_payload",
        fail_python_unpack,
    )
    chunk_rows = model_module._OUTPUT_PROJECTION_CHUNK_ROWS
    vocab_size = chunk_rows + 17
    model = _fake_model(vocab_size=vocab_size)
    assert model.weight_store is not None
    output_head = DeepSeekV4FlashTensor(
        "output.weight",
        (model.shape.hidden_size, model.shape.vocab_size),
        GGML_TYPE_Q8_0,
        0,
        model.shape.vocab_size * (2 + model.shape.hidden_size),
    )
    output_norm = DeepSeekV4FlashTensor(
        "output_norm.weight",
        (model.shape.hidden_size,),
        GGML_TYPE_F16,
        0,
        model.shape.hidden_size * 2,
    )
    model.weight_store.bindings.output_head = output_head
    model.weight_store.bindings.output_norm = output_norm
    rows = []
    for row in range(vocab_size):
        value = 10 if row == chunk_rows + 5 else 0
        rows.append(_pack_q8_0_block(1.0, tuple([value] + [0] * 31)))
    raw_payload = torch.tensor(
        tuple(b"".join(rows)),
        dtype=torch.uint8,
        device="cuda",
    )

    class FakeRawStager:
        def __init__(self) -> None:
            self.raw_calls = 0

        def stage_vector(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
            del tensor
            return torch.ones(
                (model.shape.hidden_size,),
                dtype=torch.float32,
                device="cuda",
            )

        def stage_q8_raw_payload(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
            del tensor
            self.raw_calls += 1
            return raw_payload

    class RawOnlyBackend(_ReadyBackend):
        def output_logits(self, **kwargs: torch.Tensor | int) -> torch.Tensor:
            del kwargs
            raise AssertionError("raw output argmax must not use split output logits")

        def output_argmax_with_value(
            self,
            **kwargs: torch.Tensor | int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del kwargs
            raise AssertionError("raw output argmax must not use split argmax")

    model.gpu_backend = RawOnlyBackend()  # type: ignore[assignment]
    monkeypatch.setattr(
        model,
        "_stage_output_hyper_connection_cuda",
        lambda stager, *, stream_elements: (
            torch.zeros((4, stream_elements), dtype=torch.float32, device="cuda"),
            torch.ones((1,), dtype=torch.float32, device="cuda"),
            torch.zeros((4,), dtype=torch.float32, device="cuda"),
        ),
    )
    stager = FakeRawStager()

    token = model._output_token_argmax_chunked_cuda(
        model.weight_store,
        stager=stager,  # type: ignore[arg-type]
        streams=torch.ones(
            (4, model.shape.hidden_size),
            dtype=torch.float32,
            device="cuda",
        ),
        device=torch.device("cuda"),
    )

    assert token.item() == chunk_rows + 5
    assert stager.raw_calls == 1


def test_real_gguf_generate_one_real_step_then_state_preserving_decode_smoke_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.environ.get("RUN_DEEPSEEK_REAL_GGUF_GENERATE") != "1":
        pytest.skip("set RUN_DEEPSEEK_REAL_GGUF_GENERATE=1 to run real GGUF smoke")
    if not TARGET_GGUF.exists():
        pytest.fail(f"RUN_DEEPSEEK_REAL_GGUF_GENERATE=1 but {TARGET_GGUF} is missing")
    if not torch.cuda.is_available():
        pytest.fail("RUN_DEEPSEEK_REAL_GGUF_GENERATE=1 but CUDA is unavailable")

    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        model = DeepSeekV4FlashForCausalLM(
            weight_store=store,
            runtime_budget=_runtime_budget(4096),
            gpu_backend=_ReadyBackend(),  # type: ignore[arg-type]
        )
        prompt = torch.tensor([1], dtype=torch.long, device="cuda")
        original_token_step_token_id = model._forward_kernel_token_step_token_id
        real_step_positions: list[int] = []
        one_token_fake_positions: list[int] = []
        eight_token_fake_positions: list[int] = []
        eight_token_fake_state_ids: list[int] = []

        def install_one_real_step_wrapper(
            fake_positions: list[int],
            fake_state_ids: list[int] | None = None,
        ) -> None:
            real_steps_remaining = 1

            def one_real_step_then_fake_decode(
                *,
                token_id: int,
                state: DeepSeekV4FlashGPURequestState,
                token_idx: int,
                device: torch.device,
            ) -> torch.Tensor:
                nonlocal real_steps_remaining
                if real_steps_remaining > 0:
                    real_steps_remaining -= 1
                    real_step_positions.append(token_idx)
                    return original_token_step_token_id(
                        token_id=token_id,
                        state=state,
                        token_idx=token_idx,
                        device=device,
                    )
                fake_positions.append(token_idx)
                if fake_state_ids is not None:
                    fake_state_ids.append(id(state))
                state.advance_token()
                next_token = (token_idx + 1) % model.shape.vocab_size
                return torch.tensor(next_token, dtype=torch.long, device=device)

            def fake_decode_token_tensor(
                *,
                token_id_tensor: torch.Tensor,
                state: DeepSeekV4FlashGPURequestState,
                token_idx: int,
                device: torch.device,
            ) -> torch.Tensor:
                assert token_id_tensor.is_cuda
                fake_positions.append(token_idx)
                if fake_state_ids is not None:
                    fake_state_ids.append(id(state))
                state.advance_token()
                next_token = (token_idx + 1) % model.shape.vocab_size
                return torch.tensor(next_token, dtype=torch.long, device=device)

            monkeypatch.setattr(
                model,
                "_forward_kernel_token_step_token_id",
                one_real_step_then_fake_decode,
            )
            monkeypatch.setattr(
                model,
                "_forward_kernel_token_step_token_tensor",
                fake_decode_token_tensor,
            )

        install_one_real_step_wrapper(one_token_fake_positions)
        one_token = model.generate_greedy_kernel(prompt, max_tokens=1)
        install_one_real_step_wrapper(
            eight_token_fake_positions,
            eight_token_fake_state_ids,
        )
        eight_tokens = model.generate_greedy_kernel(prompt, max_tokens=8)

    assert one_token.device.type == "cuda"
    assert eight_tokens.device == prompt.device
    assert one_token.shape == (prompt.numel() + 1,)
    assert eight_tokens.shape == (prompt.numel() + 8,)
    assert real_step_positions == [0, 0]
    assert one_token_fake_positions == []
    assert eight_token_fake_positions == list(range(1, 8))
    assert len(set(eight_token_fake_state_ids)) == 1


def test_real_gguf_generate_full_decode_smoke_is_slow_opt_in() -> None:
    if os.environ.get("RUN_DEEPSEEK_REAL_GGUF_GENERATE_FULL") != "1":
        pytest.skip(
            "set RUN_DEEPSEEK_REAL_GGUF_GENERATE_FULL=1 to run full real GGUF decode"
        )
    if not TARGET_GGUF.exists():
        pytest.fail(
            f"RUN_DEEPSEEK_REAL_GGUF_GENERATE_FULL=1 but {TARGET_GGUF} is missing"
        )
    if not torch.cuda.is_available():
        pytest.fail("RUN_DEEPSEEK_REAL_GGUF_GENERATE_FULL=1 but CUDA is unavailable")

    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        model = DeepSeekV4FlashForCausalLM(
            weight_store=store,
            runtime_budget=_runtime_budget(4096),
            gpu_backend=_ReadyBackend(),  # type: ignore[arg-type]
        )
        prompt = torch.tensor([1], dtype=torch.long, device="cuda")

        output = model.generate_greedy_kernel(prompt, max_tokens=8)

    assert output.device == prompt.device
    assert output.shape == (prompt.numel() + 8,)
