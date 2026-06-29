from __future__ import annotations

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.config import (
    DeepSeekV4FlashContextEstimate,
    DeepSeekV4FlashRuntimeBudget,
    DeepSeekV4FlashShape,
)
from vllm.model_executor.models.deepseek_v4_flash.decode_graph import (
    DeepSeekV4FlashDecodeGraph,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashGroupedExpertTensors,
)


class _ReadyBackend:
    is_ready = True
    missing_kernels: tuple[str, ...] = ()
    _stats = {"cpu_token_sync_points": 0}

    def require_ready(self) -> None:
        return None

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

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
    def __init__(
        self,
        shape: DeepSeekV4FlashShape,
        *,
        eos_token_id: int | None = None,
    ) -> None:
        self.model = type(
            "GGUF",
            (),
            {
                "metadata": (
                    {}
                    if eos_token_id is None
                    else {"tokenizer.ggml.eos_token_id": eos_token_id}
                )
            },
        )()
        self.bindings = type(
            "Bindings",
            (),
            {
                "token_embedding": object(),
            },
        )()


class _FakeLayer:
    def __init__(
        self,
        layer_index: int,
        grouped_experts: DeepSeekV4FlashGroupedExpertTensors | None,
    ) -> None:
        self.layer_index = layer_index
        self.grouped_experts = grouped_experts
        self.expert_token_to_expert_ids = None


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
    eos_token_id: int | None = None,
) -> DeepSeekV4FlashForCausalLM:
    shape = DeepSeekV4FlashShape(
        num_layers=0,
        hidden_size=32,
        vocab_size=vocab_size,
        head_dim=32,
    )
    return DeepSeekV4FlashForCausalLM(
        shape=shape,
        weight_store=_FakeStore(  # type: ignore[arg-type]
            shape,
            eos_token_id=eos_token_id,
        ),
        runtime_budget=_runtime_budget(context_length),
        gpu_backend=_ReadyBackend(),  # type: ignore[arg-type]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_generate_greedy_kernel_batched_matches_single_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_model(context_length=16)

    def fake_token_step_token_id(
        *,
        token_id: int,
        state: object,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        del token_idx
        state.advance_token()
        return torch.tensor(
            (token_id + 1) % model.shape.vocab_size,
            dtype=torch.long,
            device=device,
        )

    def fake_token_step_token_tensor(
        *,
        token_id_tensor: torch.Tensor,
        state: object,
        token_idx: int,
        device: torch.device,
        **kwargs: object,
    ) -> torch.Tensor:
        del token_idx, kwargs
        state.advance_token()
        return (token_id_tensor + 1).to(
            dtype=torch.long,
            device=device,
        ) % model.shape.vocab_size

    def fake_token_step_batched(
        *,
        token_id_tensor: torch.Tensor,
        states: list[object],
        token_indices: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        del token_indices
        for state in states:
            state.advance_token()
        return (token_id_tensor + 1).to(
            dtype=torch.long,
            device=device,
        ) % model.shape.vocab_size

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
    monkeypatch.setattr(
        model,
        "_forward_kernel_token_step_batched",
        fake_token_step_batched,
    )

    prompts = [
        torch.tensor([1], dtype=torch.long, device="cuda"),
        torch.tensor([2, 3], dtype=torch.long, device="cuda"),
        torch.tensor([4, 5, 6], dtype=torch.long, device="cuda"),
    ]
    max_tokens = 4

    single_slot_outputs = [
        model.generate_greedy_kernel(prompt, max_tokens=max_tokens)
        for prompt in prompts
    ]
    batched_outputs = model.generate_greedy_kernel_batched(
        prompts,
        max_tokens=max_tokens,
    )

    assert len(batched_outputs) == len(prompts)
    for single, batched, prompt in zip(
        single_slot_outputs, batched_outputs, prompts, strict=True
    ):
        assert batched.tolist() == single.tolist()
        assert batched.shape == (prompt.numel() + max_tokens,)
        assert batched.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_generate_greedy_kernel_batched_variable_prompt_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_model(context_length=16)

    def fake_token_step_token_id(
        *,
        token_id: int,
        state: object,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        del token_idx
        state.advance_token()
        return torch.tensor(
            (token_id + 1) % model.shape.vocab_size,
            dtype=torch.long,
            device=device,
        )

    def fake_token_step_batched(
        *,
        token_id_tensor: torch.Tensor,
        states: list[object],
        token_indices: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        del token_indices
        for state in states:
            state.advance_token()
        return (token_id_tensor + 1).to(
            dtype=torch.long,
            device=device,
        ) % model.shape.vocab_size

    monkeypatch.setattr(
        model,
        "_forward_kernel_token_step_token_id",
        fake_token_step_token_id,
    )
    monkeypatch.setattr(
        model,
        "_forward_kernel_token_step_batched",
        fake_token_step_batched,
    )

    prompts = [
        torch.tensor([1], dtype=torch.long, device="cuda"),
        torch.tensor([2, 3], dtype=torch.long, device="cuda"),
        torch.tensor([4, 5, 6, 7], dtype=torch.long, device="cuda"),
    ]
    max_tokens = 3

    batched_outputs = model.generate_greedy_kernel_batched(
        prompts,
        max_tokens=max_tokens,
    )

    for prompt, output in zip(prompts, batched_outputs, strict=True):
        assert output[: prompt.numel()].tolist() == prompt.tolist()
        assert output.shape == (prompt.numel() + max_tokens,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_generate_greedy_kernel_batched_stops_on_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_model(context_length=16, eos_token_id=3)

    def fake_token_step_token_id(
        *,
        token_id: int,
        state: object,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        del token_idx
        state.advance_token()
        return torch.tensor(
            (token_id + 1) % model.shape.vocab_size,
            dtype=torch.long,
            device=device,
        )

    def fake_token_step_batched(
        *,
        token_id_tensor: torch.Tensor,
        states: list[object],
        token_indices: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        del token_indices
        for state in states:
            state.advance_token()
        return (token_id_tensor + 1).to(
            dtype=torch.long,
            device=device,
        ) % model.shape.vocab_size

    monkeypatch.setattr(
        model,
        "_forward_kernel_token_step_token_id",
        fake_token_step_token_id,
    )
    monkeypatch.setattr(
        model,
        "_forward_kernel_token_step_batched",
        fake_token_step_batched,
    )

    prompts = [
        torch.tensor([2], dtype=torch.long, device="cuda"),
        torch.tensor([0], dtype=torch.long, device="cuda"),
        torch.tensor([0], dtype=torch.long, device="cuda"),
    ]
    max_tokens = 2

    batched_outputs = model.generate_greedy_kernel_batched(
        prompts,
        max_tokens=max_tokens,
    )

    assert batched_outputs[0].tolist() == [2, 3]
    assert batched_outputs[1].tolist() == [0, 1, 2]
    assert batched_outputs[2].tolist() == [0, 1, 2]


def test_generate_greedy_kernel_batched_rejects_empty_input_ids_list() -> None:
    model = _fake_model()

    with pytest.raises(ValueError, match="input_ids_list must not be empty"):
        model.generate_greedy_kernel_batched([], max_tokens=1)


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_generate_greedy_kernel_batched_rejects_non_positive_max_tokens(
    max_tokens: int,
) -> None:
    model = _fake_model()
    input_ids = torch.tensor([1], dtype=torch.long, device="cuda")

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        model.generate_greedy_kernel_batched([input_ids], max_tokens=max_tokens)


def test_generate_greedy_kernel_batched_rejects_non_1d_input_tensors() -> None:
    model = _fake_model()
    input_ids = torch.tensor([[1]], dtype=torch.long, device="cuda")

    with pytest.raises(ValueError, match="1-D"):
        model.generate_greedy_kernel_batched([input_ids], max_tokens=1)


def test_generate_greedy_kernel_batched_rejects_cpu_input_ids() -> None:
    model = _fake_model()
    input_ids = torch.tensor([1], dtype=torch.long)

    with pytest.raises(ValueError, match="CUDA input tensors"):
        model.generate_greedy_kernel_batched([input_ids], max_tokens=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_generate_greedy_kernel_batched_rejects_hyper_connections() -> None:
    model = _fake_model(context_length=8)
    assert model.weight_store is not None

    hyper_layer = _FakeLayer(0, None)
    hyper_layer.attention_hyper_connection = object()
    model.weight_store.bindings.layers = [hyper_layer]

    input_ids = torch.tensor([1], dtype=torch.long, device="cuda")

    with pytest.raises(NotImplementedError, match="hyper-connections"):
        model.generate_greedy_kernel_batched([input_ids], max_tokens=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_generate_greedy_kernel_batched_never_uses_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_model(context_length=16)

    def fake_token_step_token_id(
        *,
        token_id: int,
        state: object,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        del token_idx
        state.advance_token()
        return torch.tensor(
            (token_id + 1) % model.shape.vocab_size,
            dtype=torch.long,
            device=device,
        )

    def fake_token_step_batched(
        *,
        token_id_tensor: torch.Tensor,
        states: list[object],
        token_indices: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        del token_indices
        for state in states:
            state.advance_token()
        return (token_id_tensor + 1).to(
            dtype=torch.long,
            device=device,
        ) % model.shape.vocab_size

    monkeypatch.setattr(
        model,
        "_forward_kernel_token_step_token_id",
        fake_token_step_token_id,
    )
    monkeypatch.setattr(
        model,
        "_forward_kernel_token_step_batched",
        fake_token_step_batched,
    )

    capture_count = 0
    replay_count = 0
    original_capture = DeepSeekV4FlashDecodeGraph.capture
    original_replay = DeepSeekV4FlashDecodeGraph.replay

    def counting_capture(
        cls: type[DeepSeekV4FlashDecodeGraph],
        *args: object,
        **kwargs: object,
    ) -> DeepSeekV4FlashDecodeGraph:
        nonlocal capture_count
        capture_count += 1
        return original_capture(*args, **kwargs)

    def counting_replay(
        self: DeepSeekV4FlashDecodeGraph,
        token_id_tensor: torch.Tensor,
        *,
        compressed_counts_by_layer: dict[int, int] | None = None,
    ) -> torch.Tensor:
        nonlocal replay_count
        replay_count += 1
        return original_replay(
            self,
            token_id_tensor,
            compressed_counts_by_layer=compressed_counts_by_layer,
        )

    monkeypatch.setattr(
        DeepSeekV4FlashDecodeGraph,
        "capture",
        classmethod(counting_capture),
    )
    monkeypatch.setattr(DeepSeekV4FlashDecodeGraph, "replay", counting_replay)

    model.generate_greedy_kernel_batched(
        [torch.tensor([1], dtype=torch.long, device="cuda")],
        max_tokens=3,
    )

    assert capture_count == 0
    assert replay_count == 0
