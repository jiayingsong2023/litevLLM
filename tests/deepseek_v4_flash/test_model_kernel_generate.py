from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import vllm.model_executor.models.deepseek_v4_flash.model as model_module
from vllm.model_executor.models.deepseek_v4_flash.config import (
    DeepSeekV4FlashContextEstimate,
    DeepSeekV4FlashRuntimeBudget,
    DeepSeekV4FlashShape,
)
from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_F16,
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


def test_model_profile_summary_reports_disabled_profiler_by_default() -> None:
    model = _fake_model()

    assert model.deepseek_profile() == {
        "enabled": False,
        "events": [],
        "counters": {},
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_model_enable_deepseek_profile_records_generate_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_model()
    model.enable_deepseek_profile()

    def fake_token_step(
        *,
        token_id: int,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        del token_id, token_idx
        logits = torch.zeros((1, model.shape.vocab_size), device=device)
        logits[0, 2] = 1.0
        state.advance_token()
        return logits

    monkeypatch.setattr(
        model,
        "_validate_generate_greedy_kernel_input",
        lambda input_ids, *, max_tokens: input_ids.device,
    )
    monkeypatch.setattr(model.gpu_backend, "require_ready", lambda: None)
    monkeypatch.setattr(model, "_forward_kernel_token_step", fake_token_step)

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_generate_greedy_kernel_appends_one_token_and_advances_shared_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fake_model()
    seen_state_ids: list[int] = []
    seen_positions: list[int] = []

    def fake_token_step(
        *,
        token_id: int,
        state: DeepSeekV4FlashGPURequestState,
        token_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        del token_id
        seen_state_ids.append(id(state))
        seen_positions.append(token_idx)
        logits = torch.zeros((1, model.shape.vocab_size), device=device)
        logits[0, 7] = 1.0
        state.advance_token()
        return logits

    monkeypatch.setattr(model, "_forward_kernel_token_step", fake_token_step)

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

    def fake_token_step(
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
        logits = torch.zeros((1, model.shape.vocab_size), device=device)
        logits[0, next_token] = 1.0
        state.advance_token()
        return logits

    monkeypatch.setattr(model, "_forward_kernel_token_step", fake_token_step)

    output = model.generate_greedy_kernel(
        torch.tensor([4, 5], dtype=torch.int32, device="cuda"),
        max_tokens=8,
    )

    assert output.device.type == "cuda"
    assert output.dtype == torch.long
    assert output.tolist() == [4, 5, 2, 3, 4, 5, 6, 7, 8, 9]
    assert seen_positions == list(range(9))
    assert seen_token_ids == [4, 5, 2, 3, 4, 5, 6, 7, 8]
    assert len(set(seen_state_ids)) == 1


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
        original_token_step = model._forward_kernel_token_step
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
                    return original_token_step(
                        token_id=token_id,
                        state=state,
                        token_idx=token_idx,
                        device=device,
                    )
                fake_positions.append(token_idx)
                if fake_state_ids is not None:
                    fake_state_ids.append(id(state))
                logits = torch.zeros((1, model.shape.vocab_size), device=device)
                logits[0, (token_idx + 1) % model.shape.vocab_size] = 1.0
                state.advance_token()
                return logits

            monkeypatch.setattr(
                model,
                "_forward_kernel_token_step",
                one_real_step_then_fake_decode,
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
