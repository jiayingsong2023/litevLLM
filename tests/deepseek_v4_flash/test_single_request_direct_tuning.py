# Chosen baseline from quick sweep: KV=fp16, BLOCK=32 (best tokens/s observed).
# See task-1-report.md for full sweep numbers.
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

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
    def __init__(self, layer_index: int) -> None:
        self.layer_index = layer_index
        self.grouped_experts: DeepSeekV4FlashGroupedExpertTensors | None = None
        self.expert_token_to_expert_ids: DeepSeekV4FlashTensor | None = None


def _runtime_budget(
    context_length: int,
    *,
    uma_budget_bytes: int = 1,
    min_system_headroom_bytes: int = 0,
) -> DeepSeekV4FlashRuntimeBudget:
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
        uma_budget_bytes=uma_budget_bytes,
        min_system_headroom_bytes=min_system_headroom_bytes,
    )


def _fake_model(
    *,
    context_length: int = 16,
    vocab_size: int = 11,
    runtime_budget: DeepSeekV4FlashRuntimeBudget | None = None,
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
        runtime_budget=runtime_budget or _runtime_budget(context_length),
        gpu_backend=_ReadyBackend(),  # type: ignore[arg-type]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_token_id_decode_passes_capped_compressed_counts() -> None:
    """The token-id fallback path must cap compressed_count like the graph path."""
    model = _fake_model(context_length=16, vocab_size=11)
    # Layers 0/1 are sliding; layers 2+ are compressed.  Layer 4 is ratio-4.
    model.weight_store.bindings.layers = [  # type: ignore[attr-defined]
        _FakeLayer(0),
        _FakeLayer(1),
        _FakeLayer(2),
        _FakeLayer(3),
        _FakeLayer(4),
    ]
    device = torch.device("cuda")

    captured: dict[int, int | None] = {}

    def fake_forward(
        hidden: torch.Tensor,
        *,
        layer: object,
        compressed_count: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        assert hasattr(layer, "layer_index")
        captured[layer.layer_index] = compressed_count  # type: ignore[attr-defined]
        return hidden

    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=16,
            hidden_size=32,
            device=device,
        )
    )
    state.token_position = 7

    with (
        patch.object(
            model_module,
            "deepseek_v4_flash_sliding_layer_forward",
            lambda hidden, **kwargs: hidden,
        ),
        patch.object(
            model_module,
            "deepseek_v4_flash_compressed_layer_forward",
            fake_forward,
        ),
        patch.object(
            model,
            "_stage_token_embedding_cuda",
            lambda store, token_id, *, device: torch.zeros(
                model.shape.hidden_size,
                device=device,
                dtype=torch.float32,
            ),
        ),
        patch.object(
            model,
            "_schedule_next_layer_expert_prefetch",
            lambda stager, next_layer, *, token_id: None,
        ),
        patch.object(
            model,
            "_kernel_output_streams",
            lambda hidden: torch.zeros(
                4,
                model.shape.hidden_size,
                device=hidden.device,
                dtype=torch.float32,
            ),
        ),
        patch.object(
            model,
            "_output_token_argmax_chunked_cuda",
            lambda store, *, stager, streams, device: torch.tensor(
                1,
                device=device,
                dtype=torch.long,
            ),
        ),
    ):
        model._forward_kernel_token_step_token_id(
            token_id=0,
            state=state,
            token_idx=7,
            device=device,
        )

    # Even ratio-4 compressed layer at token_idx=7 sees (7+1)//4 == 2 rows.
    assert captured.get(4) == 2, (
        f"expected compressed_count=2 for layer 4, got {captured}"
    )
    # Odd ratio-128 compressed layer at token_idx=7 sees (7+1)//128 == 0 rows.
    assert captured.get(3) == 0, (
        f"expected compressed_count=0 for layer 3, got {captured}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_hot_expert_pinning_adds_manual_pins() -> None:
    # Reuse the real GGUF fixture or a synthetic weight store.
    # For a minimal unit test, create a model with a mocked weight store
    # that exposes a single hash-routed layer.
    model = MagicMock(spec=DeepSeekV4FlashForCausalLM)
    stager = MagicMock()
    stager.device = torch.device("cuda")
    model._get_gpu_weight_stager = lambda device: stager
    model.runtime_budget = None

    # Patch _require_weight_store to return a fake store
    fake_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    fake_layer = MagicMock()
    fake_layer.layer_index = 2
    fake_layer.expert_token_to_expert_ids = fake_table
    fake_store = MagicMock()
    fake_store.bindings.layers = [fake_layer]
    model._require_weight_store = lambda: fake_store

    DeepSeekV4FlashForCausalLM.pin_hot_experts_for_input_ids(
        model,
        torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
        torch.device("cuda"),
    )
    assert stager.pin_grouped_expert.call_count >= 4


def test_hot_expert_pinning_stages_deepseek_tensor() -> None:
    """The DeepSeekV4FlashTensor table branch uses _stage_expert_token_table."""
    model = MagicMock(spec=DeepSeekV4FlashForCausalLM)
    stager = MagicMock()
    stager.device = torch.device("cpu")
    model._get_gpu_weight_stager = lambda device: stager

    fake_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    fake_tensor = DeepSeekV4FlashTensor(
        "expert_token_to_expert_ids",
        (2, 2),
        GGML_TYPE_F16,
        0,
        16,
    )
    fake_layer = MagicMock()
    fake_layer.layer_index = 2
    fake_layer.expert_token_to_expert_ids = fake_tensor
    fake_store = MagicMock()
    fake_store.bindings.layers = [fake_layer]
    model._require_weight_store = lambda: fake_store

    stager.store.tensor_to_torch.return_value = fake_table

    DeepSeekV4FlashForCausalLM.pin_hot_experts_for_input_ids(
        model,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.device("cpu"),
    )

    stager.store.tensor_to_torch.assert_called_once_with(fake_tensor, dtype=torch.int32)
    assert stager.pin_grouped_expert.call_count >= 4


def test_gpu_staging_budget_respects_override_gb() -> None:
    """FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB adds to the base budget."""
    # Give enough headroom so the 1 GB extra is not capped.
    model = _fake_model(
        context_length=16,
        runtime_budget=_runtime_budget(
            16,
            uma_budget_bytes=4 * 1024 * 1024 * 1024,
            min_system_headroom_bytes=1024 * 1024 * 1024,
        ),
    )
    base = model._gpu_staging_budget_bytes()
    with patch.dict(
        os.environ,
        {"FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB": "1"},
    ):
        overridden = model._gpu_staging_budget_bytes()
    assert overridden is not None
    assert base is not None
    assert overridden == base + 1024 * 1024 * 1024


def test_gpu_staging_budget_malformed_env_defaults_to_zero() -> None:
    """A malformed FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB is ignored."""
    model = _fake_model(
        context_length=16,
        runtime_budget=_runtime_budget(
            16,
            uma_budget_bytes=4 * 1024 * 1024 * 1024,
            min_system_headroom_bytes=1024 * 1024 * 1024,
        ),
    )
    base = model._gpu_staging_budget_bytes()
    with (
        patch.dict(
            os.environ,
            {"FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB": "not-a-number"},
        ),
        pytest.warns(UserWarning, match="malformed"),
    ):
        malformed = model._gpu_staging_budget_bytes()
    assert malformed is not None
    assert base is not None
    assert malformed == base


def test_full_resident_env_removes_staging_budget_cap() -> None:
    model = _fake_model(
        context_length=16,
        runtime_budget=_runtime_budget(
            16,
            uma_budget_bytes=4 * 1024 * 1024 * 1024,
            min_system_headroom_bytes=1024 * 1024 * 1024,
        ),
    )

    with patch.dict(
        os.environ,
        {"FASTINFERENCE_DEEPSEEK_V4_FLASH_FULL_RESIDENT": "1"},
    ):
        stager = model._get_gpu_weight_stager(torch.device("cuda"))

    assert stager.max_staged_bytes is None
    assert stager.full_resident_enabled is True
