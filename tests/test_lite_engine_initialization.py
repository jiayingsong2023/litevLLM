# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LiteEngine initialization helpers.

These tests avoid GPU dependencies by exercising the pure helpers and by
mocking torch tensors / devices. They verify the new initializer modules used
by ``LiteEngine.__init__`` behave consistently with the old inline logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.engine.block_allocator import BlockAllocator
from vllm.engine.initialization import (
    KVCacheAllocator,
    LiteRuntimeAssembler,
    MemoryAuditor,
)
from vllm.engine.initialization.kv_cache_allocator import (
    compute_kv_scale_theory_bytes,
    compute_kv_theory_bytes,
    layer_kv_cache_shape_for_layer,
    resolve_layer_kv_specs,
)


class DummySelfAttn:
    def __init__(self, num_kv_heads: int, head_dim: int) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


class DummyLayer:
    def __init__(self, num_kv_heads: int, head_dim: int) -> None:
        self.self_attn = DummySelfAttn(num_kv_heads, head_dim)


class DummyModel:
    def __init__(self, layers: list[DummyLayer]) -> None:
        self.model = MagicMock()
        self.model.layers = layers


def test_resolve_layer_kv_specs_success() -> None:
    layers = [DummyLayer(8, 128), DummyLayer(4, 64)]
    model = DummyModel(layers)
    specs = resolve_layer_kv_specs(model, num_layers=2)
    assert specs == [(8, 128), (4, 64)]


def test_resolve_layer_kv_specs_length_mismatch() -> None:
    layers = [DummyLayer(8, 128)]
    model = DummyModel(layers)
    assert resolve_layer_kv_specs(model, num_layers=2) is None


def test_resolve_layer_kv_specs_missing_self_attn() -> None:
    layer = MagicMock()
    del layer.self_attn
    model = DummyModel([layer])
    assert resolve_layer_kv_specs(model, num_layers=1) is None


def test_layer_kv_cache_shape_fallback() -> None:
    shape = layer_kv_cache_shape_for_layer(
        layer_kv_specs=None,
        layer_idx=0,
        kv_dtype=torch.float16,
        fallback_num_kv_heads=8,
        fallback_kv_head_dim=128,
    )
    assert shape == (8, 128)


def test_layer_kv_cache_shape_uint8_packs_head_dim() -> None:
    shape = layer_kv_cache_shape_for_layer(
        layer_kv_specs=[(8, 128)],
        layer_idx=0,
        kv_dtype=torch.uint8,
        fallback_num_kv_heads=8,
        fallback_kv_head_dim=128,
    )
    assert shape == (8, 64)


def test_layer_kv_cache_shape_fallback_uint8_uses_packed_dim() -> None:
    """When specs are unavailable, fallback_kv_head_dim is already cache-side."""
    shape = layer_kv_cache_shape_for_layer(
        layer_kv_specs=None,
        layer_idx=0,
        kv_dtype=torch.uint8,
        fallback_num_kv_heads=8,
        fallback_kv_head_dim=64,
    )
    assert shape == (8, 64)


def test_compute_kv_theory_bytes_uniform() -> None:
    # 2 layers, 4 blocks, block_size 8, 2 kv heads, 16 head_dim, fp16 = 2 bytes
    # data = 2 * 2 * 4 * 8 * 2 * 16 * 2 = 8192
    assert (
        compute_kv_theory_bytes(
            layer_kv_specs=None,
            num_layers=2,
            num_total_blocks=4,
            block_size=8,
            fallback_num_kv_heads=2,
            fallback_kv_head_dim=16,
            kv_dtype=torch.float16,
            needs_scale_cache=False,
        )
        == 8192
    )


def test_compute_kv_theory_bytes_with_scale() -> None:
    # data same as above = 8192
    # scale = 2 * 2 * 4 * 8 * 2 * 4 = 1024
    assert (
        compute_kv_theory_bytes(
            layer_kv_specs=None,
            num_layers=2,
            num_total_blocks=4,
            block_size=8,
            fallback_num_kv_heads=2,
            fallback_kv_head_dim=16,
            kv_dtype=torch.float16,
            needs_scale_cache=True,
        )
        == 9216
    )


def test_compute_kv_theory_bytes_per_layer_specs() -> None:
    # Specs: (1 head, 8 dim), (2 heads, 16 dim), fp16 = 2 bytes
    # Layer 0 data: 2 * 2 * 4 * 1 * 8 * 2 = 256
    # Layer 1 data: 2 * 2 * 4 * 2 * 16 * 2 = 1024
    assert (
        compute_kv_theory_bytes(
            layer_kv_specs=[(1, 8), (2, 16)],
            num_layers=2,
            num_total_blocks=2,
            block_size=4,
            fallback_num_kv_heads=99,
            fallback_kv_head_dim=99,
            kv_dtype=torch.float16,
            needs_scale_cache=False,
        )
        == 1280
    )


def test_compute_kv_scale_theory_bytes() -> None:
    # scale = 2 * 2 * 4 * 8 * 2 * 4 = 1024
    assert (
        compute_kv_scale_theory_bytes(
            layer_kv_specs=None,
            num_layers=2,
            num_total_blocks=4,
            block_size=8,
            fallback_num_kv_heads=2,
            fallback_kv_head_dim=16,
        )
        == 1024
    )


def _make_meta_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="meta")


def test_memory_auditor_counts_params_and_buffers() -> None:
    param = torch.nn.Parameter(_make_meta_tensor((2, 4), torch.float16))
    buffer = _make_meta_tensor((3,), torch.float32)
    model = MagicMock(spec=torch.nn.Module)
    model.named_parameters.return_value = [("p", param)]
    model.named_buffers.return_value = [("b", buffer)]

    with patch(
        "vllm.model_executor.layers.quantization.tensor.get_awq_runtime_stats",
        return_value={"cache_bytes": 64},
    ):
        auditor = MemoryAuditor(device=torch.device("meta"), topn=5)
        audit = auditor.audit(model)

    assert audit["params_count"] == 1
    assert audit["buffers_count"] == 1
    assert audit["params_total_bytes"] == 16  # 2*4*2
    assert audit["buffers_total_bytes"] == 12  # 3*4
    assert audit["awq_cache_bytes"] == 64


def test_memory_auditor_filters_off_device() -> None:
    param = torch.nn.Parameter(torch.empty((2, 4), dtype=torch.float16, device="cpu"))
    model = MagicMock(spec=torch.nn.Module)
    model.named_parameters.return_value = [("p", param)]
    model.named_buffers.return_value = []

    auditor = MemoryAuditor(device=torch.device("meta"), topn=5)
    audit = auditor.audit(model)
    assert audit["params_count"] == 0


def test_kv_cache_allocator_no_scale() -> None:
    allocator = KVCacheAllocator(
        num_layers=2,
        num_total_blocks=4,
        block_size=8,
        device=torch.device("meta"),
    )
    kv_caches, kv_scale_caches = allocator.allocate(
        layer_kv_specs=None,
        kv_dtype=torch.float16,
        kv_head_dim=16,
        fallback_num_kv_heads=2,
        fallback_kv_head_dim=16,
        needs_scale_cache=False,
    )
    assert len(kv_caches) == 2
    assert all(k is not None and v is not None for k, v in kv_caches)
    assert kv_scale_caches == [(None, None), (None, None)]
    # Meta-device tensors carry shape/dtype without allocating memory.
    for k, v in kv_caches:
        assert k.shape == (4, 8, 2, 16)
        assert k.dtype == torch.float16
        assert v.shape == k.shape


def test_kv_cache_allocator_with_scale() -> None:
    allocator = KVCacheAllocator(
        num_layers=1,
        num_total_blocks=2,
        block_size=4,
        device=torch.device("meta"),
    )
    kv_caches, kv_scale_caches = allocator.allocate(
        layer_kv_specs=[(2, 16)],
        kv_dtype=torch.uint8,
        kv_head_dim=16,
        fallback_num_kv_heads=2,
        fallback_kv_head_dim=16,
        needs_scale_cache=True,
    )
    assert len(kv_caches) == 1
    assert len(kv_scale_caches) == 1
    ks, vs = kv_scale_caches[0]
    assert ks is not None and vs is not None
    k, v = kv_caches[0]
    assert k.shape == (2, 4, 2, 8)  # uint8 packs head_dim / 2
    assert ks.shape == (2, 4, 2, 1)


def test_lite_runtimeAssembler_signature_compiles_context() -> None:
    """Smoke-test that the assembler forwards to the factory with all fields."""
    kv_caches: list[torch.Tensor] = []
    kv_scale_caches: list[torch.Tensor] = []
    with patch(
        "vllm.engine.initialization.runtime_component_factory.LiteRuntimeFactory.build"
    ) as mock_build:
        mock_build.return_value = {"dummy": True}
        result = LiteRuntimeAssembler.assemble(
            block_allocator=BlockAllocator(num_total_blocks=8),
            kv_caches=kv_caches,
            kv_scale_caches=kv_scale_caches,
            num_blocks_per_seq=4,
            block_size=8,
            device=torch.device("meta"),
            max_model_len=128,
            num_layers=2,
            inf_config=MagicMock(),
            stack_per_layer_carries=None,
            split_per_layer_carries=None,
            model=MagicMock(spec=torch.nn.Module),
            fast_input_ids=MagicMock(spec=torch.Tensor),
            fast_positions=MagicMock(spec=torch.Tensor),
            fast_slot_mapping=MagicMock(spec=torch.Tensor),
            fast_seq_lens=MagicMock(spec=torch.Tensor),
            step_token_budget=16,
            decode_priority_enabled=False,
            prefill_chunk_size=128,
            prefill_reserved_tokens=0,
            prefill_reserve_backlog=0,
            prefill_catchup_ratio=1.0,
            prefill_microbatch_size=1,
            min_prefill_chunk_size=1,
            max_prefill_chunk_size=None,
            prefill_sla_ttft_ms=0.0,
            max_active_requests=1,
            scheduler_policy="fcfs",
            backend_policy="eager",
            scheduler=MagicMock(),
            observer=MagicMock(),
            lora_registry=MagicMock(),
            queue_timeout_s=0.0,
        )
    assert result == {"dummy": True}
    call_args = mock_build.call_args
    assert call_args is not None
    context = call_args[0][0]
    assert context.kv_caches is kv_caches
    assert context.kv_scale_caches is kv_scale_caches
    assert context.max_model_len == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
