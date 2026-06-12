from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_Q8_0,
    DeepSeekV4FlashTensor,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_weight_staging import (
    DeepSeekV4FlashGPUWeightStager,
)
from vllm.model_executor.models.deepseek_v4_flash.moe import grouped_expert_reference
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashGroupedExpertTensors,
    open_deepseek_v4_flash_weight_store,
)

TARGET_GGUF = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


class _FakeGroupedExpertStore:
    def __init__(self) -> None:
        self.decode_count = 0
        self.matrices: dict[tuple[str, int], torch.Tensor] = {}

    def decode_grouped_expert_matrix(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> torch.Tensor:
        self.decode_count += 1
        return self.matrices[(tensor.name, expert_id)].clone()

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        raise AssertionError(f"unexpected matrix decode for {tensor.name}")

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise AssertionError(f"unexpected vector read for {tensor.name} as {dtype}")


class _FakeStagingStore(_FakeGroupedExpertStore):
    def __init__(self) -> None:
        super().__init__()
        self.matrix_decode_count = 0
        self.vector_read_count = 0
        self.generic_matrices: dict[str, torch.Tensor] = {}
        self.vectors: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        self.matrix_decode_count += 1
        return self.generic_matrices[tensor.name].clone()

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        self.vector_read_count += 1
        return self.vectors[(tensor.name, dtype)].clone()


def _tensor(name: str, dims: tuple[int, ...] = (2, 2, 1)) -> DeepSeekV4FlashTensor:
    return DeepSeekV4FlashTensor(
        name=name,
        dims=dims,
        tensor_type=GGML_TYPE_Q8_0,
        offset=0,
        nbytes=0,
    )


def test_gpu_weight_stager_rejects_negative_staging_budget() -> None:
    store = _FakeStagingStore()

    with pytest.raises(ValueError, match="max staged bytes"):
        DeepSeekV4FlashGPUWeightStager(
            store,
            device="cuda",
            max_staged_bytes=-1,
        )


def test_gpu_weight_stager_records_cache_hits_and_misses_without_gpu() -> None:
    store = _FakeStagingStore()
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    stager.record_cache_miss("dynamic", 16, tensor_name="a")
    stager.record_cache_hit("dynamic", tensor_name="a")
    stager.record_cache_miss("grouped", 32, tensor_name="b")

    assert stager.cache_stats() == {
        "dynamic_hits": 1,
        "dynamic_misses": 1,
        "grouped_hits": 0,
        "grouped_misses": 1,
        "loaded_bytes": 48,
    }


def test_gpu_weight_stager_memory_stats_include_cache_stats() -> None:
    store = _FakeStagingStore()
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    stager.record_cache_miss("dynamic", 8, tensor_name="a")

    stats = stager.memory_stats()

    assert stats["loaded_bytes"] == 8
    assert stats["dynamic_misses"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_caches_decoded_grouped_expert_matrix() -> None:
    store = _FakeGroupedExpertStore()
    tensor = _tensor("blk.1.ffn_gate_exps.weight")
    store.matrices[(tensor.name, 0)] = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    first = stager.stage_grouped_expert_matrix(tensor, 0)
    second = stager.stage_grouped_expert_matrix(tensor, 0)

    assert first.device.type == "cuda"
    assert first.data_ptr() == second.data_ptr()
    assert store.decode_count == 1
    torch.testing.assert_close(first.cpu(), store.matrices[(tensor.name, 0)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_tracks_staged_bytes_once_per_cache_key() -> None:
    store = _FakeStagingStore()
    matrix_tensor = _tensor("blk.1.ffn_gate_inp.weight", dims=(2, 2))
    vector_tensor = _tensor("blk.1.attn_norm.weight", dims=(4,))
    store.generic_matrices[matrix_tensor.name] = torch.eye(2, dtype=torch.float32)
    store.vectors[(vector_tensor.name, torch.float16)] = torch.ones(
        4,
        dtype=torch.float16,
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    stager.stage_matrix(matrix_tensor)
    stager.stage_matrix(matrix_tensor)
    stager.stage_vector(vector_tensor, dtype=torch.float16)
    stager.stage_vector(vector_tensor, dtype=torch.float16)

    assert stager.staged_bytes == 24
    assert stager.memory_stats() == {
        "staged_bytes": 24,
        "max_staged_bytes": None,
        "dynamic_entries": 2,
        "grouped_entries": 0,
        "dynamic_hits": 2,
        "dynamic_misses": 2,
        "grouped_hits": 0,
        "grouped_misses": 0,
        "loaded_bytes": 24,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_rejects_new_tensor_when_staging_budget_is_exceeded() -> None:
    store = _FakeStagingStore()
    matrix_tensor = _tensor("blk.1.ffn_gate_inp.weight", dims=(2, 2))
    store.generic_matrices[matrix_tensor.name] = torch.eye(2, dtype=torch.float32)
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=15,
    )

    with pytest.raises(RuntimeError, match="staging cache exceeds memory budget"):
        stager.stage_matrix(matrix_tensor)

    assert stager.staged_bytes == 0
    stats = stager.cache_stats()
    assert stats["dynamic_misses"] == 0
    assert stats["loaded_bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_caches_decoded_matrix_by_cache_key() -> None:
    store = _FakeStagingStore()
    tensor = _tensor("blk.1.ffn_gate_inp.weight", dims=(2, 2))
    store.generic_matrices[tensor.name] = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    first = stager.stage_matrix(tensor)
    second = stager.stage_matrix(tensor)

    assert first.device.type == "cuda"
    assert first.data_ptr() == second.data_ptr()
    assert store.matrix_decode_count == 1
    torch.testing.assert_close(first.cpu(), store.generic_matrices[tensor.name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_caches_vector_by_cache_key() -> None:
    store = _FakeStagingStore()
    tensor = _tensor("blk.1.attn_norm.weight", dims=(4,))
    store.vectors[(tensor.name, torch.float32)] = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float32,
    )
    store.vectors[(tensor.name, torch.float16)] = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float16,
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    first = stager.stage_vector(tensor)
    second = stager.stage_vector(tensor)
    half = stager.stage_vector(tensor, dtype=torch.float16)
    cached_half = stager.stage_vector(tensor, dtype=torch.float16)

    assert first.device.type == "cuda"
    assert first.data_ptr() == second.data_ptr()
    assert half.device.type == "cuda"
    assert half.data_ptr() == cached_half.data_ptr()
    assert first.data_ptr() != half.data_ptr()
    assert store.vector_read_count == 2
    torch.testing.assert_close(first.cpu(), store.vectors[(tensor.name, torch.float32)])
    torch.testing.assert_close(half.cpu(), store.vectors[(tensor.name, torch.float16)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_keeps_matrix_and_vector_cache_keys_distinct() -> None:
    store = _FakeStagingStore()
    tensor = _tensor("shared.name.weight", dims=(2, 2))
    store.generic_matrices[tensor.name] = torch.eye(2, dtype=torch.float32)
    store.vectors[(tensor.name, torch.float32)] = torch.ones(2, dtype=torch.float32)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    matrix = stager.stage_matrix(tensor)
    vector = stager.stage_vector(tensor)
    cached_matrix = stager.stage_matrix(tensor)
    cached_vector = stager.stage_vector(tensor)

    assert matrix.device.type == "cuda"
    assert vector.device.type == "cuda"
    assert matrix.data_ptr() == cached_matrix.data_ptr()
    assert vector.data_ptr() == cached_vector.data_ptr()
    assert matrix.data_ptr() != vector.data_ptr()
    assert store.matrix_decode_count == 1
    assert store.vector_read_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_weight_stager_clear_dynamic_cache_preserves_grouped_expert_cache() -> None:
    store = _FakeStagingStore()
    matrix_tensor = _tensor("blk.1.ffn_gate_inp.weight", dims=(2, 2))
    expert_tensor = _tensor("blk.1.ffn_gate_exps.weight")
    store.generic_matrices[matrix_tensor.name] = torch.eye(2, dtype=torch.float32)
    store.matrices[(expert_tensor.name, 0)] = torch.eye(2, dtype=torch.float32)
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

    first_matrix = stager.stage_matrix(matrix_tensor)
    first_expert = stager.stage_grouped_expert_matrix(expert_tensor, 0)
    assert stager.staged_bytes == 32
    stager.clear_dynamic_cache()
    assert stager.staged_bytes == 16
    second_matrix = stager.stage_matrix(matrix_tensor)
    second_expert = stager.stage_grouped_expert_matrix(expert_tensor, 0)

    assert first_matrix.data_ptr() != second_matrix.data_ptr()
    assert first_expert.data_ptr() == second_expert.data_ptr()
    assert store.matrix_decode_count == 2
    assert store.decode_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_backend_runs_staged_q2_iq2_expert_gemm_on_device() -> None:
    device = torch.device("cuda")
    hidden = torch.tensor([1.0, -2.0], device=device)
    gate = torch.tensor([[0.5, 1.0], [1.5, -0.5]], device=device)
    up = torch.tensor([[2.0, -1.0], [0.25, 0.75]], device=device)
    down = torch.tensor([[1.0, 0.5], [-0.25, 2.0]], device=device)
    backend = DeepSeekV4FlashGPUBackend()

    output = backend.routed_expert_gemm(
        hidden=hidden,
        gate_weight=gate,
        up_weight=up,
        down_weight=down,
    )

    expected = grouped_expert_reference(
        hidden.cpu(),
        gate.cpu(),
        up.cpu(),
        down.cpu(),
    ).to(device)
    assert output.device.type == "cuda"
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.skipif(
    os.environ.get("RUN_DEEPSEEK_REAL_GGUF_STAGING") != "1",
    reason="real GGUF staging is opt-in",
)
@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target GGUF not downloaded")
def test_real_gguf_grouped_expert_stages_to_gpu() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        layer = next(
            layer
            for layer in store.bindings.layers
            if layer.grouped_experts is not None
        )
        assert layer.grouped_experts is not None
        stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")

        staged = stager.stage_grouped_expert(
            DeepSeekV4FlashGroupedExpertTensors(
                gate=layer.grouped_experts.gate,
                up=layer.grouped_experts.up,
                down=layer.grouped_experts.down,
            ),
            expert_id=0,
        )
        router_layer = next(layer for layer in store.bindings.layers if layer.router)
        assert router_layer.router is not None
        attention_norm_layer = next(
            layer for layer in store.bindings.layers if layer.attention_norm
        )
        assert attention_norm_layer.attention_norm is not None

        router = stager.stage_matrix(router_layer.router)
        attention_norm = stager.stage_vector(attention_norm_layer.attention_norm)
        output_norm = (
            stager.stage_vector(store.bindings.output_norm)
            if store.bindings.output_norm is not None
            else None
        )

    assert staged.gate.device.type == "cuda"
    assert staged.up.device.type == "cuda"
    assert staged.down.device.type == "cuda"
    assert router.device.type == "cuda"
    assert attention_norm.device.type == "cuda"
    if output_norm is not None:
        assert output_norm.device.type == "cuda"
    assert staged.gate.ndim == 2
    assert staged.up.ndim == 2
    assert staged.down.ndim == 2
    assert router.ndim == 2
    assert attention_norm.ndim == 1
    if output_norm is not None:
        assert output_norm.ndim == 1
    assert torch.isfinite(staged.gate).all()
    assert torch.isfinite(router).all()
    assert torch.isfinite(attention_norm).all()
    if output_norm is not None:
        assert torch.isfinite(output_norm).all()
