from __future__ import annotations

import os
import struct
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q8_0,
    DeepSeekV4FlashTensor,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_layers import (
    _env_limited_top_k,
    _run_hash_routed_experts,
    _run_staged_routed_experts,
    _run_staged_shared_expert,
    deepseek_v4_flash_q8_0_tensor_projection,
    deepseek_v4_flash_router_topk,
    deepseek_v4_flash_sliding_layer_forward,
    deepseek_v4_flash_staged_matrix_projection,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_runtime import (
    DeepSeekV4FlashGPUCacheConfig,
    DeepSeekV4FlashGPURequestState,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_weight_staging import (
    DeepSeekV4FlashGPUWeightStager,
    DeepSeekV4FlashStagedQuantizedExpertPayload,
)
from vllm.model_executor.models.deepseek_v4_flash.profiler import (
    DeepSeekV4FlashProfiler,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashGroupedExpertTensors,
    DeepSeekV4FlashHyperConnectionTensors,
    DeepSeekV4FlashLayerSemanticBindings,
    DeepSeekV4FlashQuantizedExpertPayload,
    open_deepseek_v4_flash_weight_store,
)

TARGET_GGUF = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


class _FakeLayerStore:
    def __init__(self) -> None:
        self.decode_count = 0
        self.matrices: dict[str, torch.Tensor] = {}
        self.vectors: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self.expert_matrices: dict[tuple[str, int], torch.Tensor] = {}
        self.raw_payload_read_count = 0
        self.raw_payloads: dict[tuple[str, int], bytes] = {}
        self.payloads: dict[str, bytes] = {}

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        self.decode_count += 1
        return self.matrices[tensor.name].clone()

    def tensor_payload(self, tensor: DeepSeekV4FlashTensor) -> memoryview:
        return memoryview(bytearray(self.payloads[tensor.name]))

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
        return self.expert_matrices[(tensor.name, expert_id)].clone()

    def raw_grouped_expert_payload(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> DeepSeekV4FlashQuantizedExpertPayload:
        if (tensor.name, expert_id) not in self.raw_payloads:
            raise AttributeError("raw grouped expert payload unavailable")
        self.raw_payload_read_count += 1
        input_size, output_size, _expert_count = tensor.dims
        return DeepSeekV4FlashQuantizedExpertPayload(
            tensor_name=tensor.name,
            expert_id=expert_id,
            ggml_type=tensor.tensor_type,
            rows=output_size,
            columns=input_size,
            payload=memoryview(bytearray(self.raw_payloads[(tensor.name, expert_id)])),
        )


class _RecordingBackend:
    def __init__(self) -> None:
        self.sliding_attention_calls = 0
        self.routed_expert_ids: list[int] = []
        self.query_widths: list[int] = []

    def sliding_attention(
        self,
        *,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        attn_sinks: torch.Tensor | None,
        token_idx: int,
    ) -> torch.Tensor:
        assert query.is_cuda
        assert kv_rows.is_cuda
        assert attn_sinks is None or attn_sinks.is_cuda
        assert token_idx == 3
        self.sliding_attention_calls += 1
        self.query_widths.append(query.numel())
        return query

    def routed_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        assert hidden.is_cuda
        assert gate_weight.is_cuda
        assert up_weight.is_cuda
        assert down_weight.is_cuda
        self.routed_expert_ids.append(int(gate_weight[0, 0].item()))
        return torch.ones_like(hidden, dtype=torch.float32)


class _MarkerBackend(_RecordingBackend):
    def sliding_attention(
        self,
        *,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        attn_sinks: torch.Tensor | None,
        token_idx: int,
    ) -> torch.Tensor:
        self.sliding_attention_calls += 1
        self.query_widths.append(query.numel())
        return torch.zeros_like(query)

    def routed_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        marker = float(gate_weight[0, 0].item())
        self.routed_expert_ids.append(int(marker))
        return torch.full_like(hidden, marker, dtype=torch.float32)


class _RoutedExpertTestStager:
    def __init__(self, *, raw_available: bool = True) -> None:
        self.raw_available = raw_available
        self.raw_calls: list[tuple[str, int, int | None]] = []
        self.dense_calls: list[tuple[int, int | None]] = []
        self._cache_stats: dict[str, int] = {}

    def stage_grouped_expert_payload(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
        *,
        layer_idx: int | None = None,
    ) -> DeepSeekV4FlashStagedQuantizedExpertPayload:
        if not self.raw_available:
            raise AttributeError("raw grouped expert payload unavailable")
        self.raw_calls.append((tensor.name, expert_id, layer_idx))
        return DeepSeekV4FlashStagedQuantizedExpertPayload(
            tensor_name=tensor.name,
            expert_id=expert_id,
            ggml_type=tensor.tensor_type,
            rows=tensor.dims[1],
            columns=tensor.dims[0],
            payload=torch.tensor([expert_id], dtype=torch.uint8),
        )

    def stage_grouped_expert_payloads_for_ids(
        self,
        tensors: DeepSeekV4FlashGroupedExpertTensors,
        expert_ids: torch.Tensor,
        *,
        layer_idx: int | None = None,
    ) -> list[
        tuple[
            int,
            DeepSeekV4FlashStagedQuantizedExpertPayload,
            DeepSeekV4FlashStagedQuantizedExpertPayload,
            DeepSeekV4FlashStagedQuantizedExpertPayload,
        ]
    ]:
        self._cache_stats["batched_payload_stage_calls"] = (
            self._cache_stats.get("batched_payload_stage_calls", 0) + 1
        )
        selected_ids = [
            int(expert_id)
            for expert_id in expert_ids.detach().reshape(-1).to("cpu").tolist()
        ]
        return [
            (
                expert_id,
                self.stage_grouped_expert_payload(
                    tensors.gate,
                    expert_id,
                    layer_idx=layer_idx,
                ),
                self.stage_grouped_expert_payload(
                    tensors.up,
                    expert_id,
                    layer_idx=layer_idx,
                ),
                self.stage_grouped_expert_payload(
                    tensors.down,
                    expert_id,
                    layer_idx=layer_idx,
                ),
            )
            for expert_id in selected_ids
        ]

    def cache_stats(self) -> dict[str, int]:
        return dict(self._cache_stats)

    def stage_grouped_expert(
        self,
        tensors: DeepSeekV4FlashGroupedExpertTensors,
        expert_id: int,
        *,
        layer_idx: int | None = None,
    ):
        del tensors
        self.dense_calls.append((expert_id, layer_idx))
        marker = torch.full((2, 2), float(expert_id))
        return type(
            "_StagedExpert",
            (),
            {
                "expert_id": expert_id,
                "gate": marker,
                "up": marker,
                "down": marker,
            },
        )()


class _QuantizedRecordingBackend:
    def __init__(self, *, quantized_available: bool = True) -> None:
        self.quantized_available = quantized_available
        self.quantized_expert_ids: list[int] = []
        self.dense_expert_ids: list[int] = []

    def quantized_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_payload: DeepSeekV4FlashStagedQuantizedExpertPayload,
        up_payload: DeepSeekV4FlashStagedQuantizedExpertPayload,
        down_payload: DeepSeekV4FlashStagedQuantizedExpertPayload,
    ) -> torch.Tensor:
        del up_payload, down_payload
        if not self.quantized_available:
            raise NotImplementedError("quantized expert path unavailable")
        self.quantized_expert_ids.append(gate_payload.expert_id)
        return torch.full_like(hidden, float(gate_payload.expert_id))

    def routed_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        del up_weight, down_weight
        expert_id = int(gate_weight[0, 0])
        self.dense_expert_ids.append(expert_id)
        return torch.full_like(hidden, float(expert_id))


class _FusedQuantizedRecordingBackend(_QuantizedRecordingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.fused_calls = 0
        self.workspace_shapes: list[tuple[int, ...]] = []
        self.payload_stack_shapes: dict[str, tuple[int, ...] | None] = {}

    def fused_quantized_selected_experts_gemm(
        self,
        *,
        hidden: torch.Tensor,
        expert_weights: torch.Tensor,
        payloads: list[
            tuple[
                int,
                DeepSeekV4FlashStagedQuantizedExpertPayload,
                DeepSeekV4FlashStagedQuantizedExpertPayload,
                DeepSeekV4FlashStagedQuantizedExpertPayload,
            ]
        ],
        workspace: torch.Tensor,
        gate_stack: torch.Tensor | None = None,
        up_stack: torch.Tensor | None = None,
        down_stack: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del expert_weights
        self.fused_calls += 1
        self.workspace_shapes.append(tuple(workspace.shape))
        self.payload_stack_shapes = {
            "gate": tuple(gate_stack.shape) if gate_stack is not None else None,
            "up": tuple(up_stack.shape) if up_stack is not None else None,
            "down": tuple(down_stack.shape) if down_stack is not None else None,
        }
        return torch.full_like(hidden, float(sum(item[0] for item in payloads)))


def _tensor(name: str, dims: tuple[int, ...]) -> DeepSeekV4FlashTensor:
    return DeepSeekV4FlashTensor(
        name=name,
        dims=dims,
        tensor_type=GGML_TYPE_Q8_0,
        offset=0,
        nbytes=0,
    )


def _q8_payload(rows: list[tuple[float, tuple[int, ...]]]) -> bytes:
    payload = bytearray()
    for scale, values in rows:
        if len(values) != 32:
            raise ValueError("test Q8 rows must contain one 32-value block")
        payload.extend(struct.pack("<e", scale))
        payload.extend(struct.pack("<" + "b" * 32, *values))
    return bytes(payload)


def test_env_limited_top_k_caps_positive_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FASTINFERENCE_DEEPSEEK_V4_FLASH_ROUTER_TOP_K", "2")

    assert (
        _env_limited_top_k(
            6,
            env_name="FASTINFERENCE_DEEPSEEK_V4_FLASH_ROUTER_TOP_K",
        )
        == 2
    )


def test_env_limited_top_k_ignores_bad_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for value in ("0", "-1", "bad"):
        monkeypatch.setenv("FASTINFERENCE_DEEPSEEK_V4_FLASH_ROUTER_TOP_K", value)

        assert (
            _env_limited_top_k(
                6,
                env_name="FASTINFERENCE_DEEPSEEK_V4_FLASH_ROUTER_TOP_K",
            )
            == 6
        )


def _add_identity_layer_weights(
    store: _FakeLayerStore,
    tensors: dict[str, DeepSeekV4FlashTensor],
    *,
    hidden_size: int,
    num_experts: int,
) -> None:
    store.vectors[(tensors["attn_norm"].name, torch.float32)] = torch.ones(hidden_size)
    store.vectors[(tensors["ffn_norm"].name, torch.float32)] = torch.ones(hidden_size)
    store.matrices[tensors["attn_q"].name] = torch.eye(hidden_size)
    store.matrices[tensors["attn_out"].name] = torch.eye(hidden_size)
    store.matrices[tensors["router"].name] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    for expert_id in range(num_experts):
        gate = torch.eye(hidden_size)
        gate[0, 0] = float(expert_id)
        store.expert_matrices[(tensors["gate"].name, expert_id)] = gate
        store.expert_matrices[(tensors["up"].name, expert_id)] = torch.eye(hidden_size)
        store.expert_matrices[(tensors["down"].name, expert_id)] = torch.eye(
            hidden_size
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_q8_tensor_projection_uses_raw_payload_without_stage_matrix() -> None:
    store = _FakeLayerStore()
    tensor = _tensor("blk.0.attn_output_a.weight", (32, 2))
    store.payloads[tensor.name] = _q8_payload(
        [
            (0.5, tuple([2] + [0] * 31)),
            (0.25, tuple([0, 4] + [0] * 30)),
        ]
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    stager.profiler = DeepSeekV4FlashProfiler(enabled=True)
    hidden = torch.zeros((32,), dtype=torch.float32, device="cuda")
    hidden[0] = 3.0
    hidden[1] = 5.0

    first = deepseek_v4_flash_q8_0_tensor_projection(hidden, tensor, stager)
    second = deepseek_v4_flash_q8_0_tensor_projection(hidden, tensor, stager)

    assert store.decode_count == 0
    assert stager.cache_stats()["dynamic_misses"] == 1
    assert stager.cache_stats()["dynamic_hits"] == 1
    assert [event["name"] for event in stager.profiler.to_dict()["events"]] == [
        "stage_q8_raw",
        "stage_q8_raw",
    ]
    torch.testing.assert_close(
        first,
        torch.tensor([3.0, 5.0], dtype=torch.float32, device="cuda"),
    )
    torch.testing.assert_close(second, first)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_staged_shared_expert_raw_q8_path_applies_deepseek_clamp() -> None:
    store = _FakeLayerStore()
    gate = _tensor("shared.gate", (32, 32))
    up = _tensor("shared.up", (32, 32))
    down = _tensor("shared.down", (32, 32))
    store.payloads[gate.name] = _q8_payload(
        [(0.5, tuple([40] + [0] * 31))] + [(1.0, tuple([0] * 32))] * 31
    )
    store.payloads[up.name] = _q8_payload(
        [(0.5, tuple([40] + [0] * 31))] + [(1.0, tuple([0] * 32))] * 31
    )
    store.payloads[down.name] = _q8_payload(
        [(1.0, tuple([1] + [0] * 31))] + [(1.0, tuple([0] * 32))] * 31
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    hidden = torch.zeros((32,), dtype=torch.float32, device="cuda")
    hidden[0] = 1.0
    grouped = DeepSeekV4FlashGroupedExpertTensors(gate=gate, up=up, down=down)

    output = _run_staged_shared_expert(
        hidden,
        grouped,
        stager=stager,
        backend=_RecordingBackend(),
    )

    expected = torch.zeros_like(output)
    expected[0] = torch.nn.functional.silu(torch.tensor(10.0, device="cuda")) * 10.0
    torch.testing.assert_close(output, expected, rtol=1.0e-4, atol=1.0e-4)


def test_staged_routed_experts_prefers_quantized_payload_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_item(_tensor: torch.Tensor) -> int:
        raise AssertionError("expert routing must not call Tensor.item() per expert")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("gate", (2, 2, 4)),
        up=_tensor("up", (2, 2, 4)),
        down=_tensor("down", (2, 2, 4)),
    )
    stager = _RoutedExpertTestStager()
    backend = _QuantizedRecordingBackend()

    output = _run_staged_routed_experts(
        torch.ones(2),
        torch.tensor([3, 1], dtype=torch.int64),
        torch.tensor([0.25, 0.75], dtype=torch.float32),
        grouped_experts=grouped,
        stager=stager,
        backend=backend,
        layer_idx=7,
    )

    assert backend.quantized_expert_ids == [3, 1]
    assert backend.dense_expert_ids == []
    assert stager.dense_calls == []
    assert stager.raw_calls == [
        ("gate", 3, 7),
        ("up", 3, 7),
        ("down", 3, 7),
        ("gate", 1, 7),
        ("up", 1, 7),
        ("down", 1, 7),
    ]
    torch.testing.assert_close(output, torch.full((2,), 1.5))


def test_staged_routed_experts_prefers_fused_selected_path_without_state() -> None:
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=replace(_tensor("gate", (2, 4, 4)), tensor_type=GGML_TYPE_IQ2_XXS),
        up=replace(_tensor("up", (2, 4, 4)), tensor_type=GGML_TYPE_IQ2_XXS),
        down=replace(_tensor("down", (4, 2, 4)), tensor_type=GGML_TYPE_Q2_K),
    )
    stager = _RoutedExpertTestStager()
    backend = _FusedQuantizedRecordingBackend()

    output = _run_staged_routed_experts(
        torch.ones(2),
        torch.tensor([3, 1], dtype=torch.int64),
        torch.tensor([0.25, 0.75], dtype=torch.float32),
        grouped_experts=grouped,
        stager=stager,
        backend=backend,
        layer_idx=7,
    )

    assert backend.fused_calls == 1
    assert backend.quantized_expert_ids == []
    assert backend.workspace_shapes == [(2, 4)]
    torch.testing.assert_close(output, torch.full((2,), 4.0))


def test_staged_routed_experts_uses_direct_payload_path_with_state() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=replace(_tensor("gate", (2, 4, 4)), tensor_type=GGML_TYPE_IQ2_XXS),
        up=replace(_tensor("up", (2, 4, 4)), tensor_type=GGML_TYPE_IQ2_XXS),
        down=replace(_tensor("down", (4, 2, 4)), tensor_type=GGML_TYPE_Q2_K),
    )
    stager = _RoutedExpertTestStager()
    backend = _FusedQuantizedRecordingBackend()
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=8,
            hidden_size=2,
            device=device,
        )
    )

    output = _run_staged_routed_experts(
        torch.ones(2, device=device),
        torch.tensor([3, 1], dtype=torch.int64, device=device),
        torch.tensor([0.25, 0.75], dtype=torch.float32, device=device),
        grouped_experts=grouped,
        stager=stager,
        backend=backend,
        state=state,
        layer_idx=7,
    )

    assert backend.fused_calls == 1
    assert backend.payload_stack_shapes == {
        "gate": None,
        "up": None,
        "down": None,
    }
    torch.testing.assert_close(output, torch.full((2,), 4.0, device=device))


def test_staged_routed_experts_profiles_fused_selected_path() -> None:
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=replace(_tensor("gate", (2, 4, 4)), tensor_type=GGML_TYPE_IQ2_XXS),
        up=replace(_tensor("up", (2, 4, 4)), tensor_type=GGML_TYPE_IQ2_XXS),
        down=replace(_tensor("down", (4, 2, 4)), tensor_type=GGML_TYPE_Q2_K),
    )
    stager = _RoutedExpertTestStager()
    stager.profiler = DeepSeekV4FlashProfiler(enabled=True)

    _run_staged_routed_experts(
        torch.ones(2),
        torch.tensor([3, 1], dtype=torch.int64),
        torch.tensor([0.25, 0.75], dtype=torch.float32),
        grouped_experts=grouped,
        stager=stager,
        backend=_FusedQuantizedRecordingBackend(),
        layer_idx=7,
    )

    events = stager.profiler.to_dict()["events"]
    kernel_events = [
        event for event in events if event["name"] == "router_selected_experts_kernel"
    ]
    assert len(kernel_events) == 1
    assert kernel_events[0]["metadata"] == {"layer_idx": 7, "expert_count": 2}


def test_staged_routed_experts_reuses_grouped_payload_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_to = torch.Tensor.to

    def fake_cuda_to(
        tensor: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        device = kwargs.get("device")
        if device is None and args and isinstance(args[0], torch.device | str):
            device = args[0]
        is_cuda_target = device is not None and torch.device(device).type == "cuda"
        if is_cuda_target:
            filtered_kwargs = dict(kwargs)
            filtered_kwargs.pop("device", None)
            filtered_args = args[1:] if args and device == args[0] else args
            return original_to(tensor, *filtered_args, **filtered_kwargs).clone()
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_cuda_to)
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("gate", (2, 2, 4)),
        up=_tensor("up", (2, 2, 4)),
        down=_tensor("down", (2, 2, 4)),
    )
    store = _FakeLayerStore()
    for tensor in (grouped.gate, grouped.up, grouped.down):
        for expert_id in (3, 1):
            store.raw_payloads[(tensor.name, expert_id)] = bytes(
                [expert_id, len(tensor.name) % 251]
            )
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=1 << 20,
    )
    backend = _QuantizedRecordingBackend()
    expert_ids = torch.tensor([3, 1], dtype=torch.int64)
    expert_weights = torch.tensor([0.25, 0.75], dtype=torch.float32)

    first = _run_staged_routed_experts(
        torch.ones(2),
        expert_ids,
        expert_weights,
        grouped_experts=grouped,
        stager=stager,
        backend=backend,
        layer_idx=7,
    )
    second = _run_staged_routed_experts(
        torch.ones(2),
        expert_ids,
        expert_weights,
        grouped_experts=grouped,
        stager=stager,
        backend=backend,
        layer_idx=7,
    )

    stats = stager.cache_stats()
    assert stats.get("batched_payload_stage_calls", 0) == 2
    assert stats.get("grouped_misses", 0) == 6
    assert stats.get("grouped_hits", 0) == 6
    assert "selected_payload_cache_hits" not in stats
    assert store.raw_payload_read_count == 6
    torch.testing.assert_close(first, torch.full((2,), 1.5))
    torch.testing.assert_close(second, first)


def test_staged_routed_experts_profiles_stage_and_kernel_sections() -> None:
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("gate", (2, 2, 4)),
        up=_tensor("up", (2, 2, 4)),
        down=_tensor("down", (2, 2, 4)),
    )
    stager = _RoutedExpertTestStager()
    stager.profiler = DeepSeekV4FlashProfiler(enabled=True)
    backend = _QuantizedRecordingBackend()

    _run_staged_routed_experts(
        torch.ones(2),
        torch.tensor([3, 1], dtype=torch.int64),
        torch.tensor([0.25, 0.75], dtype=torch.float32),
        grouped_experts=grouped,
        stager=stager,
        backend=backend,
        layer_idx=7,
    )

    events = stager.profiler.to_dict()["events"]
    event_names = [event["name"] for event in events]

    assert event_names == [
        "router_expert_stage",
        "router_expert_kernel",
        "router_expert_kernel",
    ]
    assert events[0]["metadata"] == {"layer_idx": 7, "expert_id": -1}
    assert events[1]["metadata"] == {"layer_idx": 7, "expert_id": 3}
    assert events[2]["metadata"] == {"layer_idx": 7, "expert_id": 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_staged_routed_experts_materializes_only_bounded_cuda_expert_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_cpu(_tensor: torch.Tensor) -> torch.Tensor:
        raise AssertionError("expert routing must not call Tensor.cpu()")

    monkeypatch.setattr(torch.Tensor, "cpu", fail_cpu)
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("gate", (2, 2, 4)),
        up=_tensor("up", (2, 2, 4)),
        down=_tensor("down", (2, 2, 4)),
    )
    stager = _RoutedExpertTestStager()
    backend = _QuantizedRecordingBackend()

    output = _run_staged_routed_experts(
        torch.ones(2, device="cuda"),
        torch.tensor([3, 1], dtype=torch.int64, device="cuda"),
        torch.tensor([0.25, 0.75], dtype=torch.float32, device="cuda"),
        grouped_experts=grouped,
        stager=stager,
        backend=backend,
        layer_idx=7,
    )

    assert backend.quantized_expert_ids == [3, 1]
    assert stager.cache_stats().get("batched_payload_stage_calls", 0) == 1
    assert stager.cache_stats().get("routed_expert_id_materializations", 0) == 0
    torch.testing.assert_close(output.to("cpu"), torch.full((2,), 1.5))


def test_hash_routed_experts_reuses_staged_token_table() -> None:
    token_table = _tensor("token_to_experts", (2, 8))
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("gate", (2, 2, 4)),
        up=_tensor("up", (2, 2, 4)),
        down=_tensor("down", (2, 2, 4)),
    )
    router = _tensor("router", (2, 4))
    layer = DeepSeekV4FlashLayerSemanticBindings(
        layer_index=5,
        router=router,
        grouped_experts=grouped,
        expert_token_to_expert_ids=token_table,
    )

    class FakeStore:
        def __init__(self) -> None:
            self.tensor_reads = 0
            self.matrices = {router.name: torch.zeros(4, 2)}

        def tensor_to_torch(
            self,
            tensor: DeepSeekV4FlashTensor,
            *,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            assert tensor is token_table
            assert dtype is torch.int32
            self.tensor_reads += 1
            table = torch.full((2, 8), -1, dtype=torch.int32)
            table[:, 3] = torch.tensor([2, 0], dtype=torch.int32)
            return table

    class FakeStager(_RoutedExpertTestStager):
        def __init__(self) -> None:
            super().__init__()
            self.store = FakeStore()
            self.device = torch.device("cpu")
            self._dynamic_cache: dict[object, torch.Tensor] = {}
            self._cache_entry_bytes: dict[object, int] = {}
            self._staged_bytes = 0

        def _dynamic_cache_key(
            self,
            tensor: DeepSeekV4FlashTensor,
            *,
            dtype: torch.dtype,
            extra: tuple[int | str, ...],
        ) -> tuple[str, torch.dtype, tuple[int | str, ...]]:
            return tensor.name, dtype, extra

        def _prepare_cache_insert(self, nbytes: int) -> bool:
            assert nbytes >= 0
            return True

        def record_cache_hit(self, cache_name: str, *, tensor_name: str) -> None:
            self._cache_stats[f"{cache_name}_hits"] = (
                self._cache_stats.get(f"{cache_name}_hits", 0) + 1
            )

        def record_cache_miss(
            self,
            cache_name: str,
            loaded_bytes: int,
            *,
            tensor_name: str,
        ) -> None:
            assert loaded_bytes > 0
            self._cache_stats[f"{cache_name}_misses"] = (
                self._cache_stats.get(f"{cache_name}_misses", 0) + 1
            )

        def _register_cached_entry(
            self,
            cache_key: object,
            tensor: torch.Tensor,
            nbytes: int,
        ) -> None:
            self._dynamic_cache[cache_key] = tensor
            self._cache_entry_bytes[cache_key] = nbytes
            self._staged_bytes += nbytes

        def stage_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
            return self.store.matrices[tensor.name]

    stager = FakeStager()
    backend = _QuantizedRecordingBackend()

    first = _run_hash_routed_experts(
        torch.ones(2),
        layer=layer,
        grouped_experts=grouped,
        stager=stager,
        backend=backend,
        token_id=3,
    )
    second = _run_hash_routed_experts(
        torch.ones(2),
        layer=layer,
        grouped_experts=grouped,
        stager=stager,
        backend=backend,
        token_id=3,
    )

    assert stager.store.tensor_reads == 1
    assert stager._cache_stats["dynamic_misses"] == 1
    assert stager._cache_stats["dynamic_hits"] == 1
    torch.testing.assert_close(first, torch.full((2,), 1.5))
    torch.testing.assert_close(second, torch.full((2,), 1.5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_hash_routed_experts_keeps_batch_one_token_table_on_cpu() -> None:
    token_table = _tensor("token_to_experts", (2, 8))
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("gate", (2, 2, 4)),
        up=_tensor("up", (2, 2, 4)),
        down=_tensor("down", (2, 2, 4)),
    )
    layer = DeepSeekV4FlashLayerSemanticBindings(
        layer_index=5,
        grouped_experts=grouped,
        expert_token_to_expert_ids=token_table,
    )

    class FakeStore:
        def __init__(self) -> None:
            self.tensor_reads = 0

        def tensor_to_torch(
            self,
            tensor: DeepSeekV4FlashTensor,
            *,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            assert tensor is token_table
            assert dtype is torch.int32
            self.tensor_reads += 1
            table = torch.full((2, 8), -1, dtype=torch.int32)
            table[:, 3] = torch.tensor([2, 0], dtype=torch.int32)
            return table

    class FakeStager(_RoutedExpertTestStager):
        def __init__(self) -> None:
            super().__init__()
            self.store = FakeStore()
            self.device = torch.device("cuda")
            self._dynamic_cache: dict[object, torch.Tensor] = {}
            self._cache_entry_bytes: dict[object, int] = {}
            self._staged_bytes = 0

        def _dynamic_cache_key(
            self,
            tensor: DeepSeekV4FlashTensor,
            *,
            dtype: torch.dtype,
            extra: tuple[int | str, ...],
        ) -> tuple[str, torch.dtype, tuple[int | str, ...]]:
            return tensor.name, dtype, extra

        def _prepare_cache_insert(self, nbytes: int) -> bool:
            assert nbytes > 0
            return True

        def record_cache_hit(self, cache_name: str, *, tensor_name: str) -> None:
            self._cache_stats[f"{cache_name}_hits"] = (
                self._cache_stats.get(f"{cache_name}_hits", 0) + 1
            )

        def record_cache_miss(
            self,
            cache_name: str,
            loaded_bytes: int,
            *,
            tensor_name: str,
        ) -> None:
            assert loaded_bytes > 0
            self._cache_stats[f"{cache_name}_misses"] = (
                self._cache_stats.get(f"{cache_name}_misses", 0) + 1
            )

        def _register_cached_entry(
            self,
            cache_key: object,
            tensor: torch.Tensor,
            nbytes: int,
        ) -> None:
            self._dynamic_cache[cache_key] = tensor
            self._cache_entry_bytes[cache_key] = nbytes
            self._staged_bytes += nbytes

    stager = FakeStager()
    output = _run_hash_routed_experts(
        torch.ones(2, device="cuda"),
        layer=layer,
        grouped_experts=grouped,
        stager=stager,
        backend=_QuantizedRecordingBackend(),
        token_id=3,
    )

    cached_tables = list(stager._dynamic_cache.values())
    assert stager.store.tensor_reads == 1
    assert cached_tables
    assert all(table.device.type == "cpu" for table in cached_tables)
    torch.testing.assert_close(output.to("cpu"), torch.full((2,), 1.5))


def test_hash_routed_experts_uses_router_weights_for_selected_experts() -> None:
    token_table = _tensor("token_to_experts", (2, 8))
    router = _tensor("router", (2, 4))
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("gate", (2, 2, 4)),
        up=_tensor("up", (2, 2, 4)),
        down=_tensor("down", (2, 2, 4)),
    )
    layer = DeepSeekV4FlashLayerSemanticBindings(
        layer_index=5,
        router=router,
        grouped_experts=grouped,
        expert_token_to_expert_ids=token_table,
    )

    class FakeStore:
        def tensor_to_torch(
            self,
            tensor: DeepSeekV4FlashTensor,
            *,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            del tensor
            assert dtype is torch.int32
            table = torch.full((2, 8), -1, dtype=torch.int32)
            table[:, 3] = torch.tensor([2, 0], dtype=torch.int32)
            return table

    class FakeStager(_RoutedExpertTestStager):
        def __init__(self) -> None:
            super().__init__()
            self.store = FakeStore()
            self.device = torch.device("cpu")
            self._dynamic_cache: dict[object, torch.Tensor] = {}
            self._cache_entry_bytes: dict[object, int] = {}
            self._staged_bytes = 0

        def _dynamic_cache_key(
            self,
            tensor: DeepSeekV4FlashTensor,
            *,
            dtype: torch.dtype,
            extra: tuple[int | str, ...],
        ) -> tuple[str, torch.dtype, tuple[int | str, ...]]:
            return tensor.name, dtype, extra

        def _prepare_cache_insert(self, nbytes: int) -> bool:
            del nbytes
            return True

        def record_cache_hit(self, cache_name: str, *, tensor_name: str) -> None:
            del cache_name, tensor_name

        def record_cache_miss(
            self,
            cache_name: str,
            loaded_bytes: int,
            *,
            tensor_name: str,
        ) -> None:
            del cache_name, loaded_bytes, tensor_name

        def _register_cached_entry(
            self,
            cache_key: object,
            tensor: torch.Tensor,
            nbytes: int,
        ) -> None:
            del nbytes
            self._dynamic_cache[cache_key] = tensor

        def stage_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
            assert tensor is router
            return torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [4.0, 0.0],
                    [0.0, 0.0],
                ],
                dtype=torch.float32,
            )

    hidden = torch.tensor([1.0, 0.0])
    output = _run_hash_routed_experts(
        hidden,
        layer=layer,
        grouped_experts=grouped,
        stager=FakeStager(),
        backend=_QuantizedRecordingBackend(),
        token_id=3,
    )

    scores = torch.nn.functional.softplus(torch.tensor([4.0, 0.0])).sqrt()
    expected_expert_2_weight = scores[0] / scores.sum() * 1.5
    torch.testing.assert_close(output, torch.full((2,), 2.0 * expected_expert_2_weight))


def test_staged_routed_experts_falls_back_to_dense_when_raw_unavailable() -> None:
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("gate", (2, 2, 2)),
        up=_tensor("up", (2, 2, 2)),
        down=_tensor("down", (2, 2, 2)),
    )
    stager = _RoutedExpertTestStager(raw_available=False)
    backend = _QuantizedRecordingBackend()

    output = _run_staged_routed_experts(
        torch.ones(2),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0.5, 0.5], dtype=torch.float32),
        grouped_experts=grouped,
        stager=stager,
        backend=backend,
        layer_idx=11,
    )

    assert backend.quantized_expert_ids == []
    assert backend.dense_expert_ids == [0, 1]
    assert stager.dense_calls == [(0, 11), (1, 11)]
    torch.testing.assert_close(output, torch.full((2,), 0.5))


def test_staged_routed_experts_falls_back_when_quantized_backend_missing() -> None:
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("gate", (2, 2, 2)),
        up=_tensor("up", (2, 2, 2)),
        down=_tensor("down", (2, 2, 2)),
    )
    stager = _RoutedExpertTestStager()
    backend = _QuantizedRecordingBackend(quantized_available=False)

    output = _run_staged_routed_experts(
        torch.ones(2),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0.25, 0.75], dtype=torch.float32),
        grouped_experts=grouped,
        stager=stager,
        backend=backend,
        layer_idx=13,
    )

    assert backend.quantized_expert_ids == []
    assert backend.dense_expert_ids == [0, 1]
    assert stager.dense_calls == [(0, 13), (1, 13)]
    torch.testing.assert_close(output, torch.full((2,), 0.75))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_sliding_layer_forward_uses_staged_attention_and_selected_experts() -> None:
    hidden_size = 4
    num_experts = 3
    store = _FakeLayerStore()
    tensors = {
        "attn_norm": _tensor("blk.0.attn_norm.weight", (hidden_size,)),
        "attn_q": _tensor("blk.0.attn_q.weight", (hidden_size, hidden_size)),
        "attn_out": _tensor("blk.0.attn_output.weight", (hidden_size, hidden_size)),
        "ffn_norm": _tensor("blk.0.ffn_norm.weight", (hidden_size,)),
        "router": _tensor("blk.0.ffn_gate_inp.weight", (hidden_size, num_experts)),
        "gate": _tensor("blk.0.ffn_gate_exps.weight", (hidden_size, hidden_size, 3)),
        "up": _tensor("blk.0.ffn_up_exps.weight", (hidden_size, hidden_size, 3)),
        "down": _tensor("blk.0.ffn_down_exps.weight", (hidden_size, hidden_size, 3)),
    }
    _add_identity_layer_weights(
        store,
        tensors,
        hidden_size=hidden_size,
        num_experts=num_experts,
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
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    backend = _RecordingBackend()
    hidden = torch.tensor([1.0, 2.0, 0.0, 0.0], device="cuda")

    output = deepseek_v4_flash_sliding_layer_forward(
        hidden,
        layer=layer,
        stager=stager,
        backend=backend,
        token_idx=3,
        router_top_k=2,
    )

    assert output.device.type == "cuda"
    assert output.shape == hidden.shape
    assert backend.sliding_attention_calls == 1
    assert backend.routed_expert_ids == [1, 0]
    torch.testing.assert_close(
        deepseek_v4_flash_staged_matrix_projection(
            torch.tensor([1.0, 2.0], device="cuda"),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda"),
        ),
        torch.tensor([5.0, 11.0], device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_sliding_layer_forward_uses_two_stage_attention_output_projection() -> None:
    hidden_size = 4
    num_experts = 3
    store = _FakeLayerStore()
    tensors = {
        "attn_norm": _tensor("blk.0.attn_norm.weight", (hidden_size,)),
        "attn_q": _tensor("blk.0.attn_q.weight", (hidden_size, hidden_size)),
        "attn_q_b": _tensor("blk.0.attn_q_b.weight", (hidden_size, 8)),
        "attn_out_a": _tensor("blk.0.attn_output_a.weight", (1, 8)),
        "attn_out_b": _tensor("blk.0.attn_output_b.weight", (8, hidden_size)),
        "ffn_norm": _tensor("blk.0.ffn_norm.weight", (hidden_size,)),
        "router": _tensor("blk.0.ffn_gate_inp.weight", (hidden_size, num_experts)),
        "gate": _tensor("blk.0.ffn_gate_exps.weight", (hidden_size, hidden_size, 3)),
        "up": _tensor("blk.0.ffn_up_exps.weight", (hidden_size, hidden_size, 3)),
        "down": _tensor("blk.0.ffn_down_exps.weight", (hidden_size, hidden_size, 3)),
    }
    _add_identity_layer_weights(
        store,
        {
            **tensors,
            "attn_out": tensors["attn_out_a"],
        },
        hidden_size=hidden_size,
        num_experts=num_experts,
    )
    store.matrices[tensors["attn_q_b"].name] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    store.matrices[tensors["attn_out_a"].name] = torch.ones((8, 1))
    store.matrices[tensors["attn_out_b"].name] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    layer = DeepSeekV4FlashLayerSemanticBindings(
        layer_index=0,
        attention_norm=tensors["attn_norm"],
        attention_query=tensors["attn_q"],
        attention_query_b=tensors["attn_q_b"],
        attention_output=tensors["attn_out_a"],
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
    backend = _MarkerBackend()

    output = deepseek_v4_flash_sliding_layer_forward(
        torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda"),
        layer=layer,
        stager=DeepSeekV4FlashGPUWeightStager(store, device="cuda"),
        backend=backend,
        token_idx=3,
        router_top_k=1,
    )

    assert output.shape == (hidden_size,)
    assert backend.query_widths == [8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_router_topk_applies_reference_routed_scaling_factor() -> None:
    _expert_ids, weights = deepseek_v4_flash_router_topk(
        torch.tensor([1.0, 2.0], device="cuda"),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda"),
        top_k=2,
    )

    torch.testing.assert_close(weights.sum(), torch.tensor(1.5, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_sliding_layer_forward_hash_routes_and_adds_shared_expert() -> None:
    hidden_size = 4
    store = _FakeLayerStore()
    tensors = {
        "attn_norm": _tensor("blk.0.attn_norm.weight", (hidden_size,)),
        "attn_q": _tensor("blk.0.attn_q.weight", (hidden_size, hidden_size)),
        "attn_out": _tensor("blk.0.attn_output.weight", (hidden_size, hidden_size)),
        "ffn_norm": _tensor("blk.0.ffn_norm.weight", (hidden_size,)),
        "router": _tensor("blk.0.ffn_gate_inp.weight", (hidden_size, 3)),
        "gate": _tensor("blk.0.ffn_gate_exps.weight", (hidden_size, hidden_size, 3)),
        "up": _tensor("blk.0.ffn_up_exps.weight", (hidden_size, hidden_size, 3)),
        "down": _tensor("blk.0.ffn_down_exps.weight", (hidden_size, hidden_size, 3)),
        "shared_gate": _tensor(
            "blk.0.ffn_gate_shexp.weight",
            (hidden_size, hidden_size),
        ),
        "shared_up": _tensor("blk.0.ffn_up_shexp.weight", (hidden_size, hidden_size)),
        "shared_down": _tensor(
            "blk.0.ffn_down_shexp.weight",
            (hidden_size, hidden_size),
        ),
        "token_to_experts": _tensor("blk.0.ffn_gate_tid2eid.weight", (2, 5)),
    }
    _add_identity_layer_weights(store, tensors, hidden_size=hidden_size, num_experts=3)
    for name, marker in (
        ("shared_gate", 10.0),
        ("shared_up", 1.0),
        ("shared_down", 1.0),
    ):
        matrix = torch.eye(hidden_size)
        matrix[0, 0] = marker
        store.matrices[tensors[name].name] = matrix
    store.vectors[(tensors["token_to_experts"].name, torch.int32)] = torch.tensor(
        [[0, 1, 2, 0, 2], [1, 2, 0, 1, 0]],
        dtype=torch.int32,
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
        shared_experts=DeepSeekV4FlashGroupedExpertTensors(
            gate=tensors["shared_gate"],
            up=tensors["shared_up"],
            down=tensors["shared_down"],
        ),
        expert_token_to_expert_ids=tensors["token_to_experts"],
    )
    backend = _MarkerBackend()

    output = deepseek_v4_flash_sliding_layer_forward(
        torch.zeros(hidden_size, device="cuda"),
        layer=layer,
        stager=DeepSeekV4FlashGPUWeightStager(store, device="cuda"),
        backend=backend,
        token_idx=3,
        token_id=4,
    )

    assert backend.routed_expert_ids == [2, 0, 10]
    torch.testing.assert_close(output, torch.full((hidden_size,), 11.5, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_sliding_layer_forward_expands_1d_hyper_connection_streams() -> None:
    hidden_size = 4
    store = _FakeLayerStore()
    tensors = {
        "attn_norm": _tensor("blk.0.attn_norm.weight", (hidden_size,)),
        "attn_q": _tensor("blk.0.attn_q.weight", (hidden_size, hidden_size)),
        "attn_out": _tensor("blk.0.attn_output.weight", (hidden_size, hidden_size)),
        "ffn_norm": _tensor("blk.0.ffn_norm.weight", (hidden_size,)),
        "router": _tensor("blk.0.ffn_gate_inp.weight", (hidden_size, 3)),
        "gate": _tensor("blk.0.ffn_gate_exps.weight", (hidden_size, hidden_size, 3)),
        "up": _tensor("blk.0.ffn_up_exps.weight", (hidden_size, hidden_size, 3)),
        "down": _tensor("blk.0.ffn_down_exps.weight", (hidden_size, hidden_size, 3)),
        "hc_fn": _tensor("blk.0.hc_attn_fn.weight", (16, 24)),
        "hc_base": _tensor("blk.0.hc_attn_base.weight", (24,)),
        "hc_scale": _tensor("blk.0.hc_attn_scale.weight", (3,)),
    }
    _add_identity_layer_weights(store, tensors, hidden_size=hidden_size, num_experts=3)
    store.matrices[tensors["hc_fn"].name] = torch.zeros(24, 16)
    store.vectors[(tensors["hc_base"].name, torch.float32)] = torch.zeros(24)
    store.vectors[(tensors["hc_scale"].name, torch.float32)] = torch.ones(3)
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
        attention_hyper_connection=DeepSeekV4FlashHyperConnectionTensors(
            fn=tensors["hc_fn"],
            base=tensors["hc_base"],
            scale=tensors["hc_scale"],
        ),
    )

    output = deepseek_v4_flash_sliding_layer_forward(
        torch.zeros(hidden_size, device="cuda"),
        layer=layer,
        stager=DeepSeekV4FlashGPUWeightStager(store, device="cuda"),
        backend=_MarkerBackend(),
        token_idx=3,
        router_top_k=1,
    )

    assert output.device.type == "cuda"
    assert output.shape == (4, hidden_size)
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.skipif(
    os.environ.get("RUN_DEEPSEEK_REAL_GGUF_LAYER") != "1",
    reason="real GGUF layer smoke is opt-in",
)
@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target GGUF not downloaded")
def test_real_gguf_layer0_sliding_forward_smoke() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        layer = replace(
            store.bindings.layers[0],
            attention_hyper_connection=None,
            ffn_hyper_connection=None,
        )
        stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
        hidden = torch.zeros(4096, dtype=torch.float32, device="cuda")
        hidden[0] = 1.0

        output = deepseek_v4_flash_sliding_layer_forward(
            hidden,
            layer=layer,
            stager=stager,
            backend=DeepSeekV4FlashGPUBackend(),
            token_idx=0,
        )

    assert output.device.type == "cuda"
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.skipif(
    os.environ.get("RUN_DEEPSEEK_REAL_GGUF_LAYER") != "1",
    reason="real GGUF layer smoke is opt-in",
)
@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target GGUF not downloaded")
def test_real_gguf_layer0_reports_unsupported_1d_hyper_connection() -> None:
    with (
        open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store,
        pytest.raises(ValueError, match="1-D hyper-connection"),
    ):
        deepseek_v4_flash_sliding_layer_forward(
            torch.zeros(4096, dtype=torch.float32, device="cuda"),
            layer=store.bindings.layers[0],
            stager=DeepSeekV4FlashGPUWeightStager(store, device="cuda"),
            backend=DeepSeekV4FlashGPUBackend(),
            token_idx=0,
        )
