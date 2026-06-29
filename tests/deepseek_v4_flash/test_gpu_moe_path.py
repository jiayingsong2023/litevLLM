from __future__ import annotations

import pytest
import torch

from vllm.kernels.triton.deepseek_v4_flash.moe import (
    DeepSeekV4MoEKernelInputs,
    deepseek_v4_moe,
)
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
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashGroupedExpertTensors,
    DeepSeekV4FlashQuantizedExpertPayload,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_deepseek_gpu_moe_accumulates_selected_experts_on_device() -> None:
    device = torch.device("cuda")
    hidden = torch.tensor([1.0, -2.0], device=device)
    expert_ids = torch.tensor([0, 2], dtype=torch.int32, device=device)
    expert_weights = torch.tensor([0.25, 0.75], dtype=torch.float32, device=device)
    expert_outputs = torch.tensor(
        [
            [2.0, 4.0],
            [100.0, 100.0],
            [-1.0, 3.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    output = deepseek_v4_moe(
        DeepSeekV4MoEKernelInputs(
            hidden=hidden,
            expert_ids=expert_ids,
            expert_weights=expert_weights,
            expert_outputs=expert_outputs,
        )
    )

    expected = 0.25 * expert_outputs[0] + 0.75 * expert_outputs[2]
    assert output.device.type == "cuda"
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_backend_moe_requires_gpu_tensors() -> None:
    backend = DeepSeekV4FlashGPUBackend()

    with pytest.raises(ValueError, match="must be CUDA tensors"):
        backend.routed_moe(
            hidden=torch.zeros(2),
            expert_ids=torch.tensor([0], dtype=torch.int32),
            expert_weights=torch.tensor([1.0]),
            expert_outputs=torch.zeros(1, 2),
        )


class _FakeGroupedExpertStore:
    def __init__(self) -> None:
        self.raw_payloads: dict[tuple[str, int], bytes] = {}

    def decode_grouped_expert_matrix(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> torch.Tensor:
        raise AssertionError(f"unexpected matrix decode for {tensor.name}")

    def raw_grouped_expert_payload(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> DeepSeekV4FlashQuantizedExpertPayload:
        input_size, output_size, _expert_count = tensor.dims
        payload = self.raw_payloads[(tensor.name, expert_id)]
        return DeepSeekV4FlashQuantizedExpertPayload(
            tensor_name=tensor.name,
            expert_id=expert_id,
            ggml_type=tensor.tensor_type,
            rows=output_size,
            columns=input_size,
            payload=memoryview(bytearray(payload)),
        )

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        raise AssertionError(f"unexpected matrix decode for {tensor.name}")

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise AssertionError(f"unexpected vector read for {tensor.name} as {dtype}")

    def tensor_payload(self, tensor: DeepSeekV4FlashTensor) -> memoryview:
        raise AssertionError(f"unexpected tensor payload read for {tensor.name}")


def _tensor(name: str, dims: tuple[int, ...] = (2, 2, 1)) -> DeepSeekV4FlashTensor:
    return DeepSeekV4FlashTensor(
        name=name,
        dims=dims,
        tensor_type=GGML_TYPE_Q8_0,
        offset=0,
        nbytes=0,
    )


def _patch_fake_cuda_to(monkeypatch: pytest.MonkeyPatch) -> None:
    original_to = torch.Tensor.to

    def fake_cuda_to(
        tensor: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        device = kwargs.get("device")
        if device is None and args and isinstance(args[0], torch.device | str):
            device = args[0]
        if torch.device(device).type == "cuda" if device is not None else False:
            filtered_kwargs = dict(kwargs)
            filtered_kwargs.pop("device", None)
            filtered_args = args[1:] if args and device == args[0] else args
            return original_to(tensor, *filtered_args, **filtered_kwargs).clone()
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_cuda_to)


def test_backend_stage_selected_expert_payloads_reuses_cached_payload_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_cuda_to(monkeypatch)
    store = _FakeGroupedExpertStore()
    grouped = DeepSeekV4FlashGroupedExpertTensors(
        gate=_tensor("blk.2.ffn_gate_exps.weight", dims=(2, 2, 8)),
        up=_tensor("blk.2.ffn_up_exps.weight", dims=(2, 2, 8)),
        down=_tensor("blk.2.ffn_down_exps.weight", dims=(2, 2, 8)),
    )
    for tensor in (grouped.gate, grouped.up, grouped.down):
        for expert_id in (1, 3, 5):
            store.raw_payloads[(tensor.name, expert_id)] = bytes(
                [expert_id, len(tensor.name) % 251]
            )
    stager = DeepSeekV4FlashGPUWeightStager(
        store,
        device="cuda",
        max_staged_bytes=1 << 20,
    )
    backend = DeepSeekV4FlashGPUBackend()

    first = backend.stage_selected_expert_payloads(
        stager,
        grouped,
        torch.tensor([1, 3], dtype=torch.int64),
        layer_idx=2,
    )
    second = backend.stage_selected_expert_payloads(
        stager,
        grouped,
        torch.tensor([3, 5], dtype=torch.int64),
        layer_idx=2,
    )

    assert len(first) == 2
    assert len(second) == 2
    # Overlapping expert 3 must reuse the same payload tensor objects.
    assert first[1][1].payload is second[0][1].payload
    assert first[1][2].payload is second[0][2].payload
    assert first[1][3].payload is second[0][3].payload
    # Distinct experts must have distinct payload tensor objects.
    assert first[0][1].payload is not second[1][1].payload
    assert first[0][2].payload is not second[1][2].payload
    assert first[0][3].payload is not second[1][3].payload
