from __future__ import annotations

import struct

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_Q8_0,
    DeepSeekV4FlashTensor,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_layers import (
    _project_tensor,
    deepseek_v4_flash_rms_norm,
    deepseek_v4_flash_staged_matrix_projection,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_weight_staging import (
    DeepSeekV4FlashGPUWeightStager,
)


class _FakeStore:
    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}
        self.matrices: dict[str, torch.Tensor] = {}

    def decode_matrix(self, tensor: DeepSeekV4FlashTensor) -> torch.Tensor:
        return self.matrices[tensor.name].clone()

    def tensor_payload(self, tensor: DeepSeekV4FlashTensor) -> memoryview:
        return memoryview(bytearray(self.payloads[tensor.name]))

    def tensor_to_torch(
        self,
        tensor: DeepSeekV4FlashTensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError

    def decode_grouped_expert_matrix(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def raw_grouped_expert_payload(
        self,
        tensor: DeepSeekV4FlashTensor,
        expert_id: int,
    ):
        raise NotImplementedError


def _q8_tensor(name: str, dims: tuple[int, ...]) -> DeepSeekV4FlashTensor:
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
        assert len(values) == 32
        payload.extend(struct.pack("<e", scale))
        payload.extend(struct.pack("<" + "b" * 32, *values))
    return bytes(payload)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_staged_matrix_projection() -> None:
    device = torch.device("cuda")
    hidden_size = 16
    out_features = 8
    batch = 3
    hidden = torch.randn(batch, hidden_size, device=device, dtype=torch.float32)
    weight = torch.randn(out_features, hidden_size, device=device, dtype=torch.float32)
    out = deepseek_v4_flash_staged_matrix_projection(hidden, weight)
    expected = torch.matmul(hidden, weight.T)
    assert out.shape == (batch, out_features)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_rms_norm() -> None:
    device = torch.device("cuda")
    hidden_size = 16
    batch = 3
    hidden = torch.randn(batch, hidden_size, device=device, dtype=torch.float16)
    weight = torch.randn(hidden_size, device=device, dtype=torch.float16)
    out = deepseek_v4_flash_rms_norm(hidden, weight)
    assert out.shape == (batch, hidden_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_batched_project_tensor_q8_0_fallback() -> None:
    device = torch.device("cuda")
    hidden_size = 32
    rows = 2
    batch = 3
    tensor = _q8_tensor("test.weight", (hidden_size, rows))
    store = _FakeStore()
    store.payloads[tensor.name] = _q8_payload(
        [
            (0.5, tuple([2] + [0] * 31)),
            (0.25, tuple([0, 4] + [0] * 30)),
        ]
    )
    stager = DeepSeekV4FlashGPUWeightStager(store, device=device)
    hidden = torch.zeros((batch, hidden_size), dtype=torch.float32, device=device)
    hidden[0, 0] = 3.0
    hidden[0, 1] = 5.0
    hidden[1, 0] = 6.0
    hidden[1, 1] = 10.0
    hidden[2, 0] = 1.5
    hidden[2, 1] = 2.5

    out = _project_tensor(hidden, tensor, stager)

    assert out.shape == (batch, rows)
    expected = torch.tensor(
        [[3.0, 5.0], [6.0, 10.0], [1.5, 2.5]],
        dtype=torch.float32,
        device=device,
    )
    torch.testing.assert_close(out, expected)
