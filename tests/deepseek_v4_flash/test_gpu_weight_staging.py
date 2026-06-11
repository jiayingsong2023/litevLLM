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


def _tensor(name: str) -> DeepSeekV4FlashTensor:
    return DeepSeekV4FlashTensor(
        name=name,
        dims=(2, 2, 1),
        tensor_type=GGML_TYPE_Q8_0,
        offset=0,
        nbytes=0,
    )


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

    assert staged.gate.device.type == "cuda"
    assert staged.up.device.type == "cuda"
    assert staged.down.device.type == "cuda"
    assert staged.gate.ndim == 2
    assert staged.up.ndim == 2
    assert staged.down.ndim == 2
    assert torch.isfinite(staged.gate).all()
