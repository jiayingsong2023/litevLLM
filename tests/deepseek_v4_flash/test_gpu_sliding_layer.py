from __future__ import annotations

import math

import pytest
import torch

from vllm.kernels.triton.deepseek_v4_flash.attention import (
    DeepSeekV4AttentionKernelInputs,
    deepseek_v4_attention,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_deepseek_gpu_sliding_attention_matches_torch_reference() -> None:
    device = torch.device("cuda")
    query = torch.tensor([1.0, 0.0, 0.5, -0.5], device=device)
    kv_rows = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )
    attn_sinks = torch.tensor([0.25], device=device)

    output = deepseek_v4_attention(
        DeepSeekV4AttentionKernelInputs(
            hidden=query,
            kv_rows=kv_rows,
            token_idx=0,
            attn_sinks=attn_sinks,
        )
    )

    scores = kv_rows.matmul(query) / math.sqrt(float(query.numel()))
    logits = torch.cat([scores, attn_sinks])
    probs = torch.softmax(logits, dim=0)
    expected = probs[:-1].matmul(kv_rows)

    assert output.device.type == "cuda"
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_backend_sliding_attention_requires_gpu_tensors() -> None:
    backend = DeepSeekV4FlashGPUBackend()

    with pytest.raises(ValueError, match="must be CUDA tensors"):
        backend.sliding_attention(
            query=torch.zeros(4),
            kv_rows=torch.zeros(1, 4),
            attn_sinks=torch.zeros(1),
            token_idx=0,
        )
