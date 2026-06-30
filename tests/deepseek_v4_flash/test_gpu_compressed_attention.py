from __future__ import annotations

import math
import os

import pytest
import torch

from vllm.kernels.triton.deepseek_v4_flash.cache import (
    DeepSeekV4CacheUpdateInputs,
    deepseek_v4_cache_update,
)
from vllm.kernels.triton.deepseek_v4_flash.compressed_attention import (
    DeepSeekV4CompressedAttentionTensorInputs,
    deepseek_v4_compressed_attention,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_deepseek_gpu_compressed_attention_triton_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    for head_dim in (128, 256, 512):
        for n_selected in (1, 7, 64, 512):
            query = torch.randn(head_dim, device=device, dtype=torch.float16)
            compressed_rows = torch.randn(
                n_selected + 4,
                head_dim,
                device=device,
                dtype=torch.float16,
            )
            selected_rows = torch.arange(
                n_selected,
                dtype=torch.int64,
                device=device,
            )

            monkeypatch.setenv(
                "FASTINFERENCE_DEEPSEEK_V4_FLASH_COMPRESSED_ATTN_FALLBACK", "1"
            )
            expected = deepseek_v4_compressed_attention(
                DeepSeekV4CompressedAttentionTensorInputs(
                    query=query,
                    compressed_rows=compressed_rows,
                    selected_rows=selected_rows,
                )
            )
            monkeypatch.setenv(
                "FASTINFERENCE_DEEPSEEK_V4_FLASH_COMPRESSED_ATTN_FALLBACK", "0"
            )
            got = deepseek_v4_compressed_attention(
                DeepSeekV4CompressedAttentionTensorInputs(
                    query=query,
                    compressed_rows=compressed_rows,
                    selected_rows=selected_rows,
                )
            )
            torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_deepseek_gpu_cache_update_writes_through_page_table() -> None:
    device = torch.device("cuda")
    cache_storage = torch.zeros((2, 2, 4), dtype=torch.float32, device=device)
    page_table = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_row = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)

    deepseek_v4_cache_update(
        DeepSeekV4CacheUpdateInputs(
            page_table=page_table,
            kv_row=kv_row,
            cache_storage=cache_storage,
            logical_row=1,
        )
    )

    torch.testing.assert_close(cache_storage[1, 1], kv_row)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_deepseek_gpu_compressed_attention_uses_selected_rows() -> None:
    device = torch.device("cuda")
    query = torch.tensor([1.0, 0.0, 0.5, -0.5], device=device)
    compressed_rows = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )
    selected_rows = torch.tensor([0, 2], dtype=torch.int32, device=device)

    output = deepseek_v4_compressed_attention(
        DeepSeekV4CompressedAttentionTensorInputs(
            query=query,
            compressed_rows=compressed_rows,
            selected_rows=selected_rows,
        )
    )

    selected = compressed_rows.index_select(0, selected_rows.to(torch.long))
    scores = selected.matmul(query) / math.sqrt(float(query.numel()))
    expected = torch.softmax(scores, dim=0).matmul(selected)
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_gpu_backend_compressed_attention_requires_gpu_tensors() -> None:
    backend = DeepSeekV4FlashGPUBackend()

    with pytest.raises(ValueError, match="must be CUDA tensors"):
        backend.compressed_attention(
            query=torch.zeros(4),
            compressed_rows=torch.zeros(1, 4),
            selected_rows=torch.tensor([0], dtype=torch.int32),
        )
