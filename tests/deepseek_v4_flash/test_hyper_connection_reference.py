from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.hyper_connection import (
    hyper_connection_post_reference,
    hyper_connection_pre_reference,
    sinkhorn_reference,
)
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    open_deepseek_v4_flash_weight_store,
)

TARGET_GGUF = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


def test_sinkhorn_reference_returns_nearly_balanced_matrix() -> None:
    out = sinkhorn_reference(torch.tensor([[2.0, 0.0], [1.0, 3.0]]))

    torch.testing.assert_close(out.sum(dim=0), torch.ones(2), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.sum(dim=1), torch.ones(2), atol=1e-5, rtol=1e-5)


def test_hyper_connection_pre_post_reference_shapes() -> None:
    streams = torch.arange(1.0, 7.0).reshape(2, 3)
    fn_weight = torch.zeros((6, 8))
    base = torch.zeros(8)
    scale = torch.ones(3)

    state = hyper_connection_pre_reference(streams, fn_weight, base, scale)
    out_streams = hyper_connection_post_reference(
        torch.ones(3),
        streams,
        state,
    )

    assert state.mixed.shape == (3,)
    assert state.post.shape == (2,)
    assert state.combine.shape == (2, 2)
    assert out_streams.shape == (2, 3)
    assert torch.isfinite(out_streams).all()


@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_layer0_hyper_connection_tensor_shapes() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        hc = store.bindings.layers[0].attention_hyper_connection
        assert hc is not None
        fn = store.tensor_to_torch(hc.fn, dtype=torch.float16)
        base = store.tensor_to_torch(hc.base, dtype=torch.float32)
        scale = store.tensor_to_torch(hc.scale, dtype=torch.float32)

    assert fn.shape == (16384, 24)
    assert base.shape == (24,)
    assert scale.shape == (3,)
