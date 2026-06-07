from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.block import (
    DeepSeekV4FlashBlockReference,
    DeepSeekV4FlashLayer0ReferenceRunner,
)
from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import GGML_TYPE_F32
from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    open_deepseek_v4_flash_weight_store,
)

TARGET_GGUF = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
)


def test_block_reference_preserves_hidden_shape() -> None:
    block = DeepSeekV4FlashBlockReference(
        layer_idx=0,
        hidden_size=4,
        attention=lambda hidden, token_idx, kv_cache: hidden * 0.5,
        moe=lambda hidden: hidden * 2.0,
        attn_norm_weight=torch.ones(4),
        ffn_norm_weight=torch.ones(4),
    )

    out = block.forward(torch.ones(4), token_idx=0, kv_cache=None)

    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_block_reference_rejects_attention_shape_mismatch() -> None:
    block = DeepSeekV4FlashBlockReference(
        layer_idx=0,
        hidden_size=4,
        attention=lambda hidden, token_idx, kv_cache: hidden[:2],
        moe=lambda hidden: hidden,
        attn_norm_weight=torch.ones(4),
        ffn_norm_weight=torch.ones(4),
    )

    with pytest.raises(ValueError, match="attention output shape"):
        block.forward(torch.ones(4), token_idx=0, kv_cache=None)


@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_layer0_norms_bind_to_reference_block() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        layer_0 = store.bindings.layers[0]
        assert layer_0.attention_norm is not None
        assert layer_0.ffn_norm is not None
        attn_norm = layer_0.attention_norm
        ffn_norm = layer_0.ffn_norm
        assert attn_norm.tensor_type == GGML_TYPE_F32
        assert ffn_norm.tensor_type == GGML_TYPE_F32

        block = DeepSeekV4FlashBlockReference(
            layer_idx=0,
            hidden_size=4096,
            attention=lambda hidden, token_idx, kv_cache: torch.zeros_like(hidden),
            moe=lambda hidden: torch.zeros_like(hidden),
            attn_norm_weight=store.tensor_to_torch(
                attn_norm,
                dtype=torch.float32,
            ),
            ffn_norm_weight=store.tensor_to_torch(ffn_norm, dtype=torch.float32),
        )
        out = block.forward(torch.ones(4096), token_idx=0, kv_cache=None)

    assert out.shape == (4096,)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_layer0_executes_attention_and_moe() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        runner = DeepSeekV4FlashLayer0ReferenceRunner(store)
        out = runner.forward(
            torch.ones((4, 4096), dtype=torch.float32),
            token_id=1,
            token_idx=0,
        )

    assert out.shape == (4, 4096)
    assert torch.isfinite(out).all()
