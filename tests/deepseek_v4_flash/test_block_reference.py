from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.block import (
    DeepSeekV4FlashBlockReference,
    DeepSeekV4FlashCompressedLayerReferenceRunner,
    DeepSeekV4FlashLayer0ReferenceRunner,
    DeepSeekV4FlashLayer2ReferenceRunner,
    DeepSeekV4FlashSlidingLayerReferenceRunner,
)
from vllm.model_executor.models.deepseek_v4_flash.compressed_kv import (
    DeepSeekV4CompressedKVCache,
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


@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_layer1_executes_sliding_attention_and_moe() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        runner = DeepSeekV4FlashSlidingLayerReferenceRunner(store, layer_idx=1)
        out = runner.forward(
            torch.ones((4, 4096), dtype=torch.float32),
            token_id=1,
            token_idx=0,
        )

    assert out.shape == (4, 4096)
    assert torch.isfinite(out).all()


def test_sliding_layer_runner_rejects_compressed_layers() -> None:
    with pytest.raises(ValueError, match="supports only sliding-only layers"):
        DeepSeekV4FlashSlidingLayerReferenceRunner(None, layer_idx=2)  # type: ignore[arg-type]


@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_layer2_executes_compressed_attention_and_emits_cache_row() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        runner = DeepSeekV4FlashLayer2ReferenceRunner(store)
        cache = DeepSeekV4CompressedKVCache(context_length=256, hidden_size=512)
        streams = torch.ones((4, 4096), dtype=torch.float32)
        for token_idx in range(4):
            streams = runner.forward(
                streams,
                token_id=1,
                token_idx=token_idx,
                cache=cache,
            )

    assert streams.shape == (4, 4096)
    assert torch.isfinite(streams).all()
    assert cache.read_compressed(layer_idx=2).shape == (1, 512)
    assert cache.read_indexer_rows(layer_idx=2).shape == (1, 128)


@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_layer3_executes_ratio128_attention_and_router_moe() -> None:
    with open_deepseek_v4_flash_weight_store(TARGET_GGUF) as store:
        runner = DeepSeekV4FlashCompressedLayerReferenceRunner(store, layer_idx=3)
        cache = DeepSeekV4CompressedKVCache(context_length=256, hidden_size=512)
        out = runner.forward(
            torch.ones((4, 4096), dtype=torch.float32),
            token_id=1,
            token_idx=0,
            cache=cache,
        )

    assert out.shape == (4, 4096)
    assert torch.isfinite(out).all()
    assert cache.read_compressed(layer_idx=3).shape == (0, 512)
