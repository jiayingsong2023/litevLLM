import pytest
import torch

from vllm.model_executor.models.deepseek_v4_flash.compressed_kv import (
    DeepSeekV4CompressedKVCache,
)


def test_raw_swa_cache_keeps_last_128_rows() -> None:
    cache = DeepSeekV4CompressedKVCache(context_length=256, hidden_size=4)
    for token_idx in range(130):
        row = torch.full((4,), float(token_idx))
        cache.append_raw(layer_idx=0, token_idx=token_idx, key=row, value=row + 1)

    keys, values = cache.read_raw_window(layer_idx=0, token_idx=129, window=128)

    assert keys.shape == (128, 4)
    assert values.shape == (128, 4)
    torch.testing.assert_close(keys[0], torch.full((4,), 2.0))
    torch.testing.assert_close(keys[-1], torch.full((4,), 129.0))
    torch.testing.assert_close(values[0], torch.full((4,), 3.0))
    torch.testing.assert_close(values[-1], torch.full((4,), 130.0))


def test_raw_swa_cache_reads_smaller_sorted_window() -> None:
    cache = DeepSeekV4CompressedKVCache(context_length=256, hidden_size=4)
    for token_idx in range(130):
        row = torch.full((4,), float(token_idx))
        cache.append_raw(layer_idx=0, token_idx=token_idx, key=row, value=row + 10)

    keys, values = cache.read_raw_window(layer_idx=0, token_idx=129, window=4)

    assert keys.shape == (4, 4)
    assert values.shape == (4, 4)
    torch.testing.assert_close(keys[:, 0], torch.tensor([126.0, 127.0, 128.0, 129.0]))
    torch.testing.assert_close(
        values[:, 0],
        torch.tensor([136.0, 137.0, 138.0, 139.0]),
    )


def test_raw_swa_cache_validation_rejects_bad_layer_index() -> None:
    cache = DeepSeekV4CompressedKVCache(context_length=256, hidden_size=4)

    with pytest.raises(ValueError, match="layer index"):
        cache.append_raw(
            layer_idx=43,
            token_idx=0,
            key=torch.zeros(4),
            value=torch.zeros(4),
        )

    with pytest.raises(ValueError, match="layer index"):
        cache.read_raw_window(layer_idx=-1, token_idx=0, window=1)


def test_raw_swa_cache_validation_rejects_negative_token_index() -> None:
    cache = DeepSeekV4CompressedKVCache(context_length=256, hidden_size=4)

    with pytest.raises(ValueError, match="token index"):
        cache.append_raw(
            layer_idx=0,
            token_idx=-1,
            key=torch.zeros(4),
            value=torch.zeros(4),
        )

    with pytest.raises(ValueError, match="token index"):
        cache.read_raw_window(layer_idx=0, token_idx=-1, window=1)


def test_raw_swa_cache_validation_rejects_oversized_window() -> None:
    cache = DeepSeekV4CompressedKVCache(context_length=256, hidden_size=4)

    with pytest.raises(ValueError, match="window"):
        cache.read_raw_window(layer_idx=0, token_idx=0, window=129)


def test_raw_swa_cache_validation_rejects_shape_mismatch() -> None:
    cache = DeepSeekV4CompressedKVCache(context_length=256, hidden_size=4)

    with pytest.raises(ValueError, match="key"):
        cache.append_raw(
            layer_idx=0,
            token_idx=0,
            key=torch.zeros(1, 4),
            value=torch.zeros(4),
        )

    with pytest.raises(ValueError, match="value"):
        cache.append_raw(
            layer_idx=0,
            token_idx=0,
            key=torch.zeros(4),
            value=torch.zeros(5),
        )
