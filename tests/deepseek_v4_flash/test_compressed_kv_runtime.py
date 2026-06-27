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


def test_raw_swa_cache_validation_rejects_non_default_raw_window() -> None:
    with pytest.raises(ValueError, match="raw_window"):
        DeepSeekV4CompressedKVCache(
            context_length=256,
            hidden_size=4,
            raw_window=64,
        )


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


def test_raw_swa_cache_validation_rejects_dtype_mismatch() -> None:
    cache = DeepSeekV4CompressedKVCache(context_length=256, hidden_size=4)

    with pytest.raises(ValueError, match="key dtype"):
        cache.append_raw(
            layer_idx=0,
            token_idx=0,
            key=torch.zeros(4, dtype=torch.float64),
            value=torch.zeros(4),
        )

    with pytest.raises(ValueError, match="value dtype"):
        cache.append_raw(
            layer_idx=0,
            token_idx=0,
            key=torch.zeros(4),
            value=torch.zeros(4, dtype=torch.float64),
        )


def test_raw_swa_cache_validation_rejects_device_mismatch() -> None:
    cache = DeepSeekV4CompressedKVCache(context_length=256, hidden_size=4)

    with pytest.raises(ValueError, match="key device"):
        cache.append_raw(
            layer_idx=0,
            token_idx=0,
            key=torch.empty(4, device="meta"),
            value=torch.zeros(4),
        )

    with pytest.raises(ValueError, match="value device"):
        cache.append_raw(
            layer_idx=0,
            token_idx=0,
            key=torch.zeros(4),
            value=torch.empty(4, device="meta"),
        )


def test_compressed_cache_appends_and_reads_indexed_rows() -> None:
    cache = DeepSeekV4CompressedKVCache(context_length=256, hidden_size=4)
    first = cache.append_compressed(
        layer_idx=2,
        token_idx=3,
        row=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        indexer_row=torch.arange(128, dtype=torch.float32),
    )
    second = cache.append_compressed(
        layer_idx=2,
        token_idx=7,
        row=torch.tensor([5.0, 6.0, 7.0, 8.0]),
        indexer_row=torch.arange(128, dtype=torch.float32) + 10,
    )

    assert first == 0
    assert second == 1
    selected = cache.read_compressed(layer_idx=2, row_indices=torch.tensor([1]))
    indexer_rows = cache.read_indexer_rows(layer_idx=2)

    torch.testing.assert_close(selected, torch.tensor([[5.0, 6.0, 7.0, 8.0]]))
    assert indexer_rows.shape == (2, 128)
    torch.testing.assert_close(indexer_rows[0, :3], torch.tensor([0.0, 1.0, 2.0]))
