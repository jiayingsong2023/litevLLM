import torch

from vllm.model_executor.models.deepseek_v4_flash.compressed_kv import (
    DeepSeekV4PagedKVCache,
)
from vllm.model_executor.models.deepseek_v4_flash.config import (
    DEEPSEEK_V4_FLASH_SHAPE,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_layers import (
    _write_compressed_runtime_row,
)
from vllm.model_executor.models.deepseek_v4_flash.gpu_runtime import (
    DeepSeekV4FlashGPUCacheConfig,
    DeepSeekV4FlashGPURequestState,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)


def test_paged_cache_lazily_allocates_backing_chunks_by_family() -> None:
    cache = DeepSeekV4PagedKVCache(
        context_length=256,
        hidden_size=4,
        num_layers=4,
        blocks_per_chunk=2,
    )

    assert cache.num_allocated_chunks("raw") == 0
    assert cache.num_allocated_chunks("compressed") == 0
    assert cache.num_allocated_chunks("indexer") == 0

    row = torch.ones(4)
    cache.append_raw(layer_idx=0, token_idx=0, key=row, value=row)

    assert cache.num_allocated_chunks("raw") == 1
    assert cache.num_allocated_chunks("compressed") == 0
    assert cache.num_allocated_chunks("indexer") == 0

    cache.append_compressed(
        layer_idx=2,
        token_idx=3,
        row=torch.arange(4, dtype=torch.float32),
        indexer_row=torch.arange(128, dtype=torch.float32),
    )

    assert cache.num_allocated_chunks("compressed") == 1
    assert cache.num_allocated_chunks("indexer") == 1


def test_paged_cache_preserves_raw_window_order_after_wrap() -> None:
    cache = DeepSeekV4PagedKVCache(
        context_length=256,
        hidden_size=4,
        num_layers=4,
        blocks_per_chunk=2,
    )
    for token_idx in range(130):
        row = torch.full((4,), float(token_idx))
        cache.append_raw(layer_idx=0, token_idx=token_idx, key=row, value=row + 1)

    keys, values = cache.read_raw_window(layer_idx=0, token_idx=129, window=4)

    torch.testing.assert_close(keys[:, 0], torch.tensor([126.0, 127.0, 128.0, 129.0]))
    torch.testing.assert_close(
        values[:, 0],
        torch.tensor([127.0, 128.0, 129.0, 130.0]),
    )


def test_paged_cache_reads_compressed_and_indexer_rows() -> None:
    cache = DeepSeekV4PagedKVCache(
        context_length=256,
        hidden_size=4,
        num_layers=4,
        blocks_per_chunk=2,
    )

    cache.append_compressed(
        layer_idx=2,
        token_idx=3,
        row=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        indexer_row=torch.arange(128, dtype=torch.float32),
    )
    cache.append_compressed(
        layer_idx=2,
        token_idx=7,
        row=torch.tensor([5.0, 6.0, 7.0, 8.0]),
        indexer_row=torch.arange(128, dtype=torch.float32) + 10,
    )

    selected = cache.read_compressed(layer_idx=2, row_indices=torch.tensor([1]))
    indexer_rows = cache.read_indexer_rows(layer_idx=2)

    torch.testing.assert_close(selected, torch.tensor([[5.0, 6.0, 7.0, 8.0]]))
    assert indexer_rows.shape == (2, 128)
    torch.testing.assert_close(indexer_rows[1, :3], torch.tensor([10.0, 11.0, 12.0]))


def test_paged_cache_free_reuses_zeroed_blocks_without_releasing_chunks() -> None:
    cache = DeepSeekV4PagedKVCache(
        context_length=256,
        hidden_size=4,
        num_layers=4,
        blocks_per_chunk=2,
    )

    cache.append_raw(layer_idx=0, token_idx=0, key=torch.ones(4), value=torch.ones(4))
    allocated_chunks = cache.num_allocated_chunks("raw")
    cache.free_request_blocks()

    cache.append_raw(layer_idx=0, token_idx=0, key=torch.zeros(4), value=torch.zeros(4))
    keys, values = cache.read_raw_window(layer_idx=0, token_idx=0, window=1)

    assert cache.num_allocated_chunks("raw") == allocated_chunks
    torch.testing.assert_close(keys, torch.zeros(1, 4))
    torch.testing.assert_close(values, torch.zeros(1, 4))


def test_runtime_write_helper_uses_paged_cache_api() -> None:
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=256,
            hidden_size=DEEPSEEK_V4_FLASH_SHAPE.hidden_size,
            kv_width=4,
            dtype=torch.float32,
            device="cpu",
        )
    )

    _write_compressed_runtime_row(
        state,
        layer_idx=2,
        token_idx=3,
        compressed_row=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        indexer_row=torch.arange(128, dtype=torch.float32),
    )

    cache = state.compressed_kv_cache
    assert cache._compressed_counts_cpu[2] == 1
    torch.testing.assert_close(
        cache.read_compressed(layer_idx=2),
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    )
    torch.testing.assert_close(
        cache.compressed_rows[2, 0],
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    torch.testing.assert_close(
        cache.indexer_rows[2, 0, :3],
        torch.tensor([0.0, 1.0, 2.0]),
    )


def test_paged_cache_reuses_freed_blocks_across_request_views() -> None:
    pool = DeepSeekV4PagedKVCache(
        context_length=256,
        hidden_size=4,
        num_layers=4,
        blocks_per_chunk=1,
    )
    first = pool.bind_request("first")
    second = pool.bind_request("second")

    first.append_raw(layer_idx=0, token_idx=0, key=torch.ones(4), value=torch.ones(4))
    assert pool.num_allocated_chunks("raw") == 1

    first.free_request_blocks()
    second.append_raw(
        layer_idx=0,
        token_idx=0,
        key=torch.zeros(4),
        value=torch.zeros(4),
    )
    keys, values = second.read_raw_window(layer_idx=0, token_idx=0, window=1)

    assert pool.num_allocated_chunks("raw") == 1
    torch.testing.assert_close(keys, torch.zeros(1, 4))
    torch.testing.assert_close(values, torch.zeros(1, 4))


def test_request_state_can_share_model_level_paged_cache() -> None:
    pool = DeepSeekV4PagedKVCache(
        context_length=256,
        hidden_size=4,
        num_layers=DEEPSEEK_V4_FLASH_SHAPE.num_layers,
        blocks_per_chunk=1,
    )
    config = DeepSeekV4FlashGPUCacheConfig(
        context_length=256,
        hidden_size=DEEPSEEK_V4_FLASH_SHAPE.hidden_size,
        kv_width=4,
        dtype=torch.float32,
        device="cpu",
    )
    first = DeepSeekV4FlashGPURequestState(config, kv_cache=pool, request_id="first")
    second = DeepSeekV4FlashGPURequestState(config, kv_cache=pool, request_id="second")

    first.raw_kv_cache.append_raw(
        layer_idx=0,
        token_idx=0,
        key=torch.ones(4),
        value=torch.ones(4),
    )
    first.reset()
    second.raw_kv_cache.append_raw(
        layer_idx=0,
        token_idx=0,
        key=torch.zeros(4),
        value=torch.zeros(4),
    )

    assert first.kv_cache is first.raw_kv_cache
    assert second.kv_cache is second.compressed_kv_cache
    assert pool.num_allocated_chunks("raw") == 1


def test_model_creates_request_states_from_one_paged_pool() -> None:
    model = DeepSeekV4FlashForCausalLM()

    first = model._new_gpu_request_state(context_length=256, device=torch.device("cpu"))
    second = model._new_gpu_request_state(
        context_length=256,
        device=torch.device("cpu"),
    )

    assert first.kv_cache._pool is second.kv_cache._pool
    assert first.request_id != second.request_id


def test_paged_cache_returns_empty_indexer_rows_for_ratio128_layers() -> None:
    cache = DeepSeekV4PagedKVCache(
        context_length=256,
        hidden_size=4,
        num_layers=4,
        blocks_per_chunk=1,
    )
    cache.append_compressed(
        layer_idx=3,
        token_idx=127,
        row=torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )

    rows = cache.read_indexer_rows(layer_idx=3)

    assert rows.shape == (0, DEEPSEEK_V4_FLASH_SHAPE.indexer_head_dim)
    assert rows.dtype == torch.float32


def test_paged_cache_sizes_allocator_for_concurrent_request_views() -> None:
    pool = DeepSeekV4PagedKVCache(
        context_length=128,
        hidden_size=4,
        num_layers=1,
        block_size=16,
        blocks_per_chunk=1,
        max_requests=2,
    )
    first = pool.bind_request("first")
    second = pool.bind_request("second")

    for token_idx in range(128):
        row = torch.full((4,), float(token_idx))
        first.append_raw(layer_idx=0, token_idx=token_idx, key=row, value=row)
        second.append_raw(layer_idx=0, token_idx=token_idx, key=row, value=row)

    keys, _values = second.read_raw_window(layer_idx=0, token_idx=127, window=1)

    torch.testing.assert_close(keys, torch.full((1, 4), 127.0))


def test_model_request_state_pool_tracks_max_requests() -> None:
    model = DeepSeekV4FlashForCausalLM()

    first = model._new_gpu_request_state(
        context_length=256,
        device=torch.device("cpu"),
        max_requests=2,
    )
    second = model._new_gpu_request_state(
        context_length=256,
        device=torch.device("cpu"),
        max_requests=2,
    )

    assert first.kv_cache._pool is second.kv_cache._pool
    assert first.kv_cache._pool.max_requests == 2
