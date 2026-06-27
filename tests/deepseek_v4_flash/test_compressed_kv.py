from vllm.model_executor.models.deepseek_v4_flash.compressed_kv import (
    DeepSeekV4CompressedKVLayout,
    DeepSeekV4KVPageAllocator,
)


def test_layout_counts_raw_and_compressed_rows() -> None:
    layout = DeepSeekV4CompressedKVLayout(context_length=8192)
    assert layout.raw_window == 128
    assert layout.layer_comp_capacity(0) == 0
    assert layout.layer_comp_capacity(2) == 8192 // 4 + 2
    assert layout.layer_comp_capacity(3) == 8192 // 128 + 2
    assert layout.has_indexer_cache(2) is True
    assert layout.has_indexer_cache(3) is False


def test_layout_rejects_context_above_first_release_cap() -> None:
    try:
        DeepSeekV4CompressedKVLayout(context_length=16384)
    except ValueError as exc:
        assert "8192" in str(exc)
    else:
        raise AssertionError("context above first-release cap must fail")


def test_page_allocator_maps_raw_rows_without_full_context_allocation() -> None:
    layout = DeepSeekV4CompressedKVLayout(context_length=8192)
    allocator = DeepSeekV4KVPageAllocator(layout)
    ref = allocator.allocate_raw_row(layer_idx=0, logical_row=129)
    assert ref.chunk_id == 0
    assert ref.page_id == 8
    assert ref.row_offset == 1
    assert allocator.raw_pool.max_chunk_bytes < 8192 * 512 * 4


def test_page_allocator_maps_ratio4_compressed_and_indexer_rows() -> None:
    layout = DeepSeekV4CompressedKVLayout(context_length=8192)
    allocator = DeepSeekV4KVPageAllocator(layout)
    comp_ref = allocator.allocate_compressed_row(layer_idx=2, logical_row=65)
    index_ref = allocator.allocate_indexer_row(layer_idx=2, logical_row=65)
    assert comp_ref.page_id == 1
    assert comp_ref.row_offset == 1
    assert index_ref.page_id == 1
    assert index_ref.row_offset == 1
    assert allocator.compressed_pool.row_width == 512
    assert allocator.indexer_pool.row_width == 128


def test_indexer_rows_are_rejected_for_ratio128_layers() -> None:
    layout = DeepSeekV4CompressedKVLayout(context_length=8192)
    allocator = DeepSeekV4KVPageAllocator(layout)
    try:
        allocator.allocate_indexer_row(layer_idx=3, logical_row=0)
    except ValueError as exc:
        assert "indexer" in str(exc)
    else:
        raise AssertionError("ratio-128 layers must not allocate indexer rows")


def test_compressed_rows_reject_capacity_overflow() -> None:
    layout = DeepSeekV4CompressedKVLayout(context_length=4096)
    allocator = DeepSeekV4KVPageAllocator(layout)
    try:
        allocator.allocate_compressed_row(
            layer_idx=2,
            logical_row=layout.layer_comp_capacity(2),
        )
    except ValueError as exc:
        assert "capacity" in str(exc)
    else:
        raise AssertionError("compressed rows beyond capacity must fail")
