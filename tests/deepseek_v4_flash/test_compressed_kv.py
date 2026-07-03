from vllm.model_executor.models.deepseek_v4_flash.compressed_kv import (
    DeepSeekV4CompressedKVLayout,
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


def test_layout_rejects_indexer_for_ratio128_layers() -> None:
    layout = DeepSeekV4CompressedKVLayout(context_length=8192)

    assert layout.has_indexer_cache(2) is True
    assert layout.has_indexer_cache(3) is False
