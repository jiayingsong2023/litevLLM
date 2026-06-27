from vllm.kernels.triton.deepseek_v4_flash.compressed_attention import (
    DeepSeekV4CompressedAttentionInputs,
)


def test_compressed_attention_inputs_require_page_tables() -> None:
    inputs = DeepSeekV4CompressedAttentionInputs(
        raw_page_table_name="raw_page_table",
        compressed_page_table_name="compressed_page_table",
        indexer_page_table_name="indexer_page_table",
        selected_rows_name="selected_compressed_row_ids",
    )
    assert inputs.uses_page_tables is True


def test_compressed_attention_contract_rejects_contiguous_cache_name() -> None:
    try:
        DeepSeekV4CompressedAttentionInputs(
            raw_page_table_name="raw_page_table",
            compressed_page_table_name="contiguous_comp_cache",
            indexer_page_table_name="indexer_page_table",
            selected_rows_name="selected_compressed_row_ids",
        )
    except ValueError as exc:
        assert "page table" in str(exc)
    else:
        raise AssertionError("contiguous cache contract was accepted")


def test_compressed_attention_contract_rejects_missing_page_table_name() -> None:
    try:
        DeepSeekV4CompressedAttentionInputs(
            raw_page_table_name="raw_table",
            compressed_page_table_name="compressed_page_table",
            indexer_page_table_name="indexer_page_table",
            selected_rows_name="selected_compressed_row_ids",
        )
    except ValueError as exc:
        assert "page table" in str(exc)
    else:
        raise AssertionError("compressed attention must require page-table inputs")
