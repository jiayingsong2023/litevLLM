from fixtures import write_minimal_deepseek_v4_flash_gguf

from vllm.model_executor.models.deepseek_v4_flash.weight_store import (
    DeepSeekV4FlashWeightStoreError,
    open_deepseek_v4_flash_weight_store,
)


def test_weight_store_binds_required_inspect_tensors(tmp_path):
    path = tmp_path / "deepseek-v4-flash.gguf"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight"),
    )

    with open_deepseek_v4_flash_weight_store(path) as store:
        diagnostics = store.diagnostics

        assert diagnostics.tensor_count == 2
        assert diagnostics.file_size_bytes == path.stat().st_size
        assert diagnostics.mmap_size_bytes == path.stat().st_size
        assert diagnostics.bound_tensor_count == 2
        assert diagnostics.missing_required_semantic_tensors == ()
        assert store.bindings.token_embedding.name == "token_embd.weight"
        assert store.bindings.representative_layer_tensor.name == "blk.0.attn_q.weight"


def test_weight_store_rejects_missing_required_inspect_tensor(tmp_path):
    path = tmp_path / "deepseek-v4-flash.gguf"
    write_minimal_deepseek_v4_flash_gguf(path)

    try:
        open_deepseek_v4_flash_weight_store(path)
    except DeepSeekV4FlashWeightStoreError as exc:
        assert "missing required inspect-only tensors" in str(exc)
        assert "layer_0_attention_query" in str(exc)
        assert "blk.0.attn_q.weight" in str(exc)
    else:
        raise AssertionError("missing required layer tensor was accepted")
