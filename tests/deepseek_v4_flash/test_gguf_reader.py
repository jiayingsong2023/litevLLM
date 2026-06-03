from fixtures import write_minimal_deepseek_v4_flash_gguf

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGUFParseError,
    read_deepseek_v4_flash_gguf,
)


def test_reader_accepts_minimal_target_metadata(tmp_path) -> None:
    path = tmp_path / "DeepSeek-V4-Flash-Spark-Q2-REAP-ds4.gguf"
    write_minimal_deepseek_v4_flash_gguf(path)

    model = read_deepseek_v4_flash_gguf(path)

    assert model.name == "DeepSeek V4 Flash Spark Q2 REAP"
    assert model.shape.num_layers == 43
    assert model.metadata["general.architecture"] == "deepseek4"
    assert model.tensors["token_embd.weight"].name == "token_embd.weight"


def test_reader_rejects_wrong_layer_count(tmp_path) -> None:
    path = tmp_path / "bad.gguf"
    write_minimal_deepseek_v4_flash_gguf(path, block_count=44)

    try:
        read_deepseek_v4_flash_gguf(path)
    except GGUFParseError as exc:
        assert "block_count" in str(exc)
    else:
        raise AssertionError("invalid block_count must fail")
