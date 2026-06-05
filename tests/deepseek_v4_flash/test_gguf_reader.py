from fixtures import write_minimal_deepseek_v4_flash_gguf

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGUFParseError,
    read_deepseek_v4_flash_gguf,
)


def test_reader_accepts_minimal_target_metadata(tmp_path) -> None:
    filename = (
        "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
    )
    path = tmp_path / filename
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


def test_reader_rejects_unsupported_tensor_type(tmp_path) -> None:
    path = tmp_path / "bad-tensor-type.gguf"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight",),
        tensor_types=(99,),
    )

    try:
        read_deepseek_v4_flash_gguf(path)
    except GGUFParseError as exc:
        assert "unsupported DeepSeek V4 Flash tensor type" in str(exc)
        assert "99" in str(exc)
    else:
        raise AssertionError("unsupported tensor type must fail")
