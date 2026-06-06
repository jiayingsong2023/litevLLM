import struct

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


def test_reader_accepts_real_auxiliary_tensor_types(tmp_path) -> None:
    path = tmp_path / "auxiliary-tensor-types.gguf"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=(
            "token_embd.weight",
            "blk.0.attn_sinks.weight",
            "blk.0.attn_kv_a_norm.weight",
            "blk.0.ffn_gate_tid2eid.weight",
        ),
        tensor_types=(1, 0, 1, 26),
    )

    model = read_deepseek_v4_flash_gguf(path)

    assert model.tensors["blk.0.attn_sinks.weight"].tensor_type == 0
    assert model.tensors["blk.0.ffn_gate_tid2eid.weight"].tensor_type == 26


def test_reader_skips_unused_array_metadata(tmp_path) -> None:
    path = tmp_path / "array-metadata.gguf"

    def add_array_metadata(metadata: bytearray) -> None:
        encoded_key = b"tokenizer.ggml.tokens"
        metadata.extend(struct.pack("<Q", len(encoded_key)))
        metadata.extend(encoded_key)
        metadata.extend(struct.pack("<I", 9))
        metadata.extend(struct.pack("<I", 8))
        metadata.extend(struct.pack("<Q", 2))
        for value in ("<pad>", "<eos>"):
            encoded_value = value.encode()
            metadata.extend(struct.pack("<Q", len(encoded_value)))
            metadata.extend(encoded_value)

    write_minimal_deepseek_v4_flash_gguf(path, extra_metadata=add_array_metadata)

    model = read_deepseek_v4_flash_gguf(path)

    assert model.tensors["token_embd.weight"].name == "token_embd.weight"
    assert "tokenizer.ggml.tokens" not in model.metadata
