import struct
from pathlib import Path

import pytest
from fixtures import write_minimal_deepseek_v4_flash_gguf

from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGUFParseError,
    read_deepseek_v4_flash_gguf,
)

TARGET_GGUF = Path(
    "models/DeepSeek-V4-Flash-ds4/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
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


def test_reader_rejects_overlapping_tensor_payloads(tmp_path) -> None:
    path = tmp_path / "overlap.gguf"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight", "blk.0.attn_q.weight"),
        tensor_types=(0, 0),
        tensor_dims={"token_embd.weight": (1,), "blk.0.attn_q.weight": (1,)},
        tensor_payloads={
            "token_embd.weight": b"\x00\x00\x00\x00",
            "blk.0.attn_q.weight": b"\x01\x00\x00\x00",
        },
        tensor_offsets={"token_embd.weight": 0, "blk.0.attn_q.weight": 0},
    )

    with pytest.raises(GGUFParseError, match="overlap"):
        read_deepseek_v4_flash_gguf(path)


def test_reader_rejects_out_of_file_tensor_payload(tmp_path) -> None:
    path = tmp_path / "out-of-file.gguf"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight",),
        tensor_types=(0,),
        tensor_dims={"token_embd.weight": (1,)},
        tensor_payloads={"token_embd.weight": b""},
        tensor_offsets={"token_embd.weight": 128},
    )

    with pytest.raises(GGUFParseError, match="exceeds file size"):
        read_deepseek_v4_flash_gguf(path)


def test_reader_rejects_zero_byte_tensor_payload(tmp_path) -> None:
    path = tmp_path / "zero-byte.gguf"
    write_minimal_deepseek_v4_flash_gguf(
        path,
        tensor_names=("token_embd.weight",),
        tensor_types=(0,),
        tensor_dims={"token_embd.weight": (0,)},
    )

    with pytest.raises(GGUFParseError, match="invalid byte size"):
        read_deepseek_v4_flash_gguf(path)


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


@pytest.mark.skipif(not TARGET_GGUF.exists(), reason="target DeepSeek V4 GGUF absent")
def test_real_gguf_all_tensor_ranges_are_inside_file() -> None:
    model = read_deepseek_v4_flash_gguf(TARGET_GGUF)
    file_size = TARGET_GGUF.stat().st_size

    for tensor in model.tensors.values():
        start = model.data_offset + tensor.offset
        assert 0 <= start < file_size
        assert start + tensor.nbytes <= file_size
