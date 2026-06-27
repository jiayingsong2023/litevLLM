from __future__ import annotations

import struct
from collections.abc import Callable
from pathlib import Path

GGUF_MAGIC = 0x46554747
DEEPSEEK_V4_FLASH_METADATA_COUNT = 14
GGUF_DEFAULT_ALIGNMENT = 32
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q2_K = 10
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_I32 = 26


def _write_string(buf: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    buf.extend(struct.pack("<Q", len(encoded)))
    buf.extend(encoded)


def _write_kv_string(buf: bytearray, key: str, value: str) -> None:
    _write_string(buf, key)
    buf.extend(struct.pack("<I", 8))
    _write_string(buf, value)


def _write_kv_u32(buf: bytearray, key: str, value: int) -> None:
    _write_string(buf, key)
    buf.extend(struct.pack("<I", 4))
    buf.extend(struct.pack("<I", value))


def _write_tensor(
    buf: bytearray, name: str, dims: tuple[int, ...], tensor_type: int, offset: int
) -> None:
    _write_string(buf, name)
    buf.extend(struct.pack("<I", len(dims)))
    for dim in dims:
        buf.extend(struct.pack("<Q", dim))
    buf.extend(struct.pack("<I", tensor_type))
    buf.extend(struct.pack("<Q", offset))


def _align_offset(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _tensor_nbytes(dims: tuple[int, ...], tensor_type: int) -> int:
    element_count = 1
    for dim in dims:
        element_count *= dim
    if tensor_type == GGML_TYPE_F32:
        return element_count * 4
    if tensor_type == GGML_TYPE_F16:
        return element_count * 2
    if tensor_type == GGML_TYPE_I32:
        return element_count * 4
    if tensor_type == GGML_TYPE_Q8_0:
        return element_count // 32 * 34
    if tensor_type == GGML_TYPE_Q2_K:
        return element_count // 256 * 84
    if tensor_type == GGML_TYPE_IQ2_XXS:
        return element_count // 256 * 66
    return 0


def write_minimal_deepseek_v4_flash_gguf(
    path: Path,
    *,
    block_count: int = 43,
    tensor_names: tuple[str, ...] = ("token_embd.weight",),
    tensor_types: tuple[int, ...] | None = None,
    tensor_dims: dict[str, tuple[int, ...]] | None = None,
    tensor_payloads: dict[str, bytes] | None = None,
    tensor_offsets: dict[str, int] | None = None,
    tensor_offsets_by_index: tuple[int, ...] | None = None,
    extra_metadata: Callable[[bytearray], None] | None = None,
) -> None:
    metadata = bytearray()
    _write_kv_string(metadata, "general.architecture", "deepseek4")
    _write_kv_string(metadata, "general.name", "DeepSeek V4 Flash Spark Q2 REAP")
    _write_kv_u32(metadata, "deepseek4.block_count", block_count)
    _write_kv_u32(metadata, "deepseek4.attention.head_count", 64)
    _write_kv_u32(metadata, "deepseek4.attention.head_count_kv", 1)
    _write_kv_u32(metadata, "deepseek4.attention.key_length", 512)
    _write_kv_u32(metadata, "deepseek4.attention.sliding_window", 128)
    _write_kv_u32(metadata, "deepseek4.attention.indexer.head_count", 64)
    _write_kv_u32(metadata, "deepseek4.attention.indexer.key_length", 128)
    _write_kv_u32(metadata, "deepseek4.attention.indexer.top_k", 512)
    _write_kv_u32(metadata, "deepseek4.expert_count", 256)
    _write_kv_u32(metadata, "deepseek4.expert_used_count", 6)
    _write_kv_u32(metadata, "deepseek4.embedding_length", 4096)
    _write_kv_u32(metadata, "deepseek4.vocab_size", 129280)
    if extra_metadata is not None:
        extra_metadata(metadata)

    tensors = bytearray()
    if tensor_types is None:
        tensor_types = tuple(8 for _ in tensor_names)
    if len(tensor_types) != len(tensor_names):
        raise ValueError("tensor_types length must match tensor_names length")
    if tensor_offsets_by_index is not None and len(tensor_offsets_by_index) != len(
        tensor_names
    ):
        raise ValueError(
            "tensor_offsets_by_index length must match tensor_names length"
        )
    if tensor_payloads is None:
        tensor_payloads = {}
    if tensor_offsets is None:
        tensor_offsets = {}

    tensor_specs: list[tuple[str, tuple[int, ...], int, int]] = []
    next_tensor_offset = 0
    for offset, name in enumerate(tensor_names):
        if tensor_dims is not None and name in tensor_dims:
            dims = tensor_dims[name]
        else:
            dims = (4096, 129280) if name == "token_embd.weight" else (4096, 4096)
        if tensor_offsets_by_index is not None:
            tensor_offset = tensor_offsets_by_index[offset]
        else:
            tensor_offset = tensor_offsets.get(name, next_tensor_offset)
        tensor_specs.append((name, dims, tensor_types[offset], tensor_offset))
        _write_tensor(tensors, name, dims, tensor_types[offset], tensor_offset)
        next_tensor_offset = _align_offset(
            tensor_offset + _tensor_nbytes(dims, tensor_types[offset]),
            GGUF_DEFAULT_ALIGNMENT,
        )
    header = struct.pack(
        "<IIQQ",
        GGUF_MAGIC,
        3,
        len(tensor_names),
        DEEPSEEK_V4_FLASH_METADATA_COUNT + (1 if extra_metadata is not None else 0),
    )
    if tensor_payloads:
        data_start = _align_offset(
            len(header) + len(metadata) + len(tensors),
            GGUF_DEFAULT_ALIGNMENT,
        )
        tensors = bytearray()
        data = bytearray()
        for name, dims, tensor_type, default_offset in tensor_specs:
            offset = default_offset
            payload = tensor_payloads.get(name, b"")
            if offset > len(data):
                data.extend(b"\x00" * (offset - len(data)))
            data.extend(payload)
            _write_tensor(tensors, name, dims, tensor_type, offset)
        padding = b"\x00" * (data_start - len(header) - len(metadata) - len(tensors))
        path.write_bytes(header + metadata + tensors + padding + data)
        return
    data_start = _align_offset(
        len(header) + len(metadata) + len(tensors),
        GGUF_DEFAULT_ALIGNMENT,
    )
    padding = b"\x00" * (data_start - len(header) - len(metadata) - len(tensors))
    max_payload_end = max(
        offset + _tensor_nbytes(dims, tensor_type)
        for _, dims, tensor_type, offset in tensor_specs
    )
    with path.open("wb") as fp:
        fp.write(header + metadata + tensors + padding)
        if max_payload_end > 0:
            fp.seek(data_start + max_payload_end - 1)
            fp.write(b"\x00")
