from __future__ import annotations

import struct
from collections.abc import Callable
from pathlib import Path

GGUF_MAGIC = 0x46554747
DEEPSEEK_V4_FLASH_METADATA_COUNT = 14
GGUF_DEFAULT_ALIGNMENT = 32


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


def write_minimal_deepseek_v4_flash_gguf(
    path: Path,
    *,
    block_count: int = 43,
    tensor_names: tuple[str, ...] = ("token_embd.weight",),
    tensor_types: tuple[int, ...] | None = None,
    tensor_dims: dict[str, tuple[int, ...]] | None = None,
    tensor_payloads: dict[str, bytes] | None = None,
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
    if tensor_payloads is None:
        tensor_payloads = {}

    tensor_specs: list[tuple[str, tuple[int, ...], int, int]] = []
    for offset, name in enumerate(tensor_names):
        if tensor_dims is not None and name in tensor_dims:
            dims = tensor_dims[name]
        else:
            dims = (4096, 129280) if name == "token_embd.weight" else (4096, 4096)
        tensor_specs.append((name, dims, tensor_types[offset], offset * 32))
        _write_tensor(tensors, name, dims, tensor_types[offset], offset * 32)
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
        for name, dims, tensor_type, _ in tensor_specs:
            offset = len(data)
            payload = tensor_payloads.get(name, b"")
            data.extend(payload)
            _write_tensor(tensors, name, dims, tensor_type, offset)
        padding = b"\x00" * (data_start - len(header) - len(metadata) - len(tensors))
        path.write_bytes(header + metadata + tensors + padding + data)
        return
    path.write_bytes(header + metadata + tensors + b"\x00" * 64)
