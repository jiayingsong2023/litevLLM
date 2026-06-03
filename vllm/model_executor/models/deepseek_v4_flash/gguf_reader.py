# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEEPSEEK_V4_FLASH_SHAPE, DeepSeekV4FlashShape

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8


class GGUFParseError(ValueError):
    pass


@dataclass(frozen=True)
class DeepSeekV4FlashTensor:
    name: str
    dims: tuple[int, ...]
    tensor_type: int
    offset: int


@dataclass(frozen=True)
class DeepSeekV4FlashGGUF:
    path: Path
    name: str
    metadata: dict[str, Any]
    tensors: dict[str, DeepSeekV4FlashTensor]
    shape: DeepSeekV4FlashShape = DEEPSEEK_V4_FLASH_SHAPE


class _Cursor:
    def __init__(self, data: memoryview) -> None:
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise GGUFParseError("truncated GGUF")
        out = self.data[self.pos : self.pos + n].tobytes()
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8")


_EXPECTED_METADATA: dict[str, Any] = {
    "general.architecture": "deepseek4",
    "deepseek4.block_count": DEEPSEEK_V4_FLASH_SHAPE.num_layers,
    "deepseek4.attention.head_count": DEEPSEEK_V4_FLASH_SHAPE.num_attention_heads,
    "deepseek4.attention.head_count_kv": DEEPSEEK_V4_FLASH_SHAPE.num_kv_heads,
    "deepseek4.attention.key_length": DEEPSEEK_V4_FLASH_SHAPE.head_dim,
    "deepseek4.attention.sliding_window": DEEPSEEK_V4_FLASH_SHAPE.sliding_window,
    "deepseek4.attention.indexer.head_count": DEEPSEEK_V4_FLASH_SHAPE.indexer_heads,
    "deepseek4.attention.indexer.key_length": DEEPSEEK_V4_FLASH_SHAPE.indexer_head_dim,
    "deepseek4.attention.indexer.top_k": DEEPSEEK_V4_FLASH_SHAPE.indexer_top_k,
    "deepseek4.expert_count": DEEPSEEK_V4_FLASH_SHAPE.num_experts,
    "deepseek4.expert_used_count": DEEPSEEK_V4_FLASH_SHAPE.num_experts_per_tok,
    "deepseek4.embedding_length": DEEPSEEK_V4_FLASH_SHAPE.hidden_size,
    "deepseek4.vocab_size": DEEPSEEK_V4_FLASH_SHAPE.vocab_size,
}

def _read_value(cursor: _Cursor, value_type: int) -> Any:
    if value_type == GGUF_TYPE_UINT32:
        return cursor.u32()
    if value_type == GGUF_TYPE_STRING:
        return cursor.string()
    raise GGUFParseError(f"unsupported GGUF metadata type: {value_type}")


def _read_metadata_entry(cursor: _Cursor) -> tuple[str, Any]:
    key = cursor.string()
    value_type = cursor.u32()
    return key, _read_value(cursor, value_type)


def _validate_metadata(metadata: dict[str, Any]) -> None:
    for key, expected in _EXPECTED_METADATA.items():
        actual = metadata.get(key)
        if actual != expected:
            raise GGUFParseError(
                f"{key.removeprefix('deepseek4.')} must be {expected}; got {actual}"
            )

    name = metadata.get("general.name")
    if not isinstance(name, str) or not name:
        raise GGUFParseError("general.name must be a non-empty string")


def _read_tensor(cursor: _Cursor) -> DeepSeekV4FlashTensor:
    name = cursor.string()
    n_dims = cursor.u32()
    dims = tuple(cursor.u64() for _ in range(n_dims))
    tensor_type = cursor.u32()
    offset = cursor.u64()
    return DeepSeekV4FlashTensor(
        name=name,
        dims=dims,
        tensor_type=tensor_type,
        offset=offset,
    )


def _read_from_view(path: Path, data: memoryview) -> DeepSeekV4FlashGGUF:
    cursor = _Cursor(data)
    magic = cursor.u32()
    if magic != GGUF_MAGIC:
        raise GGUFParseError(f"invalid GGUF magic: 0x{magic:08x}")

    version = cursor.u32()
    if version != GGUF_VERSION:
        raise GGUFParseError(f"unsupported GGUF version: {version}")

    metadata_count = cursor.u64()
    tensor_count = cursor.u64()

    metadata: dict[str, Any] = {}
    for _ in range(metadata_count):
        key, value = _read_metadata_entry(cursor)
        metadata[key] = value

    _validate_metadata(metadata)

    tensors: dict[str, DeepSeekV4FlashTensor] = {}
    for _ in range(tensor_count):
        tensor = _read_tensor(cursor)
        tensors[tensor.name] = tensor

    return DeepSeekV4FlashGGUF(
        path=path,
        name=metadata["general.name"],
        metadata=metadata,
        tensors=tensors,
    )


def read_deepseek_v4_flash_gguf(path: Path | str) -> DeepSeekV4FlashGGUF:
    gguf_path = Path(path)
    with gguf_path.open("rb") as fp:
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        view = memoryview(mm)
        try:
            return _read_from_view(gguf_path, view)
        finally:
            view.release()
            mm.close()
