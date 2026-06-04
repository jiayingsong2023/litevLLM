# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

from .config import (
    DEEPSEEK_V4_FLASH_SHAPE,
    DeepSeekV4FlashMemoryPolicy,
    layer_compress_ratio,
)


@dataclass(frozen=True)
class DeepSeekV4PageRef:
    chunk_id: int
    page_id: int
    row_offset: int


@dataclass(frozen=True)
class DeepSeekV4KVPagePool:
    name: str
    page_rows: int
    pages_per_chunk: int
    row_width: int
    bytes_per_value: int

    def __post_init__(self) -> None:
        if self.page_rows <= 0:
            raise ValueError("page_rows must be positive")
        if self.pages_per_chunk <= 0:
            raise ValueError("pages_per_chunk must be positive")
        if self.row_width <= 0:
            raise ValueError("row_width must be positive")
        if self.bytes_per_value <= 0:
            raise ValueError("bytes_per_value must be positive")

    @property
    def rows_per_chunk(self) -> int:
        return self.page_rows * self.pages_per_chunk

    @property
    def max_chunk_bytes(self) -> int:
        return self.rows_per_chunk * self.row_width * self.bytes_per_value

    def resolve(self, logical_row: int) -> DeepSeekV4PageRef:
        if logical_row < 0:
            raise ValueError("logical row must be non-negative")
        chunk_id = logical_row // self.rows_per_chunk
        row_in_chunk = logical_row % self.rows_per_chunk
        return DeepSeekV4PageRef(
            chunk_id=chunk_id,
            page_id=row_in_chunk // self.page_rows,
            row_offset=row_in_chunk % self.page_rows,
        )


@dataclass(frozen=True)
class DeepSeekV4CompressedKVLayout:
    context_length: int
    raw_window: int = DEEPSEEK_V4_FLASH_SHAPE.sliding_window

    def __post_init__(self) -> None:
        DeepSeekV4FlashMemoryPolicy().validate_context_length(self.context_length)
        if self.raw_window != DEEPSEEK_V4_FLASH_SHAPE.sliding_window:
            raise ValueError(
                "DeepSeek V4 Flash first release requires raw_window="
                f"{DEEPSEEK_V4_FLASH_SHAPE.sliding_window}"
            )

    def layer_comp_capacity(self, layer_idx: int) -> int:
        ratio = layer_compress_ratio(layer_idx)
        if ratio == 0:
            return 0
        return self.context_length // ratio + 2

    def has_indexer_cache(self, layer_idx: int) -> bool:
        return layer_compress_ratio(layer_idx) == 4


class DeepSeekV4KVPageAllocator:
    """Logical page mapping for DeepSeek V4 compressed KV.

    This preserves the PagedAttention memory property: logical KV growth is
    resolved through compact page references and chunk pools instead of one
    contiguous full-context KV allocation.
    """

    def __init__(self, layout: DeepSeekV4CompressedKVLayout) -> None:
        self.layout = layout
        self.raw_pool = DeepSeekV4KVPagePool(
            name="raw",
            page_rows=16,
            pages_per_chunk=64,
            row_width=DEEPSEEK_V4_FLASH_SHAPE.head_dim,
            bytes_per_value=4,
        )
        self.compressed_pool = DeepSeekV4KVPagePool(
            name="compressed",
            page_rows=64,
            pages_per_chunk=64,
            row_width=DEEPSEEK_V4_FLASH_SHAPE.head_dim,
            bytes_per_value=2,
        )
        self.indexer_pool = DeepSeekV4KVPagePool(
            name="indexer",
            page_rows=64,
            pages_per_chunk=64,
            row_width=DEEPSEEK_V4_FLASH_SHAPE.indexer_head_dim,
            bytes_per_value=4,
        )

    def allocate_raw_row(self, layer_idx: int, logical_row: int) -> DeepSeekV4PageRef:
        self._validate_layer(layer_idx)
        return self.raw_pool.resolve(logical_row)

    def allocate_compressed_row(
        self,
        layer_idx: int,
        logical_row: int,
    ) -> DeepSeekV4PageRef:
        self._validate_layer(layer_idx)
        if self.layout.layer_comp_capacity(layer_idx) == 0:
            raise ValueError(f"layer {layer_idx} has no compressed KV rows")
        if logical_row >= self.layout.layer_comp_capacity(layer_idx):
            raise ValueError(
                f"compressed logical row {logical_row} exceeds layer {layer_idx} "
                f"capacity {self.layout.layer_comp_capacity(layer_idx)}"
            )
        return self.compressed_pool.resolve(logical_row)

    def allocate_indexer_row(
        self,
        layer_idx: int,
        logical_row: int,
    ) -> DeepSeekV4PageRef:
        self._validate_layer(layer_idx)
        if not self.layout.has_indexer_cache(layer_idx):
            raise ValueError(f"layer {layer_idx} has no ratio-4 indexer cache")
        if logical_row >= self.layout.layer_comp_capacity(layer_idx):
            raise ValueError(
                f"indexer logical row {logical_row} exceeds layer {layer_idx} "
                f"capacity {self.layout.layer_comp_capacity(layer_idx)}"
            )
        return self.indexer_pool.resolve(logical_row)

    def _validate_layer(self, layer_idx: int) -> None:
        layer_compress_ratio(layer_idx)
