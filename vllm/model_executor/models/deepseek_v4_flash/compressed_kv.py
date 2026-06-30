# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch

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


class DeepSeekV4CompressedKVCache:
    """Correctness-first batch=1 raw SWA KV cache.

    The raw path keeps only the sliding-window rows for every layer. Logical
    token indices live beside the ring buffer so reads can return rows in token
    order after wraparound.
    """

    def __init__(
        self,
        *,
        context_length: int,
        hidden_size: int,
        raw_window: int = DEEPSEEK_V4_FLASH_SHAPE.sliding_window,
        num_layers: int = DEEPSEEK_V4_FLASH_SHAPE.num_layers,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        DeepSeekV4FlashMemoryPolicy().validate_context_length(context_length)
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if raw_window != DEEPSEEK_V4_FLASH_SHAPE.sliding_window:
            raise ValueError(
                "DeepSeek V4 Flash first release requires raw_window="
                f"{DEEPSEEK_V4_FLASH_SHAPE.sliding_window}"
            )
        if num_layers <= 0 or num_layers > DEEPSEEK_V4_FLASH_SHAPE.num_layers:
            raise ValueError(f"num_layers out of range: {num_layers}")

        self.context_length = context_length
        self.hidden_size = hidden_size
        self.raw_window = raw_window
        self.num_layers = num_layers
        self.raw_keys = torch.empty(
            (num_layers, raw_window, hidden_size),
            dtype=dtype,
            device=device,
        )
        self.raw_values = torch.empty_like(self.raw_keys)
        self.raw_token_indices = torch.full(
            (num_layers, raw_window),
            -1,
            dtype=torch.long,
            device=device,
        )
        max_compressed_rows = context_length // 4 + 2
        self.compressed_rows = torch.empty(
            (num_layers, max_compressed_rows, hidden_size),
            dtype=dtype,
            device=device,
        )
        self.compressed_token_indices = torch.full(
            (num_layers, max_compressed_rows),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.indexer_rows = torch.empty(
            (
                num_layers,
                max_compressed_rows,
                DEEPSEEK_V4_FLASH_SHAPE.indexer_head_dim,
            ),
            dtype=dtype,
            device=device,
        )
        self._compressed_counts = torch.zeros(
            (num_layers,),
            dtype=torch.long,
            device=device,
        )
        self._compressed_counts_cpu: list[int] = [0] * num_layers

    def append_raw(
        self,
        layer_idx: int,
        token_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self._validate_layer(layer_idx)
        self._validate_token_idx(token_idx)
        self._validate_raw_row("key", key)
        self._validate_raw_row("value", value)

        slot = token_idx % self.raw_window
        self.raw_keys[layer_idx, slot].copy_(key)
        self.raw_values[layer_idx, slot].copy_(value)
        self.raw_token_indices[layer_idx, slot].copy_(
            torch.full(
                (),
                token_idx,
                dtype=torch.long,
                device=self.raw_token_indices.device,
            )
        )

    def read_raw_window(
        self,
        layer_idx: int,
        token_idx: int,
        window: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_layer(layer_idx)
        self._validate_token_idx(token_idx)
        if window <= 0:
            raise ValueError("window must be positive")
        if window > self.raw_window:
            raise ValueError(f"window {window} exceeds raw_window {self.raw_window}")

        start = max(0, token_idx - window + 1)
        token_indices = self.raw_token_indices[layer_idx]
        present = (token_indices >= start) & (token_indices <= token_idx)
        slots = torch.nonzero(present, as_tuple=False).flatten()
        if slots.numel() == 0:
            empty = torch.empty(
                (0, self.hidden_size),
                dtype=self.raw_keys.dtype,
                device=self.raw_keys.device,
            )
            return empty, empty.clone()

        order = torch.argsort(token_indices[slots])
        sorted_slots = slots[order]
        return (
            self.raw_keys[layer_idx, sorted_slots],
            self.raw_values[layer_idx, sorted_slots],
        )

    def append_compressed(
        self,
        layer_idx: int,
        token_idx: int,
        row: torch.Tensor,
        *,
        indexer_row: torch.Tensor | None = None,
    ) -> int:
        self._validate_layer(layer_idx)
        self._validate_token_idx(token_idx)
        self._validate_raw_row("compressed row", row)
        if indexer_row is not None:
            if indexer_row.shape != (DEEPSEEK_V4_FLASH_SHAPE.indexer_head_dim,):
                raise ValueError(
                    "indexer row shape must be "
                    f"({DEEPSEEK_V4_FLASH_SHAPE.indexer_head_dim},); "
                    f"got {tuple(indexer_row.shape)}"
                )
            if indexer_row.dtype != self.indexer_rows.dtype:
                raise ValueError(
                    f"indexer row dtype must be {self.indexer_rows.dtype}; "
                    f"got {indexer_row.dtype}"
                )
            if indexer_row.device != self.indexer_rows.device:
                raise ValueError(
                    f"indexer row device must be {self.indexer_rows.device}; "
                    f"got {indexer_row.device}"
                )
        slot = int(self._compressed_counts[layer_idx].item())
        if slot >= self.compressed_rows.shape[1]:
            raise ValueError("compressed cache capacity exceeded")
        self.compressed_rows[layer_idx, slot].copy_(row)
        self.compressed_token_indices[layer_idx, slot] = token_idx
        if indexer_row is not None:
            self.indexer_rows[layer_idx, slot].copy_(indexer_row)
        self._compressed_counts[layer_idx] += 1
        self._compressed_counts_cpu[layer_idx] += 1
        return slot

    def read_compressed(
        self,
        layer_idx: int,
        *,
        row_indices: torch.Tensor | None = None,
        count: int | None = None,
    ) -> torch.Tensor:
        self._validate_layer(layer_idx)
        provided_count = count
        if count is None:
            count = self._compressed_counts_cpu[layer_idx]
        if row_indices is None:
            return self.compressed_rows[layer_idx, :count]
        if row_indices.ndim != 1:
            raise ValueError(f"row_indices must be 1-D; got {row_indices.ndim}-D")
        if row_indices.numel() == 0:
            return self.compressed_rows[layer_idx, :0]
        if provided_count is None and (
            torch.any(row_indices < 0) or torch.any(row_indices >= count)
        ):
            raise ValueError("compressed row index out of range")
        return self.compressed_rows[layer_idx, row_indices.to(torch.long)]

    def read_indexer_rows(
        self,
        layer_idx: int,
        *,
        count: int | None = None,
    ) -> torch.Tensor:
        self._validate_layer(layer_idx)
        if count is None:
            count = self._compressed_counts_cpu[layer_idx]
        return self.indexer_rows[layer_idx, :count]

    def _validate_layer(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer index out of range: {layer_idx}")

    def _validate_token_idx(self, token_idx: int) -> None:
        if token_idx < 0:
            raise ValueError("token index must be non-negative")
        if token_idx >= self.context_length:
            raise ValueError(
                f"token index {token_idx} exceeds context_length {self.context_length}"
            )

    def _validate_raw_row(self, name: str, row: torch.Tensor) -> None:
        if row.shape != (self.hidden_size,):
            raise ValueError(
                f"{name} shape must be ({self.hidden_size},); got {tuple(row.shape)}"
            )
        if row.dtype != self.raw_keys.dtype:
            raise ValueError(
                f"{name} dtype must be {self.raw_keys.dtype}; got {row.dtype}"
            )
        if row.device != self.raw_keys.device:
            raise ValueError(
                f"{name} device must be {self.raw_keys.device}; got {row.device}"
            )


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
