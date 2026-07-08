# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch

from .compressed_attention_triton import deepseek_v4_compressed_attention_triton

_PAGE_TABLE_ERROR = "DeepSeek V4 compressed attention requires page table inputs"


@dataclass(frozen=True)
class DeepSeekV4CompressedAttentionInputs:
    """Kernel input contract for DeepSeek V4 compressed attention.

    Memory layout:
    - raw_page_table maps logical raw SWA rows to raw page chunks.
    - compressed_page_table maps logical compressed rows to compressed chunks.
    - indexer_page_table maps ratio-4 indexer rows to indexer chunks.
    - selected_rows contains compressed logical row ids selected by the indexer.

    Tiling:
    - Kernel implementations must tile over selected logical rows and resolve
      physical addresses through page tables.
    - The contract rejects a single contiguous full-context compressed cache.
    """

    raw_page_table_name: str
    compressed_page_table_name: str
    indexer_page_table_name: str
    selected_rows_name: str

    def __post_init__(self) -> None:
        names = (
            self.raw_page_table_name,
            self.compressed_page_table_name,
            self.indexer_page_table_name,
        )
        if any("contiguous" in name or "full_context" in name for name in names):
            raise ValueError(_PAGE_TABLE_ERROR)
        if not self.uses_page_tables:
            raise ValueError(_PAGE_TABLE_ERROR)
        if not self.selected_rows_name:
            raise ValueError("selected_rows_name must be non-empty")

    @property
    def uses_page_tables(self) -> bool:
        return all(
            "page_table" in name
            for name in (
                self.raw_page_table_name,
                self.compressed_page_table_name,
                self.indexer_page_table_name,
            )
        )


@dataclass(frozen=True)
class DeepSeekV4CompressedAttentionTensorInputs:
    """Tensor contract for selected-row compressed attention.

    Memory layout:
    - query is [head_dim].
    - compressed_rows is [num_compressed_rows, head_dim].
    - selected_rows is [num_selected_rows] and indexes compressed_rows.

    Tiling:
    - Future kernels should tile over selected rows only.
    - The caller resolves paged storage into compact selected-row tensors before
      this boundary, so no full-context contiguous allocation is required.
    """

    query: torch.Tensor
    compressed_rows: torch.Tensor
    selected_rows: torch.Tensor

    def __post_init__(self) -> None:
        if self.query.ndim != 1:
            raise ValueError(f"query must be 1-D; got {self.query.ndim}-D")
        if self.compressed_rows.ndim != 2:
            raise ValueError(
                f"compressed_rows must be 2-D; got {self.compressed_rows.ndim}-D"
            )
        if self.compressed_rows.shape[1] != self.query.numel():
            raise ValueError("compressed_rows width must match query size")
        if self.selected_rows.ndim != 1:
            raise ValueError(
                f"selected_rows must be 1-D; got {self.selected_rows.ndim}-D"
            )
        if self.selected_rows.numel() == 0:
            raise ValueError("selected_rows must contain at least one row")


def _deepseek_v4_compressed_attention_reference(
    inputs: DeepSeekV4CompressedAttentionTensorInputs,
) -> torch.Tensor:
    selected = inputs.compressed_rows.index_select(
        0,
        inputs.selected_rows.to(torch.long),
    )
    scores = selected.to(torch.float32).matmul(inputs.query.to(torch.float32))
    scores = scores / math.sqrt(float(inputs.query.numel()))
    probs = torch.softmax(scores, dim=0)
    return probs.matmul(selected.to(torch.float32))


def deepseek_v4_compressed_attention(
    inputs: DeepSeekV4CompressedAttentionTensorInputs,
    *,
    section: Callable[[str], AbstractContextManager[None]] | None = None,
) -> torch.Tensor:
    tensors = (inputs.query, inputs.compressed_rows, inputs.selected_rows)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("DeepSeek V4 compressed attention inputs must be CUDA tensors")
    if (
        os.environ.get(
            "FASTINFERENCE_DEEPSEEK_V4_FLASH_COMPRESSED_ATTN_FALLBACK",
            "0",
        )
        == "1"
    ):
        return _deepseek_v4_compressed_attention_reference(inputs)
    return deepseek_v4_compressed_attention_triton(
        inputs.query,
        inputs.compressed_rows,
        inputs.selected_rows,
        section=section,
    )
