# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


@dataclass(frozen=True)
class DeepSeekV4CacheUpdateInputs:
    """Kernel input contract for DeepSeek V4 paged compressed KV updates.

    Memory layout:
    - page_table maps logical rows to physical pages.
    - kv_row is [latent_kv_dim] for the current token.
    - cache_storage is a paged backing tensor, never a single full-context row.

    Tiling:
    - Future kernels should map one program to one logical row update.
    - Physical addresses are resolved through page_table before writing.
    """

    page_table: torch.Tensor
    kv_row: torch.Tensor
    cache_storage: torch.Tensor
    logical_row: int

    def __post_init__(self) -> None:
        if self.page_table.ndim != 1:
            raise ValueError(f"page_table must be 1-D; got {self.page_table.ndim}-D")
        if self.kv_row.ndim != 1:
            raise ValueError(f"kv_row must be 1-D; got {self.kv_row.ndim}-D")
        if self.cache_storage.ndim < 2:
            raise ValueError("cache_storage must have at least 2 dimensions")
        if self.logical_row < 0:
            raise ValueError("logical_row must be non-negative")


@triton.jit
def _cache_update_kernel(
    page_table_ptr,
    kv_row_ptr,
    cache_storage_ptr,
    logical_row: tl.constexpr,
    rows_per_page: tl.constexpr,
    row_width: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
) -> None:
    # Memory layout:
    # - page_table maps logical rows to physical cache pages.
    # - kv_row is [row_width].
    # - cache_storage is [num_pages, rows_per_page, row_width].
    # Tiling:
    # - one program copies BLOCK_WIDTH contiguous columns for one logical row.
    # - the physical page lookup stays on GPU, avoiding Tensor.item() sync.
    block = tl.program_id(0)
    offsets = block * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
    mask = offsets < row_width
    physical_page = tl.load(page_table_ptr + logical_row).to(tl.int64)
    row_in_page = logical_row % rows_per_page
    cache_base = (physical_page * rows_per_page + row_in_page) * row_width
    values = tl.load(kv_row_ptr + offsets, mask=mask, other=0.0)
    tl.store(cache_storage_ptr + cache_base + offsets, values, mask=mask)


def deepseek_v4_cache_update(inputs: DeepSeekV4CacheUpdateInputs) -> None:
    tensors = (inputs.page_table, inputs.kv_row, inputs.cache_storage)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("DeepSeek V4 cache update inputs must be CUDA tensors")
    if inputs.logical_row >= inputs.page_table.numel():
        raise ValueError("logical_row exceeds page_table length")
    rows_per_page = inputs.cache_storage.shape[1]
    if inputs.cache_storage.shape[-1] != inputs.kv_row.numel():
        raise ValueError("cache row width must match kv_row width")
    row_width = inputs.kv_row.numel()
    block_width = min(1024, triton.next_power_of_2(row_width))
    grid = (triton.cdiv(row_width, block_width),)
    _cache_update_kernel[grid](
        inputs.page_table,
        inputs.kv_row.contiguous(),
        inputs.cache_storage,
        logical_row=inputs.logical_row,
        rows_per_page=rows_per_page,
        row_width=row_width,
        BLOCK_WIDTH=block_width,
        num_warps=8,
    )
