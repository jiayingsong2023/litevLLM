# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch


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


def deepseek_v4_cache_update(inputs: DeepSeekV4CacheUpdateInputs) -> None:
    del inputs
    raise NotImplementedError(
        "DeepSeek V4 Flash cache update kernel is not implemented"
    )
