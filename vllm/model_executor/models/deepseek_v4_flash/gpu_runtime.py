# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch

from .compressed_kv import (
    DeepSeekV4CompressedKVCache,
    DeepSeekV4CompressedKVLayout,
    DeepSeekV4KVPageAllocator,
)
from .config import DEEPSEEK_V4_FLASH_SHAPE


@dataclass(frozen=True)
class DeepSeekV4FlashGPUCacheConfig:
    context_length: int
    hidden_size: int
    batch_size: int = 1
    kv_width: int = DEEPSEEK_V4_FLASH_SHAPE.head_dim
    dtype: torch.dtype = torch.float16
    device: torch.device | str | None = None

    def __post_init__(self) -> None:
        if self.batch_size != 1:
            raise ValueError(
                "DeepSeek V4 Flash GPU runtime currently supports batch_size=1; "
                f"got {self.batch_size}"
            )
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.kv_width <= 0:
            raise ValueError("kv_width must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")


class DeepSeekV4FlashGPURequestState:
    """Request-local DeepSeek V4 Flash GPU cache state.

    This object owns only per-request runtime tensors. Model weights stay in
    the GPU weight stager/cache and are intentionally not referenced here.
    """

    def __init__(self, config: DeepSeekV4FlashGPUCacheConfig) -> None:
        if config.batch_size != 1:
            raise ValueError(
                "DeepSeek V4 Flash GPU runtime currently supports batch_size=1; "
                f"got {config.batch_size}"
            )
        self.config = config
        self.layout = DeepSeekV4CompressedKVLayout(
            context_length=config.context_length,
        )
        self.page_allocator = DeepSeekV4KVPageAllocator(self.layout)
        device = config.device
        if device is None:
            device = torch.device("cuda")

        self.raw_kv_cache = DeepSeekV4CompressedKVCache(
            context_length=config.context_length,
            hidden_size=config.kv_width,
            raw_window=DEEPSEEK_V4_FLASH_SHAPE.sliding_window,
            num_layers=DEEPSEEK_V4_FLASH_SHAPE.num_layers,
            dtype=config.dtype,
            device=device,
        )
        self.compressed_kv_cache = self.raw_kv_cache
        self.token_position = 0
        self._moe_workspace: dict[tuple[int, int, torch.device], torch.Tensor] = {}

    def advance_token(self) -> None:
        self.require_capacity(self.token_position)
        self.token_position += 1

    def require_capacity(self, position: int) -> None:
        if position < 0:
            raise ValueError("position must be non-negative")
        if position >= self.config.context_length:
            raise ValueError(
                f"position {position} exceeds context_length "
                f"{self.config.context_length}"
            )

    def moe_workspace(
        self,
        *,
        num_experts: int,
        intermediate_size: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = self.config.device
        if device is None:
            device = torch.device("cuda")
        key = (num_experts, intermediate_size, device)
        cached = self._moe_workspace.get(key)
        if cached is not None:
            return cached
        workspace = torch.empty(
            (num_experts, intermediate_size),
            dtype=torch.float32,
            device=device,
        )
        self._moe_workspace[key] = workspace
        return workspace

    def reset(self) -> None:
        self.token_position = 0
        self.raw_kv_cache.raw_token_indices.fill_(-1)
        self.raw_kv_cache.compressed_token_indices.fill_(-1)
        self.raw_kv_cache._compressed_counts.zero_()
        compressor_states = getattr(
            self,
            "_deepseek_v4_flash_compressor_states",
            None,
        )
        if compressor_states is not None:
            compressor_states.clear()
