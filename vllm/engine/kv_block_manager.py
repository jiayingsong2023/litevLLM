# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import Any

import torch

from vllm.engine.prefix_cache import PrefixCacheEntry


class KVBlockManager:
    def __init__(
        self,
        *,
        kv_caches: Any,
        kv_scale_caches: Any,
        num_blocks_per_seq: int,
        block_size: int,
    ) -> None:
        self.kv_caches = kv_caches
        self.kv_scale_caches = kv_scale_caches
        self.num_blocks_per_seq = int(num_blocks_per_seq)
        self.block_size = int(block_size)

    @property
    def num_layers(self) -> int:
        return len(self.kv_caches)

    def block_table_for_slot(
        self,
        slot_idx: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        start_block = int(slot_idx) * self.num_blocks_per_seq
        return torch.arange(
            start_block,
            start_block + self.num_blocks_per_seq,
            device=device,
            dtype=torch.int32,
        )

    def capture_prefix_entry(
        self,
        *,
        key: tuple[int, ...],
        slot_idx: int,
        prompt_len: int,
        last_prompt_logits: torch.Tensor,
    ) -> PrefixCacheEntry:
        used_blocks = max(1, math.ceil(int(prompt_len) / self.block_size))
        start_block = int(slot_idx) * self.num_blocks_per_seq
        end_block = start_block + used_blocks

        k_blocks: list[torch.Tensor] = []
        v_blocks: list[torch.Tensor] = []
        k_scale_blocks: list[torch.Tensor | None] = []
        v_scale_blocks: list[torch.Tensor | None] = []
        for layer_idx, (k_cache, v_cache) in enumerate(self.kv_caches):
            k_blocks.append(k_cache[start_block:end_block].clone())
            v_blocks.append(v_cache[start_block:end_block].clone())
            ks, vs = self.kv_scale_caches[layer_idx]
            k_scale_blocks.append(None if ks is None else ks[start_block:end_block].clone())
            v_scale_blocks.append(None if vs is None else vs[start_block:end_block].clone())

        return PrefixCacheEntry(
            key=key,
            prompt_len=int(prompt_len),
            used_blocks=used_blocks,
            k_blocks=k_blocks,
            v_blocks=v_blocks,
            k_scale_blocks=k_scale_blocks,
            v_scale_blocks=v_scale_blocks,
            last_prompt_logits=last_prompt_logits.detach().clone(),
        )

    def materialize_prefix_entry(
        self,
        *,
        slot_idx: int,
        entry: PrefixCacheEntry,
        prefix_len: int,
    ) -> None:
        prefix_len = max(0, min(int(prefix_len), int(entry.prompt_len)))
        if prefix_len <= 0:
            return
        start_block = int(slot_idx) * self.num_blocks_per_seq
        used_blocks = max(1, math.ceil(prefix_len / self.block_size))
        end_block = start_block + used_blocks

        for layer_idx, (k_cache, v_cache) in enumerate(self.kv_caches):
            k_cache[start_block:end_block].copy_(entry.k_blocks[layer_idx][:used_blocks])
            v_cache[start_block:end_block].copy_(entry.v_blocks[layer_idx][:used_blocks])
            ks, vs = self.kv_scale_caches[layer_idx]
            cached_ks = entry.k_scale_blocks[layer_idx]
            cached_vs = entry.v_scale_blocks[layer_idx]
            if ks is not None and cached_ks is not None:
                ks[start_block:end_block].copy_(cached_ks[:used_blocks])
            if vs is not None and cached_vs is not None:
                vs[start_block:end_block].copy_(cached_vs[:used_blocks])
