# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from vllm.engine.block_allocator import BlockAllocator
from vllm.engine.prefix_cache import PrefixCacheEntry
from vllm.utils.math_utils import cdiv


class KVBlockManager:
    def __init__(
        self,
        *,
        kv_caches: Any,
        kv_scale_caches: Any,
        num_blocks_per_seq: int,
        block_size: int,
        max_active_requests: int,
        block_allocator: BlockAllocator,
    ) -> None:
        self.kv_caches = kv_caches
        self.kv_scale_caches = kv_scale_caches
        self.num_blocks_per_seq = int(num_blocks_per_seq)
        self.block_size = int(block_size)
        self._allocator = block_allocator
        self._request_blocks: dict[str, list[int]] = {}
        self._slot_request_id: dict[int, str] = {}
        self._request_slot_id: dict[str, int] = {}
        device = kv_caches[0][0].device if kv_caches else torch.device("cpu")
        self._block_table_buffer = torch.zeros(
            (max_active_requests, self.num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

    @property
    def num_layers(self) -> int:
        return len(self.kv_caches)

    def ensure_blocks(self, request_id: str, num_tokens: int) -> int:
        """Allocate additional blocks if needed. Return newly allocated count."""
        needed = cdiv(int(num_tokens), self.block_size)
        current_blocks = self._request_blocks.get(request_id)
        current_len = len(current_blocks) if current_blocks is not None else 0
        if needed <= current_len:
            return 0

        new_ids = self._allocator.allocate(needed - current_len)
        if current_blocks is None:
            self._request_blocks[request_id] = new_ids
        else:
            current_blocks.extend(new_ids)

        # Fresh blocks may contain stale data from earlier sequences.
        for k_cache, v_cache in self.kv_caches:
            k_cache[new_ids] = 0
            v_cache[new_ids] = 0
        for ks_cache, vs_cache in self.kv_scale_caches:
            if ks_cache is not None:
                ks_cache[new_ids] = 0
            if vs_cache is not None:
                vs_cache[new_ids] = 0

        return len(new_ids)

    def ensure_blocks_for_requests(
        self,
        request_ids: list[str],
        token_counts: list[int],
    ) -> None:
        """Grow the block tables for a batch of requests."""
        for rid, token_count in zip(request_ids, token_counts):
            self.ensure_blocks(rid, token_count)

    def free_request_blocks(self, request_id: str) -> None:
        """Free all blocks held by a request and clear its block-table row."""
        block_ids = self._request_blocks.pop(request_id, [])
        if block_ids:
            self._allocator.free(block_ids)

        slot_idx = self._request_slot_id.pop(request_id, None)
        if slot_idx is not None:
            self._block_table_buffer[slot_idx].zero_()
            self._slot_request_id.pop(slot_idx, None)

    def block_table_for_slot(self, slot_idx: int) -> torch.Tensor:
        """Return the device's block-table row for the given slot."""
        return self._block_table_buffer[slot_idx]

    def update_block_table_row(self, slot_idx: int, request_id: str) -> None:
        """Copy request block IDs into the shared block-table buffer row."""
        row = self._block_table_buffer[slot_idx]
        row.zero_()

        block_ids = self._request_blocks.get(request_id, [])
        if block_ids:
            row[: len(block_ids)].copy_(
                torch.tensor(
                    block_ids,
                    dtype=torch.int32,
                    device=row.device,
                )
            )
            self._slot_request_id[slot_idx] = request_id
            self._request_slot_id[request_id] = slot_idx
        else:
            self._slot_request_id.pop(slot_idx, None)
            self._request_slot_id.pop(request_id, None)

    def capture_prefix_entry(
        self,
        *,
        key: tuple[int, ...],
        request_id: str,
        prompt_len: int,
        last_prompt_logits: torch.Tensor,
    ) -> PrefixCacheEntry:
        used_blocks = max(1, cdiv(int(prompt_len), self.block_size))
        block_ids = self._request_blocks.get(request_id, [])
        if len(block_ids) < used_blocks:
            raise RuntimeError(
                f"Request {request_id} has {len(block_ids)} blocks, "
                f"but {used_blocks} are needed to capture prefix of length {prompt_len}"
            )
        selected_ids = block_ids[:used_blocks]

        k_blocks: list[torch.Tensor] = []
        v_blocks: list[torch.Tensor] = []
        k_scale_blocks: list[torch.Tensor | None] = []
        v_scale_blocks: list[torch.Tensor | None] = []
        for layer_idx, (k_cache, v_cache) in enumerate(self.kv_caches):
            k_blocks.append(k_cache[selected_ids].clone())
            v_blocks.append(v_cache[selected_ids].clone())
            ks, vs = self.kv_scale_caches[layer_idx]
            k_scale_blocks.append(None if ks is None else ks[selected_ids].clone())
            v_scale_blocks.append(None if vs is None else vs[selected_ids].clone())

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
        request_id: str,
        entry: PrefixCacheEntry,
        prefix_len: int,
    ) -> None:
        prefix_len = max(0, min(int(prefix_len), int(entry.prompt_len)))
        if prefix_len <= 0:
            return
        used_blocks = max(1, cdiv(prefix_len, self.block_size))
        self.ensure_blocks(request_id, prefix_len)
        dst_block_ids = self._request_blocks[request_id][:used_blocks]

        for layer_idx, (k_cache, v_cache) in enumerate(self.kv_caches):
            k_cache[dst_block_ids] = entry.k_blocks[layer_idx][:used_blocks]
            v_cache[dst_block_ids] = entry.v_blocks[layer_idx][:used_blocks]
            ks, vs = self.kv_scale_caches[layer_idx]
            cached_ks = entry.k_scale_blocks[layer_idx]
            cached_vs = entry.v_scale_blocks[layer_idx]
            if ks is not None and cached_ks is not None:
                ks[dst_block_ids] = cached_ks[:used_blocks]
            if vs is not None and cached_vs is not None:
                vs[dst_block_ids] = cached_vs[:used_blocks]
