# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class PrefixCacheEntry:
    key: tuple[int, ...]
    prompt_len: int
    used_blocks: int
    k_blocks: list[torch.Tensor]
    v_blocks: list[torch.Tensor]
    k_scale_blocks: list[torch.Tensor | None]
    v_scale_blocks: list[torch.Tensor | None]
    last_prompt_logits: torch.Tensor


class PrefixCache:
    def __init__(self, max_entries: int = 8) -> None:
        self.max_entries = max(0, int(max_entries))
        self._entries: OrderedDict[tuple[int, ...], PrefixCacheEntry] = OrderedDict()
        self.lookup_count = 0
        self.exact_hit_count = 0
        self.partial_hit_count = 0
        self.miss_count = 0
        self.lookup_candidates_total = 0
        self.lookup_comparisons = 0

    def get(self, key: tuple[int, ...]) -> PrefixCacheEntry | None:
        self.lookup_count += 1
        entry = self._entries.get(key)
        if entry is None:
            self.miss_count += 1
            return None
        self.exact_hit_count += 1
        self.lookup_candidates_total += 1
        self.lookup_comparisons += len(key)
        self._entries.move_to_end(key)
        return entry

    def put(self, entry: PrefixCacheEntry) -> None:
        if self.max_entries == 0:
            return
        self._entries[entry.key] = entry
        self._entries.move_to_end(entry.key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def get_longest_prefix(
        self,
        key: tuple[int, ...],
    ) -> tuple[PrefixCacheEntry | None, int]:
        self.lookup_count += 1
        if not key:
            self.miss_count += 1
            return None, 0

        best_entry: PrefixCacheEntry | None = None
        best_prefix_len = 0
        first_token = key[0]
        candidates = [
            entry
            for entry in self._entries.values()
            if entry.key and entry.key[0] == first_token
        ]
        self.lookup_candidates_total += len(candidates)
        for entry in candidates:
            prefix_len = 0
            max_len = min(len(key), len(entry.key))
            while prefix_len < max_len and key[prefix_len] == entry.key[prefix_len]:
                prefix_len += 1
                self.lookup_comparisons += 1
            if prefix_len < max_len:
                self.lookup_comparisons += 1
            if prefix_len > best_prefix_len:
                best_entry = entry
                best_prefix_len = prefix_len

        if best_entry is not None and best_prefix_len > 0:
            if best_prefix_len == len(key):
                self.exact_hit_count += 1
            else:
                self.partial_hit_count += 1
            self._entries.move_to_end(best_entry.key)
            return best_entry, best_prefix_len
        self.miss_count += 1
        return None, 0

    def stats(self) -> dict[str, int | float]:
        hit_count = self.exact_hit_count + self.partial_hit_count
        return {
            "entries": len(self._entries),
            "capacity": self.max_entries,
            "lookups": self.lookup_count,
            "exact_hits": self.exact_hit_count,
            "partial_hits": self.partial_hit_count,
            "misses": self.miss_count,
            "lookup_candidates_total": self.lookup_candidates_total,
            "lookup_comparisons": self.lookup_comparisons,
            "hit_rate": (hit_count / self.lookup_count) if self.lookup_count else 0.0,
            "avg_candidates_per_lookup": (
                self.lookup_candidates_total / self.lookup_count
                if self.lookup_count
                else 0.0
            ),
            "avg_comparisons_per_lookup": (
                self.lookup_comparisons / self.lookup_count
                if self.lookup_count
                else 0.0
            ),
        }

    def clear(self) -> None:
        self._entries.clear()
        self.lookup_count = 0
        self.exact_hit_count = 0
        self.partial_hit_count = 0
        self.miss_count = 0
        self.lookup_candidates_total = 0
        self.lookup_comparisons = 0
