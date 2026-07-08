# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekV4FlashCacheKey:
    namespace: str
    name: str
    device: str
    dtype: str
    extra: tuple[int | str, ...] = ()


@dataclass(frozen=True)
class DeepSeekV4FlashCacheAdmissionPolicy:
    min_reuse_score: int = 1
    stream_experts: frozenset[tuple[int, int]] = frozenset()

    def should_cache_grouped_expert(
        self,
        *,
        layer_idx: int | None,
        expert_id: int,
    ) -> bool:
        if layer_idx is not None and (layer_idx, expert_id) in self.stream_experts:
            return False
        return self.min_reuse_score <= 1


@dataclass(frozen=True)
class DeepSeekV4FlashHotExpertPolicy:
    pinned_experts: frozenset[tuple[int, int]] = frozenset()

    def is_pinned_expert(self, layer_idx: int, expert_id: int) -> bool:
        return (layer_idx, expert_id) in self.pinned_experts
