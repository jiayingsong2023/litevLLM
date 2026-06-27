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
class DeepSeekV4FlashExpertPrefetchRequest:
    layer_idx: int
    expert_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.layer_idx < 0:
            raise ValueError("layer_idx must be non-negative")
        if any(expert_id < 0 for expert_id in self.expert_ids):
            raise ValueError("expert_ids must be non-negative")


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
