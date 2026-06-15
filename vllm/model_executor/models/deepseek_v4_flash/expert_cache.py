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
class DeepSeekV4FlashHotExpertPolicy:
    pinned_experts: frozenset[tuple[int, int]] = frozenset()

    def is_pinned_expert(self, layer_idx: int, expert_id: int) -> bool:
        return (layer_idx, expert_id) in self.pinned_experts
