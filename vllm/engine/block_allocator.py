# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import deque
from collections.abc import Iterable


class BlockAllocator:
    """Manages a pool of physical KV cache block IDs.

    Block ID 0 is reserved as the zeroed null block and is never handed out.
    """

    def __init__(self, num_total_blocks: int) -> None:
        self.num_total_blocks = int(num_total_blocks)
        if self.num_total_blocks < 1:
            raise ValueError("num_total_blocks must be positive")
        # Reserve ID 0; hand out IDs 1..N-1.
        self._free_ids: deque[int] = deque(range(1, self.num_total_blocks))

    def allocate(self, n: int) -> list[int]:
        n = int(n)
        if n < 0:
            raise ValueError("n must be non-negative")
        if n > len(self._free_ids):
            raise RuntimeError(
                f"Cannot allocate {n} blocks; only {len(self._free_ids)} free"
            )
        return [self._free_ids.popleft() for _ in range(n)]

    def free(self, block_ids: Iterable[int]) -> None:
        for bid in block_ids:
            bid = int(bid)
            if bid <= 0 or bid >= self.num_total_blocks:
                raise ValueError(f"Invalid block id to free: {bid}")
            self._free_ids.append(bid)

    @property
    def num_free(self) -> int:
        return len(self._free_ids)

    def can_allocate(self, n: int) -> bool:
        return int(n) <= len(self._free_ids)
