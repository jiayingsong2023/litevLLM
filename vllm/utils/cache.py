# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")


@dataclass
class CacheInfo:
    hits: int = 0
    misses: int = 0
    size: int = 0


class LRUCache[K, V]:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}

    def get(self, key: K) -> V | None:
        return None

    def put(self, key: K, value: V):
        pass

    def popitem(self, remove_pinned: bool = False):
        pass
