# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, Optional, Generic, TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")

@dataclass
class CacheInfo:
    hits: int = 0
    misses: int = 0
    size: int = 0

class LRUCache(Generic[_K, _V]):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
    def get(self, key: _K) -> Optional[_V]: return None
    def put(self, key: _K, value: _V): pass
    def popitem(self, remove_pinned: bool = False): pass
