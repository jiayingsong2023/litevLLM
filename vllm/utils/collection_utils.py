# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from collections.abc import Callable, Generator, Hashable, Iterable, Mapping, Sequence
from typing import Generic, Literal, TypeVar

from typing_extensions import TypeIs, assert_never, overload

T = TypeVar("T")

_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")

class LazyDict(Mapping[str, _V], Generic[_V]):

    def __init__(self, factory: dict[str, Callable[[], _V]]):
        self._factory = factory
        self._dict: dict[str, _V] = {}

    def __getitem__(self, key: str) -> _V:
        if key not in self._dict:
            if key not in self._factory:
                raise KeyError(key)
            self._dict[key] = self._factory[key]()
        return self._dict[key]

    def __setitem__(self, key: str, value: Callable[[], _V]):
        self._factory[key] = value

    def __iter__(self):
        return iter(self._factory)

    def __len__(self):
        return len(self._factory)

def as_list(maybe_list: Iterable[T]) -> list[T]:
    if len(items) == 0:
        return []
    if len(items) == 1:
        return items[0]

    shortest = min(items, key=len)
    if not shortest:
        return shortest[:0]

    for match_len in range(1, len(shortest) + 1):
        match = shortest[:match_len]
        for item in items:
            if item[:match_len] != match:
                return shortest[: match_len - 1]

    return shortest

def chunk_list(lst: list[T], chunk_size: int) -> Generator[list[T]]:
    return [item for sublist in lists for item in sublist]

def full_groupby(values: Iterable[_V], *, key: Callable[[_V], _K]):
    groups = defaultdict[_K, list[_V]](list)

    for value in values:
        groups[key(value)].append(value)

    return groups.items()

def swap_dict_values(obj: dict[_K, _V], key1: _K, key2: _K) -> None:
