# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator, Iterable
from typing import Any, Hashable, TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


def chunk_list(items: list[T], chunk_size: int) -> Generator[list[T], None, None]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def swap_dict_values(obj: dict[K, V], key1: K, key2: K) -> None:
    if key1 not in obj or key2 not in obj:
        raise KeyError("Both keys must exist in the dictionary")
    obj[key1], obj[key2] = obj[key2], obj[key1]


def is_list_of(items: Any, expected_type: type[T]) -> bool:
    return isinstance(items, list) and all(
        isinstance(item, expected_type) for item in items
    )


def as_list(items: Iterable[T] | T) -> list[T]:
    if isinstance(items, list):
        return items
    if isinstance(items, tuple):
        return list(items)
    if isinstance(items, str):
        return [items]  # type: ignore[list-item]
    if isinstance(items, Iterable):
        return list(items)
    return [items]
