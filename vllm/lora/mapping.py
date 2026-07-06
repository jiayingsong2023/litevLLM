# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class LoRAMapping:
    adapter_ids: list[str | None]

    @classmethod
    def from_ids(cls, adapter_ids: Sequence[str | None]) -> LoRAMapping:
        return cls([str(item) if item else None for item in adapter_ids])

    @property
    def active_adapter_names(self) -> list[str]:
        return sorted({item for item in self.adapter_ids if item})

    @property
    def adapter_count(self) -> int:
        return len(self.active_adapter_names)

    @property
    def is_mixed(self) -> bool:
        active = self.active_adapter_names
        has_base = any(item is None for item in self.adapter_ids)
        return len(active) > 1 or (bool(active) and has_base)

    def __iter__(self) -> Iterator[str | None]:
        return iter(self.adapter_ids)

    def __len__(self) -> int:
        return len(self.adapter_ids)

    def __getitem__(self, index: int) -> str | None:
        return self.adapter_ids[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LoRAMapping):
            return self.adapter_ids == other.adapter_ids
        if isinstance(other, (list, tuple)):
            return self.adapter_ids == list(other)
        return False
