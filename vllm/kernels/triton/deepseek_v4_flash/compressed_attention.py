# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

_PAGE_TABLE_ERROR = "DeepSeek V4 compressed attention requires page table inputs"


@dataclass(frozen=True)
class DeepSeekV4CompressedAttentionInputs:
    """Kernel input contract for DeepSeek V4 compressed attention.

    Memory layout:
    - raw_page_table maps logical raw SWA rows to raw page chunks.
    - compressed_page_table maps logical compressed rows to compressed chunks.
    - indexer_page_table maps ratio-4 indexer rows to indexer chunks.
    - selected_rows contains compressed logical row ids selected by the indexer.

    Tiling:
    - Kernel implementations must tile over selected logical rows and resolve
      physical addresses through page tables.
    - The contract rejects a single contiguous full-context compressed cache.
    """

    raw_page_table_name: str
    compressed_page_table_name: str
    indexer_page_table_name: str
    selected_rows_name: str

    def __post_init__(self) -> None:
        names = (
            self.raw_page_table_name,
            self.compressed_page_table_name,
            self.indexer_page_table_name,
        )
        if any("contiguous" in name or "full_context" in name for name in names):
            raise ValueError(_PAGE_TABLE_ERROR)
        if not self.uses_page_tables:
            raise ValueError(_PAGE_TABLE_ERROR)
        if not self.selected_rows_name:
            raise ValueError("selected_rows_name must be non-empty")

    @property
    def uses_page_tables(self) -> bool:
        return all(
            "page_table" in name
            for name in (
                self.raw_page_table_name,
                self.compressed_page_table_name,
                self.indexer_page_table_name,
            )
        )
