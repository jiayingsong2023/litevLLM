# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.model_executor.models.gemma4 import (
    _build_local_decode_aligned_metadata,
    _get_or_build_local_decode_aligned_metadata,
)


def test_local_decode_aligned_metadata_with_window():
    block_tables = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 25],
        ],
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([70, 48], dtype=torch.int32)
    # block_size=16:
    # req0: start=70-20=50 -> aligned_start=48 -> aligned_len=22 -> blocks [3,4]
    # req1: start=48-20=28 -> aligned_start=16 -> aligned_len=32 -> blocks [1,2]
    seq_lens_local, block_tables_local = _build_local_decode_aligned_metadata(
        block_tables=block_tables,
        seq_lens=seq_lens,
        local_window=20,
        block_size=16,
    )
    assert seq_lens_local.tolist() == [22, 32]
    assert block_tables_local.shape == (2, 2)
    assert block_tables_local[0].tolist() == [13, 14]
    assert block_tables_local[1].tolist() == [21, 22]


def test_local_decode_aligned_metadata_without_window():
    block_tables = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    seq_lens = torch.tensor([33], dtype=torch.int32)
    seq_lens_local, block_tables_local = _build_local_decode_aligned_metadata(
        block_tables=block_tables,
        seq_lens=seq_lens,
        local_window=0,
        block_size=16,
    )
    # Full history from the beginning.
    assert seq_lens_local.tolist() == [33]
    assert block_tables_local.tolist() == [[1, 2, 3]]


def test_local_decode_aligned_metadata_cache_hits_for_same_key():
    block_tables = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    seq_lens = torch.tensor([33], dtype=torch.int32)
    meta: dict[str, object] = {}
    s1, b1 = _get_or_build_local_decode_aligned_metadata(
        attn_metadata=meta,
        block_tables=block_tables,
        seq_lens=seq_lens,
        local_window=20,
        block_size=16,
    )
    s2, b2 = _get_or_build_local_decode_aligned_metadata(
        attn_metadata=meta,
        block_tables=block_tables,
        seq_lens=seq_lens,
        local_window=20,
        block_size=16,
    )
    assert s1 is s2
    assert b1 is b2


def test_local_decode_aligned_metadata_cache_miss_on_window_change():
    block_tables = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    seq_lens = torch.tensor([33], dtype=torch.int32)
    meta: dict[str, object] = {}
    s1, b1 = _get_or_build_local_decode_aligned_metadata(
        attn_metadata=meta,
        block_tables=block_tables,
        seq_lens=seq_lens,
        local_window=20,
        block_size=16,
    )
    s2, b2 = _get_or_build_local_decode_aligned_metadata(
        attn_metadata=meta,
        block_tables=block_tables,
        seq_lens=seq_lens,
        local_window=0,
        block_size=16,
    )
    assert s1 is not s2
    assert b1 is not b2
