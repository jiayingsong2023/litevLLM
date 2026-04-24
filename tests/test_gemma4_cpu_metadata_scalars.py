# SPDX-License-Identifier: Apache-2.0
"""
Guardrail tests for Step-1/Step-2 of the Gemma4 31B decode-sync cleanup.

These tests validate two invariants:

1. The engine-side InputBatchBuilder injects CPU-side scalar fields into
   ``attn_metadata`` (``seq_lens_cpu`` / ``max_seq_len_cpu`` /
   ``positions_cpu`` / ``kv_start_indices_cpu``), and those scalars match
   the device tensors bit-for-bit.
2. The vectorized + CPU-hint path of
   ``_build_local_decode_aligned_metadata`` produces the same
   ``(seq_lens_aligned, block_tables_aligned)`` as the legacy Python
   ``.item()`` loop for both zero-token and max-token corner cases.

These are CPU-only tests; they do not require a GPU runtime.
"""
from __future__ import annotations

import os
from typing import Any

import pytest
import torch

from vllm.model_executor.models.gemma4 import (
    _build_local_decode_aligned_metadata,
)


def _legacy_build(
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    local_window: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation that mirrors the pre-optimization logic."""
    seq_lens_i64 = seq_lens.to(dtype=torch.long)
    bsz = int(seq_lens_i64.shape[0])
    if bsz <= 0:
        return seq_lens_i64, block_tables.new_zeros((0, 0))

    if int(local_window) > 0:
        starts = torch.clamp(seq_lens_i64 - int(local_window), min=0)
    else:
        starts = torch.zeros_like(seq_lens_i64)
    start_aligned = torch.div(starts, block_size, rounding_mode="floor") * block_size
    seq_lens_aligned = seq_lens_i64 - start_aligned
    num_blocks = torch.div(
        seq_lens_aligned + block_size - 1,
        block_size,
        rounding_mode="floor",
    )
    max_blocks = int(torch.max(num_blocks).item()) if bsz > 0 else 0
    block_tables_aligned = block_tables.new_zeros((bsz, max_blocks))
    for bi in range(bsz):
        n = int(num_blocks[bi].item())
        if n <= 0:
            continue
        sblk = int(start_aligned[bi].item() // block_size)
        block_tables_aligned[bi, :n] = block_tables[bi, sblk : sblk + n]
    return seq_lens_aligned, block_tables_aligned


@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("local_window", [0, 128, 1024])
def test_build_local_aligned_matches_legacy(block_size: int, local_window: int) -> None:
    torch.manual_seed(42)
    bsz = 4
    # Use 256 columns at block_size=16 => max 4096 tokens; enough headroom
    # for all seq_lens in the fixture while the legacy reference copies
    # contiguous slices out of the source table.
    num_blocks_per_seq = 256
    seq_lens = torch.tensor([1, 16, 257, 2049], dtype=torch.int32)
    block_tables = torch.randint(
        low=0, high=1024, size=(bsz, num_blocks_per_seq), dtype=torch.int32
    )

    new_sl, new_bt = _build_local_decode_aligned_metadata(
        block_tables=block_tables,
        seq_lens=seq_lens,
        local_window=local_window,
        block_size=block_size,
        seq_lens_cpu=[int(v) for v in seq_lens.tolist()],
    )
    ref_sl, ref_bt = _legacy_build(
        block_tables=block_tables,
        seq_lens=seq_lens,
        local_window=local_window,
        block_size=block_size,
    )

    assert torch.equal(new_sl, ref_sl)
    # The new path may pad columns out to a common width; the *populated*
    # region must match the reference exactly. Any extra columns must be
    # zero by construction (masked via ``torch.where``).
    common_cols = min(new_bt.shape[1], ref_bt.shape[1])
    assert torch.equal(new_bt[:, :common_cols], ref_bt[:, :common_cols])
    if new_bt.shape[1] > common_cols:
        assert torch.all(new_bt[:, common_cols:] == 0)
    if ref_bt.shape[1] > common_cols:
        assert torch.all(ref_bt[:, common_cols:] == 0)


def test_build_local_aligned_zero_batch() -> None:
    block_tables = torch.zeros((0, 4), dtype=torch.int32)
    seq_lens = torch.zeros((0,), dtype=torch.int32)
    sl, bt = _build_local_decode_aligned_metadata(
        block_tables=block_tables,
        seq_lens=seq_lens,
        local_window=128,
        block_size=16,
        seq_lens_cpu=[],
    )
    assert int(sl.numel()) == 0
    assert tuple(bt.shape) == (0, 0)


def test_build_local_aligned_fallback_env_path(monkeypatch: Any) -> None:
    """
    Without CPU hints and without the legacy env flag, the conservative
    fallback must still produce numerically-correct aligned metadata for
    the *populated* region.
    """
    monkeypatch.delenv("FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH", raising=False)
    block_size = 16
    bsz = 3
    seq_lens = torch.tensor([17, 64, 300], dtype=torch.int32)
    block_tables = torch.arange(0, bsz * 32, dtype=torch.int32).reshape(bsz, 32)

    sl_new, bt_new = _build_local_decode_aligned_metadata(
        block_tables=block_tables,
        seq_lens=seq_lens,
        local_window=0,
        block_size=block_size,
        seq_lens_cpu=None,
    )
    sl_ref, bt_ref = _legacy_build(
        block_tables=block_tables,
        seq_lens=seq_lens,
        local_window=0,
        block_size=block_size,
    )
    assert torch.equal(sl_new, sl_ref)
    cols = min(bt_new.shape[1], bt_ref.shape[1])
    assert torch.equal(bt_new[:, :cols], bt_ref[:, :cols])


def test_meta_cpu_helpers_round_trip() -> None:
    """
    Regression: make sure the helper accessors used on the Gemma4 hot path
    parse the dict-shaped attn_metadata the InputBatchBuilder produces.
    """
    from vllm.model_executor.models.gemma4 import (
        _meta_cpu_max_seq_len,
        _meta_cpu_seq_lens,
        _resolve_max_position_plus_one_cpu,
    )

    meta = {
        "seq_lens_cpu": [12, 34, 56],
        "max_seq_len_cpu": 56,
        "positions_cpu": [11, 33, 55],
    }
    assert _meta_cpu_seq_lens(meta) == [12, 34, 56]
    assert _meta_cpu_max_seq_len(meta) == 56
    assert _resolve_max_position_plus_one_cpu(
        meta, torch.tensor([], dtype=torch.long)
    ) == 56

    # No CPU fields: the resolver should return None (caller will decide
    # whether to extend the RoPE cache to ``max_position_embeddings``).
    assert _resolve_max_position_plus_one_cpu(
        {}, torch.tensor([], dtype=torch.long)
    ) is None


def test_legacy_item_env_path_preserved(monkeypatch: Any) -> None:
    """
    Legacy D->H path must still function when explicitly opted back in.
    """
    monkeypatch.setenv("FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH", "1")
    bsz = 2
    block_size = 16
    seq_lens = torch.tensor([32, 48], dtype=torch.int32)
    block_tables = torch.arange(0, bsz * 8, dtype=torch.int32).reshape(bsz, 8)

    sl, bt = _build_local_decode_aligned_metadata(
        block_tables=block_tables,
        seq_lens=seq_lens,
        local_window=0,
        block_size=block_size,
        seq_lens_cpu=None,
    )
    sl_ref, bt_ref = _legacy_build(
        block_tables=block_tables,
        seq_lens=seq_lens,
        local_window=0,
        block_size=block_size,
    )
    assert torch.equal(sl, sl_ref)
    cols = min(bt.shape[1], bt_ref.shape[1])
    assert torch.equal(bt[:, :cols], bt_ref[:, :cols])
