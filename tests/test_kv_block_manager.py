# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

from vllm.engine.block_allocator import BlockAllocator
from vllm.engine.kv_block_manager import KVBlockManager


def _make_manager(
    num_layers: int = 2,
    num_total_blocks: int = 8,
    block_size: int = 16,
    num_blocks_per_seq: int = 4,
) -> KVBlockManager:
    kv_caches = []
    for _ in range(num_layers):
        k = torch.zeros(num_total_blocks, block_size, 2, 32)
        v = torch.zeros_like(k)
        kv_caches.append((k, v))
    return KVBlockManager(
        kv_caches=kv_caches,
        kv_scale_caches=[(None, None)] * num_layers,
        num_blocks_per_seq=num_blocks_per_seq,
        block_size=block_size,
        max_active_requests=4,
        block_allocator=BlockAllocator(num_total_blocks),
    )


def test_block_table_for_slot_returns_dynamic_row() -> None:
    mgr = _make_manager(num_total_blocks=16, num_blocks_per_seq=4)
    mgr.ensure_blocks("r0", 32)
    mgr.update_block_table_row(2, "r0")
    row = mgr.block_table_for_slot(2)
    assert row.shape == (4,)
    assert row[0].item() == mgr._request_blocks["r0"][0]
    assert row[1].item() == mgr._request_blocks["r0"][1]
    assert row[2].item() == 0
    assert row[3].item() == 0


def test_ensure_blocks_grows_and_pads() -> None:
    mgr = _make_manager()
    new_blocks = mgr.ensure_blocks("r0", 20)
    assert new_blocks == 2
    mgr.update_block_table_row(0, "r0")
    row = mgr.block_table_for_slot(0)
    assert row.shape == (4,)
    assert row[0].item() != 0
    assert row[1].item() != 0
    assert row[2].item() == 0
    assert row[3].item() == 0


def test_ensure_blocks_does_not_shrink() -> None:
    mgr = _make_manager()
    mgr.ensure_blocks("r0", 48)
    assert len(mgr._request_blocks["r0"]) == 3
    mgr.ensure_blocks("r0", 16)
    assert len(mgr._request_blocks["r0"]) == 3


def test_ensure_blocks_rejects_overflowing_block_table_row() -> None:
    mgr = _make_manager(num_blocks_per_seq=2)
    with pytest.raises(ValueError, match="exceeding per-request capacity 2"):
        mgr.ensure_blocks("r0", 100)


def test_ensure_blocks_for_requests() -> None:
    mgr = _make_manager(num_total_blocks=16)
    mgr.ensure_blocks_for_requests(["r0", "r1"], [16, 48])
    assert len(mgr._request_blocks["r0"]) == 1
    assert len(mgr._request_blocks["r1"]) == 3


def test_free_request_blocks_clears_row() -> None:
    mgr = _make_manager()
    mgr.ensure_blocks("r0", 32)
    mgr.update_block_table_row(1, "r0")
    mgr.free_request_blocks("r0")
    assert "r0" not in mgr._request_blocks
    assert mgr.block_table_for_slot(1).tolist() == [0, 0, 0, 0]
    assert mgr._allocator.num_free == 7


def test_capture_and_materialize_prefix_entry() -> None:
    kv_caches = [(torch.zeros((8, 2, 1, 1)), torch.zeros((8, 2, 1, 1)))]
    kv_scale_caches = [(None, None)]
    manager = KVBlockManager(
        kv_caches=kv_caches,
        kv_scale_caches=kv_scale_caches,
        num_blocks_per_seq=2,
        block_size=2,
        max_active_requests=2,
        block_allocator=BlockAllocator(8),
    )
    manager.ensure_blocks("r0", 4)
    r0_blocks = manager._request_blocks["r0"]
    kv_caches[0][0][r0_blocks[0] : r0_blocks[0] + 1].copy_(
        torch.tensor([[[[1.0]], [[2.0]]]])
    )
    kv_caches[0][1][r0_blocks[0] : r0_blocks[0] + 1].copy_(
        torch.tensor([[[[5.0]], [[6.0]]]])
    )
    kv_caches[0][0][r0_blocks[1] : r0_blocks[1] + 1].copy_(
        torch.tensor([[[[3.0]], [[4.0]]]])
    )
    kv_caches[0][1][r0_blocks[1] : r0_blocks[1] + 1].copy_(
        torch.tensor([[[[7.0]], [[8.0]]]])
    )

    entry = manager.capture_prefix_entry(
        key=(11, 12, 13),
        request_id="r0",
        prompt_len=3,
        last_prompt_logits=torch.tensor([0.1, 0.9]),
    )
    assert entry.used_blocks == 2

    manager.free_request_blocks("r0")
    manager.ensure_blocks("r1", 4)
    manager.materialize_prefix_entry(request_id="r1", entry=entry, prefix_len=3)
    r1_blocks = manager._request_blocks["r1"]
    assert torch.equal(
        kv_caches[0][0][r1_blocks[0] : r1_blocks[0] + 1],
        torch.tensor([[[[1.0]], [[2.0]]]]),
    )
    assert torch.equal(
        kv_caches[0][0][r1_blocks[1] : r1_blocks[1] + 1],
        torch.tensor([[[[3.0]], [[4.0]]]]),
    )
    assert torch.equal(
        kv_caches[0][1][r1_blocks[0] : r1_blocks[0] + 1],
        torch.tensor([[[[5.0]], [[6.0]]]]),
    )
    assert torch.equal(
        kv_caches[0][1][r1_blocks[1] : r1_blocks[1] + 1],
        torch.tensor([[[[7.0]], [[8.0]]]]),
    )


def test_prefix_capture_and_materialize() -> None:
    mgr = _make_manager(num_total_blocks=16)
    mgr.ensure_blocks("r0", 32)
    # Write a marker into r0's first block of layer 0 K cache
    k0, _ = mgr.kv_caches[0]
    block_id = mgr._request_blocks["r0"][0]
    k0[block_id, 0, 0, 0] = 3.14

    entry = mgr.capture_prefix_entry(
        key=(1, 2, 3),
        request_id="r0",
        prompt_len=32,
        last_prompt_logits=torch.zeros(10),
    )
    mgr.free_request_blocks("r0")
    mgr.ensure_blocks("r1", 32)
    mgr.materialize_prefix_entry(request_id="r1", entry=entry, prefix_len=32)

    new_block_id = mgr._request_blocks["r1"][0]
    assert torch.isclose(k0[new_block_id, 0, 0, 0], torch.tensor(3.14))


def test_freed_blocks_are_zeroed_on_reuse() -> None:
    mgr = _make_manager(num_total_blocks=8)
    mgr.ensure_blocks("r0", 16)
    k0, _ = mgr.kv_caches[0]
    block_id = mgr._request_blocks["r0"][0]
    k0[block_id, 0, 0, 0] = 2.71

    mgr.free_request_blocks("r0")
    mgr.ensure_blocks("r1", 16)
    reused_block_id = mgr._request_blocks["r1"][0]
    # The allocator will likely hand out the same block; ensure it was zeroed.
    assert k0[reused_block_id, 0, 0, 0].item() != 2.71
    assert k0[reused_block_id, 0, 0, 0].item() == 0.0


def test_slot_idx_bounds() -> None:
    mgr = _make_manager()
    with pytest.raises(IndexError):
        mgr.block_table_for_slot(-1)
    with pytest.raises(IndexError):
        mgr.block_table_for_slot(4)
    with pytest.raises(IndexError):
        mgr.update_block_table_row(4, "r0")
