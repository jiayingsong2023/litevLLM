# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm.engine.block_allocator import BlockAllocator


def test_null_block_reserved():
    ba = BlockAllocator(num_total_blocks=8)
    ids = ba.allocate(7)
    assert 0 not in ids
    assert ba.num_free == 0
    with pytest.raises(RuntimeError):
        ba.allocate(1)


def test_allocate_and_free():
    ba = BlockAllocator(num_total_blocks=5)
    ids1 = ba.allocate(2)
    ids2 = ba.allocate(2)
    assert len(set(ids1 + ids2)) == 4
    assert 0 not in ids1 + ids2
    ba.free(ids1)
    assert ba.num_free == 2
    ids3 = ba.allocate(2)
    assert set(ids3) == set(ids1)


def test_free_rejects_double_free():
    ba = BlockAllocator(num_total_blocks=5)
    ids = ba.allocate(2)
    ba.free(ids)
    with pytest.raises(ValueError, match="not currently allocated"):
        ba.free(ids)


def test_free_rejects_unallocated_and_reserved_ids():
    ba = BlockAllocator(num_total_blocks=5)
    ba.allocate(2)
    with pytest.raises(ValueError, match="Invalid block id"):
        ba.free([0])
    with pytest.raises(ValueError, match="not currently allocated"):
        ba.free([3])
