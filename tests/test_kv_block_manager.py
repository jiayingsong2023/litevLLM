# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.engine.kv_block_manager import KVBlockManager


def test_kv_block_manager_block_table_for_slot() -> None:
    manager = KVBlockManager(
        kv_caches=[],
        kv_scale_caches=[],
        num_blocks_per_seq=3,
        block_size=2,
    )

    table = manager.block_table_for_slot(2, device=torch.device("cpu"))

    assert torch.equal(table, torch.tensor([6, 7, 8], dtype=torch.int32))


def test_kv_block_manager_capture_and_materialize_prefix_entry() -> None:
    kv_caches = [(torch.zeros((4, 2, 1, 1)), torch.zeros((4, 2, 1, 1)))]
    kv_scale_caches = [(None, None)]
    manager = KVBlockManager(
        kv_caches=kv_caches,
        kv_scale_caches=kv_scale_caches,
        num_blocks_per_seq=2,
        block_size=2,
    )
    kv_caches[0][0][0:2].copy_(torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]]))
    kv_caches[0][1][0:2].copy_(torch.tensor([[[[5.0]], [[6.0]]], [[[7.0]], [[8.0]]]]))

    entry = manager.capture_prefix_entry(
        key=(11, 12, 13),
        slot_idx=0,
        prompt_len=3,
        last_prompt_logits=torch.tensor([0.1, 0.9]),
    )

    manager.materialize_prefix_entry(slot_idx=1, entry=entry, prefix_len=3)

    assert torch.equal(
        kv_caches[0][0][2:4],
        torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]]),
    )
    assert torch.equal(
        kv_caches[0][1][2:4],
        torch.tensor([[[[5.0]], [[6.0]]], [[[7.0]], [[8.0]]]]),
    )
