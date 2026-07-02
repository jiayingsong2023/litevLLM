# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.kernels.triton.compute_slot_mapping import compute_slot_mapping


def _cpu_reference(query_start_loc, positions, block_table, block_size, pad_id):
    slot_mapping = torch.full_like(positions, pad_id)
    for req_idx in range(len(query_start_loc) - 1):
        start = int(query_start_loc[req_idx].item())
        end = int(query_start_loc[req_idx + 1].item())
        for t in range(start, end):
            pos = int(positions[t].item())
            block_idx = pos // block_size
            block_id = int(block_table[req_idx, block_idx].item())
            slot_mapping[t] = block_id * block_size + (pos % block_size)
    return slot_mapping


def test_compute_slot_mapping_basic():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    block_size = 16
    block_table = torch.tensor(
        [[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int32, device=device
    )
    positions = torch.tensor([0, 15, 16, 31, 0, 17], dtype=torch.long, device=device)
    query_start_loc = torch.tensor([0, 4, 6], dtype=torch.int32, device=device)
    slot_mapping = torch.empty_like(positions)
    compute_slot_mapping(
        query_start_loc, positions, block_table, block_size, slot_mapping, pad_id=-1
    )
    expected = _cpu_reference(query_start_loc, positions, block_table, block_size, -1)
    assert torch.equal(slot_mapping.cpu(), expected.cpu())


def test_compute_slot_mapping_with_padding():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    block_size = 16
    block_table = torch.tensor([[1, 2, 0, 0]], dtype=torch.int32, device=device)
    positions = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
    query_start_loc = torch.tensor([0, 3, 3], dtype=torch.int32, device=device)
    slot_mapping = torch.full((4,), -999, dtype=torch.long, device=device)
    compute_slot_mapping(
        query_start_loc, positions, block_table, block_size, slot_mapping, pad_id=-1
    )
    expected = _cpu_reference(query_start_loc, positions, block_table, block_size, -1)
    assert torch.equal(slot_mapping[:3].cpu(), expected.cpu())
    assert slot_mapping[3].item() == -1
