# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.model_executor.models.gemma4 import (
    _decode_int4_rows,
    _gather_recent_kv,
)


def _pack_int4_from_halves(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    # low/high int4 in range [-8, 7], shape [..., D]
    low_u = (low.to(torch.int8).to(torch.uint8) & 0x0F).to(torch.uint8)
    high_u = (high.to(torch.int8).to(torch.uint8) & 0x0F).to(torch.uint8)
    return low_u | (high_u << 4)


def test_decode_int4_rows_matches_expected_sign_extend() -> None:
    # T=2, H=1, D=4 (packed D/2=2)
    # Row0: low=[-1, 2], high=[3, -4]
    # Row1: low=[7, -8], high=[-2, 1]
    low = torch.tensor([[[-1, 2]], [[7, -8]]], dtype=torch.int32)
    high = torch.tensor([[[3, -4]], [[-2, 1]]], dtype=torch.int32)
    packed = _pack_int4_from_halves(low, high)
    scales = torch.tensor([[1.0], [0.5]], dtype=torch.float32)

    out = _decode_int4_rows(packed, scales, head_dim=4)
    exp0 = torch.tensor([[-1.0, 2.0, 3.0, -4.0]], dtype=torch.float32)
    exp1 = torch.tensor([[3.5, -4.0, -1.0, 0.5]], dtype=torch.float32)
    expected = torch.stack([exp0, exp1], dim=0)
    assert torch.allclose(out, expected, atol=1e-6, rtol=0.0)


def test_gather_recent_kv_int4_shapes_and_values() -> None:
    # Build small cache: 2 blocks, block_size=4, num_kv_heads=1, head_dim=4 -> packed_dim=2
    block_size = 4
    num_kv_heads = 1
    head_dim = 4
    packed_dim = head_dim // 2
    k_cache = torch.zeros((2, block_size, num_kv_heads, packed_dim), dtype=torch.uint8)
    v_cache = torch.zeros((2, block_size, num_kv_heads, packed_dim), dtype=torch.uint8)
    k_scale = torch.ones((2, block_size, num_kv_heads, 1), dtype=torch.float32)
    v_scale = torch.ones((2, block_size, num_kv_heads, 1), dtype=torch.float32)

    # token 0 in block0/off0 => low=[1,2], high=[3,4]
    # token 5 in block1/off1 => low=[-1,-2], high=[-3,-4]
    k_cache[0, 0, 0] = _pack_int4_from_halves(
        torch.tensor([1, 2]), torch.tensor([3, 4])
    )
    v_cache[0, 0, 0] = _pack_int4_from_halves(
        torch.tensor([4, 3]), torch.tensor([2, 1])
    )
    k_cache[1, 1, 0] = _pack_int4_from_halves(
        torch.tensor([-1, -2]), torch.tensor([-3, -4])
    )
    v_cache[1, 1, 0] = _pack_int4_from_halves(
        torch.tensor([-4, -3]), torch.tensor([-2, -1])
    )

    # one sequence, 6 tokens, block table maps first two logical blocks to physical 0,1
    block_tables = torch.tensor([[0, 1]], dtype=torch.int64)
    seq_lens = torch.tensor([6], dtype=torch.int32)

    k, v = _gather_recent_kv(
        (k_cache, v_cache),
        block_tables,
        seq_lens,
        batch_idx=0,
        num_kv_heads=1,
        head_dim=4,
        local_window=1,  # only last token (idx=5)
        kv_cache_dtype="turbo_int4",
        kv_scale_cache=(k_scale, v_scale),
    )
    assert tuple(k.shape) == (1, 1, 1, 4)
    assert tuple(v.shape) == (1, 1, 1, 4)
    assert torch.allclose(k[0, 0, 0], torch.tensor([-1.0, -2.0, -3.0, -4.0]))
    assert torch.allclose(v[0, 0, 0], torch.tensor([-4.0, -3.0, -2.0, -1.0]))
