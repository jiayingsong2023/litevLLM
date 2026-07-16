from __future__ import annotations

import pytest
import torch

from vllm.kernels.triton.paged_attention import (
    _paged_attention_debug_snapshot,
    paged_attention_v1,
)


def test_paged_attention_debug_snapshot_rejects_out_of_range_block() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA or ROCm")
    device = torch.device("cuda")
    query = torch.empty((2, 1, 4), device=device)
    key_cache = torch.empty((4, 16, 1, 4), device=device)
    block_tables = torch.tensor([[0, 1], [2, 4]], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([32, 32], device=device, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="metadata out of bounds"):
        _paged_attention_debug_snapshot(
            query=query,
            key_cache=key_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=16,
            scope="global",
            layer_type="full_attention",
        )


@pytest.mark.parametrize(
    ("scope", "head_size", "num_heads", "num_kv_heads"),
    (("local", 256, 8, 8), ("global", 512, 16, 1)),
)
@torch.inference_mode()
def test_fp8_paged_attention_batched_rows_match_independent_decode(
    scope: str, head_size: int, num_heads: int, num_kv_heads: int
) -> None:
    if not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("requires ROCm/CUDA FP8 support")
    device = torch.device("cuda")
    block_size = 16
    seq_lens = torch.tensor([128, 512, 2048, 2048], device=device, dtype=torch.int32)
    blocks_per_row = [
        int((length + block_size - 1) // block_size) for length in seq_lens
    ]
    total_blocks = sum(blocks_per_row)
    torch.manual_seed(17)
    query = torch.randn((4, num_heads, head_size), device=device, dtype=torch.bfloat16)
    key_cache = torch.randn(
        (total_blocks, block_size, num_kv_heads, head_size),
        device=device,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    value_cache = torch.randn_like(key_cache.to(torch.bfloat16)).to(torch.float8_e4m3fn)
    block_tables = torch.zeros(
        (4, max(blocks_per_row)), device=device, dtype=torch.int32
    )
    block_offset = 0
    for row, count in enumerate(blocks_per_row):
        block_tables[row, :count] = torch.arange(
            block_offset, block_offset + count, device=device, dtype=torch.int32
        )
        block_offset += count

    def run(
        q: torch.Tensor, tables: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        paged_attention_v1(
            out,
            q,
            key_cache,
            value_cache,
            num_heads,
            head_size**-0.5,
            tables,
            lengths,
            block_size,
            int(lengths.max().item()),
            None,
            "fp8",
            num_kv_heads=num_kv_heads,
            attn_scope=scope,
        )
        return out

    batched = run(query, block_tables, seq_lens)
    independent = torch.cat(
        [
            run(
                query[row : row + 1],
                block_tables[row : row + 1],
                seq_lens[row : row + 1],
            )
            for row in range(4)
        ],
        dim=0,
    )
    torch.cuda.synchronize()
    assert torch.equal(batched, independent)
