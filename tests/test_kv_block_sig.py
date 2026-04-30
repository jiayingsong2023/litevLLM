# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from vllm.kernels.triton.kv_sig import kv_sig_finalize


def _sig_ref(sig_temp: torch.Tensor) -> torch.Tensor:
    """Reference: mean of per-token K[:sig_dim] over block_size tokens."""
    return sig_temp.float().mean(dim=1)


@pytest.mark.parametrize("num_blocks", [1, 4, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("sig_dim", [16, 32, 64])
@pytest.mark.parametrize("block_size", [16])
def test_kv_sig_finalize_numerics(num_blocks, num_kv_heads, sig_dim, block_size):
    torch.manual_seed(42)
    sig_temp = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        sig_dim,
        dtype=torch.float16,
        device="cuda",
    )
    sig_cache = torch.zeros(
        num_blocks,
        num_kv_heads,
        sig_dim,
        dtype=torch.float16,
        device="cuda",
    )
    block_ids = torch.arange(num_blocks, device="cuda", dtype=torch.long)

    kv_sig_finalize(sig_temp, sig_cache, block_ids)
    torch.cuda.synchronize()

    expected = _sig_ref(sig_temp)
    actual = sig_cache.float()

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_kv_sig_finalize_empty_tensors():
    """Empty tensors should be a no-op."""
    empty = torch.empty(0, device="cuda")
    kv_sig_finalize(empty, empty, empty)


def test_kv_sig_finalize_subset_blocks():
    """Only finalize a subset of blocks."""
    num_blocks, num_kv_heads, sig_dim, block_size = 8, 2, 32, 16
    torch.manual_seed(99)
    sig_temp = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        sig_dim,
        dtype=torch.float16,
        device="cuda",
    )
    sig_cache = torch.zeros(
        num_blocks,
        num_kv_heads,
        sig_dim,
        dtype=torch.float16,
        device="cuda",
    )
    block_ids = torch.tensor([2, 5], device="cuda", dtype=torch.long)

    kv_sig_finalize(sig_temp, sig_cache, block_ids)
    torch.cuda.synchronize()

    expected = _sig_ref(sig_temp)
    for b in range(num_blocks):
        if b in (2, 5):
            torch.testing.assert_close(
                sig_cache[b].float(),
                expected[b].float(),
                rtol=1e-3,
                atol=1e-3,
            )
        else:
            assert (sig_cache[b] == 0).all(), f"Block {b} should be zero"
