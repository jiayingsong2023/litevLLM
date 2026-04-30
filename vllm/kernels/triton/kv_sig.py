# SPDX-License-Identifier: Apache-2.0
"""
KV Block Signature Finalize Kernel

Averages per-token K[:sig_dim] contributions in the temp buffer to produce
a single signature vector per (block, head). Launched once per block after
all block_size tokens have been written (block-fill boundary).

Memory layout:
  sig_temp:  (num_total_blocks, block_size, num_kv_heads, sig_dim)  float16
  sig_cache: (num_total_blocks, num_kv_heads, sig_dim)              float16

The kernel sums over block_size tokens and writes mean = sum / block_size.
"""
from __future__ import annotations

import torch
from vllm.triton_utils import triton, tl


@triton.jit
def _kv_sig_finalize_kernel(
    Sig_temp_ptr,
    Sig_cache_ptr,
    Block_ids_ptr,
    stride_stb,
    stride_sts,
    stride_sth,
    stride_std,
    stride_sb,
    stride_sh,
    stride_sd,
    BLOCK_SIZE: tl.constexpr,
    SIG_DIM: tl.constexpr,
):
    """One program per (block, head). Averages over token dim."""
    idx = tl.program_id(0)
    block_idx = tl.load(Block_ids_ptr + idx).to(tl.int64)
    head_idx = tl.program_id(1)

    off_d = tl.arange(0, SIG_DIM)

    acc = tl.zeros([SIG_DIM], dtype=tl.float32)
    for t in range(BLOCK_SIZE):
        ptr = (
            Sig_temp_ptr
            + block_idx * stride_stb
            + t * stride_sts
            + head_idx * stride_sth
            + off_d * stride_std
        )
        val = tl.load(ptr).to(tl.float32)
        acc += val

    mean = acc / BLOCK_SIZE

    out_ptr = (
        Sig_cache_ptr
        + block_idx * stride_sb
        + head_idx * stride_sh
        + off_d * stride_sd
    )
    tl.store(out_ptr, mean.to(Sig_cache_ptr.dtype.element_ty))


def kv_sig_finalize(
    sig_temp: torch.Tensor,
    sig_cache: torch.Tensor,
    block_ids: torch.Tensor,
) -> None:
    """Finalize signatures for specific blocks.

    Args:
        sig_temp: (num_total_blocks, block_size, num_kv_heads, sig_dim) float16
        sig_cache: (num_total_blocks, num_kv_heads, sig_dim) float16
        block_ids: 1D long tensor of block indices to finalize
    """
    if sig_temp.numel() == 0 or sig_cache.numel() == 0:
        return
    if block_ids.numel() == 0:
        return

    num_kv_heads = sig_cache.shape[1]
    sig_dim = sig_cache.shape[2]
    block_size = sig_temp.shape[1]

    grid = (block_ids.shape[0], num_kv_heads)

    _kv_sig_finalize_kernel[grid](
        sig_temp,
        sig_cache,
        block_ids,
        sig_temp.stride(0),
        sig_temp.stride(1),
        sig_temp.stride(2),
        sig_temp.stride(3),
        sig_cache.stride(0),
        sig_cache.stride(1),
        sig_cache.stride(2),
        BLOCK_SIZE=block_size,
        SIG_DIM=sig_dim,
    )
