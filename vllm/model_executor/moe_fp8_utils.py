# SPDX-License-Identifier: Apache-2.0
"""Shared MoE FP8 block quant/dequant helpers and runtime flags."""
from __future__ import annotations

import torch

_MOE_FP8_BLOCK = 64


def moe_fp8_block_size() -> int:
    return _MOE_FP8_BLOCK


def moe_fp8_enabled() -> bool:
    return False


def moe_offload_enabled() -> bool:
    return False


def qwen35_moe_fp8_enabled() -> bool:
    """Legacy alias for older Qwen3.5-specific call sites."""
    return moe_fp8_enabled()


def qwen35_moe_offload_enabled() -> bool:
    """Legacy alias for older Qwen3.5-specific call sites."""
    return moe_offload_enabled()


def moe_expert_lru_size() -> int:
    return 32


def dims_ok_for_moe_fp8(hidden: int, inter: int) -> bool:
    """FP8 block quant uses 64x64 blocks; all logical dims must align."""
    h, i = int(hidden), int(inter)
    if i <= 0 or h <= 0:
        return False
    two_i = 2 * i
    return (
        two_i % _MOE_FP8_BLOCK == 0
        and h % _MOE_FP8_BLOCK == 0
        and i % _MOE_FP8_BLOCK == 0
    )


def fp8_block_quantize_2d(w: torch.Tensor, block_size: int = _MOE_FP8_BLOCK):
    """
    Convert a 2D FP16/BF16 tensor to float8_e4m3fn + block scales.
    Matches vllm fused_moe._convert_to_fp8_with_scales layout.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _convert_to_fp8_with_scales,
    )

    out_dim, in_dim = w.shape
    assert out_dim % block_size == 0 and in_dim % block_size == 0, (
        w.shape,
    )
    return _convert_to_fp8_with_scales(w.to(torch.float32), BS=block_size)


def fp8_scale_shape_2d(
    out_dim: int, in_dim: int, block_size: int = _MOE_FP8_BLOCK
) -> tuple[int, int]:
    return (out_dim // block_size, in_dim // block_size)


def moe_fp8_dequant_to_linear_weight(
    w_fp8: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int = _MOE_FP8_BLOCK,
) -> torch.Tensor:
    """Invert fp8_block_quantize_2d for F.linear (weights [out_dim, in_dim])."""
    out_dim, in_dim = w_fp8.shape
    w_blocks = w_fp8.view(out_dim // block_size, block_size, in_dim // block_size, block_size)
    w_blocks = w_blocks.to(torch.float32).permute(0, 2, 1, 3)
    w_deq = (w_blocks * scales.unsqueeze(-1).unsqueeze(-1)).permute(0, 2, 1, 3).reshape(
        out_dim, in_dim
    )
    return w_deq.to(out_dtype)
