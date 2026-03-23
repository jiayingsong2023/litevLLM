# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 MoE FP8 block quant/dequant and env flags (shared by loader + model)."""
from __future__ import annotations

import os
import torch

_MOE_FP8_BLOCK = 64


def moe_fp8_block_size() -> int:
    return _MOE_FP8_BLOCK


def qwen35_moe_fp8_enabled() -> bool:
    return os.environ.get("FASTINFERENCE_QWEN35_MOE_FP8", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def qwen35_moe_offload_enabled() -> bool:
    return os.environ.get("FASTINFERENCE_QWEN35_MOE_OFFLOAD", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def qwen35_moe_packed_gguf_enabled() -> bool:
    """MoE GGUF: keep packed uint8 expert weights at load (lowers host RSS peak). Incompatible with FP8 MoE."""
    from vllm.model_executor.moe_gguf_packed import qwen35_moe_packed_gguf_enabled as _impl

    return _impl()


def moe_expert_lru_size() -> int:
    return max(1, int(os.environ.get("FASTINFERENCE_MOE_LRU_SIZE", "32")))


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
