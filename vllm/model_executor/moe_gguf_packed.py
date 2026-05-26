# SPDX-License-Identifier: Apache-2.0
"""Shared MoE expert GGUF packed-row helpers."""
from __future__ import annotations

import numpy as np
import torch

_SUPPORTED_MOE_GGUF_TYPES = frozenset({2, 12, 14})  # Q4_0, Q4_K, Q6_K (same as GGUFWeight)


def moe_packed_gguf_enabled() -> bool:
    return False


def qwen35_moe_packed_gguf_enabled() -> bool:
    """Legacy alias for older Qwen3.5-specific call sites."""
    return moe_packed_gguf_enabled()


def gguf_quant_type_supported_for_moe_packed(qtype: int) -> bool:
    return int(qtype) in _SUPPORTED_MOE_GGUF_TYPES


def numpy_gguf_data_to_packed_2d(
    data_np: np.ndarray,
    logical_shape: tuple[int, ...],
    qtype: int,
) -> np.ndarray:
    """
    Reshape raw GGUF tensor bytes to 2D [total_logical_rows, packed_last_dim] uint8.
    logical_shape is PyTorch order e.g. (E, moe_inter, hidden) or (E, hidden, moe_inter).
    """
    from gguf import GGMLQuantizationType, quant_shape_to_byte_shape

    qt = GGMLQuantizationType(int(qtype))
    byte_shape = quant_shape_to_byte_shape(logical_shape, qt)
    flat = np.asarray(data_np).ravel()
    need = int(np.prod(byte_shape, dtype=np.int64))
    if flat.size != need:
        raise ValueError(
            f"GGUF MoE packed size mismatch: got {flat.size} bytes, need {need} for "
            f"logical_shape={logical_shape} qtype={qt.name} byte_shape={byte_shape}"
        )
    arr = flat.reshape(byte_shape)
    rows = int(np.prod(byte_shape[:-1], dtype=np.int64))
    packed_last = int(byte_shape[-1])
    return np.ascontiguousarray(arr.reshape(rows, packed_last))


def dequant_packed_rows_to_fp16(
    packed_2d: torch.Tensor,
    row_begin: int,
    row_end: int,
    n_cols_logical: int,
    quant_type: int,
) -> torch.Tensor:
    """
    Dequantize only packed rows [row_begin, row_end) to FP16 [row_end - row_begin, n_cols_logical].

    packed_2d must be uint8 with shape [total_rows, packed_cols] matching gguf layout for quant_type.
    """
    from gguf import dequantize, GGMLQuantizationType

    if packed_2d.dtype != torch.uint8:
        raise TypeError(f"packed_2d must be uint8, got {packed_2d.dtype}")
    if row_begin < 0 or row_end <= row_begin or row_end > packed_2d.shape[0]:
        raise ValueError(
            f"bad row range [{row_begin}, {row_end}) for packed rows {packed_2d.shape[0]}"
        )
    sub = packed_2d[row_begin:row_end].contiguous()
    w_np = np.asarray(sub.cpu().numpy(), dtype=np.uint8, order="C")
    dequant_np = dequantize(w_np, GGMLQuantizationType(int(quant_type)))
    res = torch.from_numpy(np.asarray(dequant_np, copy=True)).to(
        device=packed_2d.device, dtype=torch.float16
    )
    n_rows = row_end - row_begin
    total = n_rows * n_cols_logical
    if res.numel() >= total:
        return res.view(-1)[:total].view(n_rows, n_cols_logical)
    out = torch.zeros(total, device=res.device, dtype=torch.float16)
    out[: res.numel()] = res.view(-1)
    return out.view(n_rows, n_cols_logical)
