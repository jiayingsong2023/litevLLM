# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from vllm.utils.torch_utils import dtype_nbytes


def resolve_layer_kv_specs(model: Any, num_layers: int) -> list[tuple[int, int]] | None:
    """Best-effort per-layer KV specs in unpacked domain: (num_kv_heads, head_dim).

    Falls back to model-capability-wide uniform dimensions when model internals
    are not inspectable.
    """
    try:
        layers = list(getattr(getattr(model, "model", None), "layers", []))
        if not layers:
            return None
        specs: list[tuple[int, int]] = []
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                return None
            nkv = int(attn.num_kv_heads)
            hdim = int(attn.head_dim)
            if nkv <= 0 or hdim <= 0:
                return None
            specs.append((nkv, hdim))
        if len(specs) != int(num_layers):
            return None
        return specs
    except Exception:
        return None


def layer_kv_cache_shape_for_layer(
    layer_kv_specs: list[tuple[int, int]] | None,
    layer_idx: int,
    kv_dtype: torch.dtype,
    fallback_num_kv_heads: int,
    fallback_kv_head_dim: int,
) -> tuple[int, int]:
    """Return KV cache (num_kv_heads, head_dim) for a layer.

    When ``layer_kv_specs`` is provided, ``hdim`` is the unpacked model head
    dimension and must be halved for packed uint8 (TurboQuant INT4). When
    falling back to capability-wide dims, ``fallback_kv_head_dim`` is already
    the cache-side dimension (packed or not), so it is returned as-is.
    """
    if layer_kv_specs is None:
        return int(fallback_num_kv_heads), int(fallback_kv_head_dim)
    nkv, hdim = layer_kv_specs[layer_idx]
    if kv_dtype == torch.uint8:
        return int(nkv), int(hdim // 2)
    return int(nkv), int(hdim)


def compute_kv_theory_bytes(
    *,
    layer_kv_specs: list[tuple[int, int]] | None,
    num_layers: int,
    num_total_blocks: int,
    block_size: int,
    fallback_num_kv_heads: int,
    fallback_kv_head_dim: int,
    kv_dtype: torch.dtype,
    needs_scale_cache: bool,
) -> int:
    if layer_kv_specs is None:
        data = int(
            num_layers
            * 2
            * num_total_blocks
            * block_size
            * fallback_num_kv_heads
            * fallback_kv_head_dim
            * dtype_nbytes(kv_dtype)
        )
        if not needs_scale_cache:
            return data
        return data + int(
            num_layers
            * 2
            * num_total_blocks
            * block_size
            * fallback_num_kv_heads
            * dtype_nbytes(torch.float32)
        )
    data = 0
    scale = 0
    for i in range(num_layers):
        nkv, cache_hdim = layer_kv_cache_shape_for_layer(
            layer_kv_specs,
            i,
            kv_dtype,
            fallback_num_kv_heads,
            fallback_kv_head_dim,
        )
        data += (
            2
            * num_total_blocks
            * block_size
            * nkv
            * cache_hdim
            * dtype_nbytes(kv_dtype)
        )
        scale += 2 * num_total_blocks * block_size * nkv * dtype_nbytes(torch.float32)
    return int(data + (scale if needs_scale_cache else 0))


def compute_kv_scale_theory_bytes(
    *,
    layer_kv_specs: list[tuple[int, int]] | None,
    num_layers: int,
    num_total_blocks: int,
    block_size: int,
    fallback_num_kv_heads: int,
    fallback_kv_head_dim: int,
) -> int:
    if layer_kv_specs is None:
        return int(
            num_layers
            * 2
            * num_total_blocks
            * block_size
            * fallback_num_kv_heads
            * dtype_nbytes(torch.float32)
        )
    total = 0
    for i in range(num_layers):
        nkv, _cache_hdim = layer_kv_cache_shape_for_layer(
            layer_kv_specs,
            i,
            torch.uint8,
            fallback_num_kv_heads,
            fallback_kv_head_dim,
        )
        total += 2 * num_total_blocks * block_size * nkv * dtype_nbytes(torch.float32)
