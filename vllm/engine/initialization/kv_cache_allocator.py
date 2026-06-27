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
    if layer_kv_specs is None:
        nkv, hdim = fallback_num_kv_heads, fallback_kv_head_dim
    else:
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
    return int(total)


class KVCacheAllocator:
    """Allocates paged KV caches and optional per-token scale caches."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_total_blocks: int,
        block_size: int,
        device: torch.device,
    ) -> None:
        self.num_layers = num_layers
        self.num_total_blocks = num_total_blocks
        self.block_size = block_size
        self.device = device

    def allocate(
        self,
        layer_kv_specs: list[tuple[int, int]] | None,
        kv_dtype: torch.dtype,
        kv_head_dim: int,
        fallback_num_kv_heads: int,
        fallback_kv_head_dim: int,
        needs_scale_cache: bool,
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor]],
        list[tuple[torch.Tensor | None, torch.Tensor | None]],
    ]:
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        kv_scale_caches: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []

        for layer_idx in range(self.num_layers):
            layer_num_kv_heads, layer_kv_head_dim = layer_kv_cache_shape_for_layer(
                layer_kv_specs,
                layer_idx,
                kv_dtype,
                fallback_num_kv_heads,
                fallback_kv_head_dim,
            )
            # Shape: (num_total_blocks, block_size, heads, head_size)
            k = torch.zeros(
                (
                    self.num_total_blocks,
                    self.block_size,
                    layer_num_kv_heads,
                    layer_kv_head_dim,
                ),
                device=self.device,
                dtype=kv_dtype,
            )
            v = torch.zeros_like(k)
            kv_caches.append((k, v))

        if needs_scale_cache:
            for layer_idx in range(self.num_layers):
                layer_num_kv_heads, _layer_kv_head_dim = layer_kv_cache_shape_for_layer(
                    layer_kv_specs,
                    layer_idx,
                    torch.uint8,
                    fallback_num_kv_heads,
                    fallback_kv_head_dim,
                )
                # Per-token, per-head scale:
                # (num_total_blocks, block_size, num_kv_heads, 1)
                ks = torch.zeros(
                    (
                        self.num_total_blocks,
                        self.block_size,
                        layer_num_kv_heads,
                        1,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
                vs = torch.zeros_like(ks)
                kv_scale_caches.append((ks, vs))
        else:
            kv_scale_caches = [(None, None)] * self.num_layers

        return kv_caches, kv_scale_caches
