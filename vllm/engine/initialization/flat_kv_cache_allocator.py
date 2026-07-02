# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.engine.initialization.kv_cache_allocator import (
    layer_kv_cache_shape_for_layer,
)


class FlatKVCacheAllocator:
    """Allocates a single flat GPU buffer for all per-layer K/V caches.

    Block ID 0 is reserved as the zeroed null block and is never handed out.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_total_blocks: int,
        block_size: int,
        device: torch.device,
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_total_blocks = int(num_total_blocks)
        self.block_size = int(block_size)
        self.device = device

    def _layer_shape(
        self,
        layer_kv_specs: list[tuple[int, int]] | None,
        layer_idx: int,
        kv_dtype: torch.dtype,
        fallback_num_kv_heads: int,
        fallback_kv_head_dim: int,
    ) -> tuple[int, int]:
        """Helper to centralize per-layer shape lookups."""
        return layer_kv_cache_shape_for_layer(
            layer_kv_specs,
            layer_idx,
            kv_dtype,
            fallback_num_kv_heads,
            fallback_kv_head_dim,
        )

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
        int,
    ]:
        block_elems_per_layer: list[int] = []
        for layer_idx in range(self.num_layers):
            nkv, hdim = self._layer_shape(
                layer_kv_specs,
                layer_idx,
                kv_dtype,
                fallback_num_kv_heads,
                fallback_kv_head_dim,
            )
            block_elems_per_layer.append(self.block_size * nkv * hdim)

        total_block_elems = sum(block_elems_per_layer)
        half_size = self.num_total_blocks * total_block_elems
        kv_buffer = torch.zeros(
            half_size * 2,
            dtype=kv_dtype,
            device=self.device,
        )

        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        offset = 0
        for layer_idx, block_elems in enumerate(block_elems_per_layer):
            layer_elems = self.num_total_blocks * block_elems
            nkv, hdim = self._layer_shape(
                layer_kv_specs,
                layer_idx,
                kv_dtype,
                fallback_num_kv_heads,
                fallback_kv_head_dim,
            )
            k_view = kv_buffer[offset : offset + layer_elems].view(
                self.num_total_blocks,
                self.block_size,
                nkv,
                hdim,
            )
            offset += layer_elems
            v_view = kv_buffer[offset : offset + layer_elems].view_as(k_view)
            offset += layer_elems
            kv_caches.append((k_view, v_view))

        kv_scale_caches: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
        if needs_scale_cache:
            layer_scale_elems: list[int] = []
            for layer_idx in range(self.num_layers):
                nkv, _ = self._layer_shape(
                    layer_kv_specs,
                    layer_idx,
                    torch.uint8,
                    fallback_num_kv_heads,
                    fallback_kv_head_dim,
                )
                layer_scale_elems.append(self.num_total_blocks * self.block_size * nkv)
            total_scale_elems = sum(layer_scale_elems) * 2
            scale_buffer = torch.zeros(
                total_scale_elems,
                dtype=torch.float32,
                device=self.device,
            )
            scale_offset = 0
            for layer_idx, scale_elems in enumerate(layer_scale_elems):
                nkv, _ = self._layer_shape(
                    layer_kv_specs,
                    layer_idx,
                    torch.uint8,
                    fallback_num_kv_heads,
                    fallback_kv_head_dim,
                )
                ks_view = scale_buffer[scale_offset : scale_offset + scale_elems].view(
                    self.num_total_blocks, self.block_size, nkv, 1
                )
                scale_offset += scale_elems
                vs_view = scale_buffer[
                    scale_offset : scale_offset + scale_elems
                ].view_as(ks_view)
                scale_offset += scale_elems
                kv_scale_caches.append((ks_view, vs_view))
        else:
            kv_scale_caches = [(None, None)] * self.num_layers

        return kv_caches, kv_scale_caches, self.num_total_blocks
