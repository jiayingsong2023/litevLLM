# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from .config import Gemma4LayerConfig, _GEMMA4_ROPE_CACHE_POOL
from vllm.model_executor.models.lite_config import LiteConfig
from .policy_utils import (
    _gemma4_model_policy_truthy,
    _gemma4_policy_value,
    _resolve_gemma4_rope_cache_max_pos,
    _resolve_gemma4_rope_cache_pool_limit,
)


def _layer_type_for_idx(config: LiteConfig, layer_idx: int) -> str:
    layer_types = getattr(config, "layer_types", None)
    if isinstance(layer_types, list) and layer_idx < len(layer_types):
        return str(layer_types[layer_idx]).lower()
    return "global"


def _is_local_layer(layer_type: str) -> bool:
    return any(x in layer_type for x in ("local", "sliding"))


class Gemma4LayerRotaryEmbedding(nn.Module):
    def __init__(
        self,
        config: LiteConfig,
        head_size: int,
        layer_type: str,
        runtime_config: Any = None,
        layer_config: Gemma4LayerConfig | None = None,
    ):
        super().__init__()
        self.head_size = int(head_size)
        self.layer_type = layer_type
        self.max_position_embeddings_limit = int(config.max_position_embeddings)
        self.max_position_embeddings = _resolve_gemma4_rope_cache_max_pos(
            config, runtime_config
        )
        self._rope_cache_pool_limit = _resolve_gemma4_rope_cache_pool_limit(
            runtime_config
        )
        self._rope_cache_pool = (
            layer_config.rope_cache_pool
            if layer_config is not None
            else _GEMMA4_ROPE_CACHE_POOL
        )
        self.apply_rotary_emb = ApplyRotaryEmb(is_neox_style=True)

        rope_params = {}
        cfg_rope = getattr(config, "rope_parameters", None)
        if isinstance(cfg_rope, dict):
            layer_rope = cfg_rope.get(layer_type)
            if isinstance(layer_rope, dict):
                rope_params = layer_rope
        self.base = float(
            rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        )
        self.rope_type = str(rope_params.get("rope_type", "default"))
        self.partial_rotary_factor = float(
            rope_params.get("partial_rotary_factor", 1.0)
        )
        self._inv_freq_cpu = self._build_inv_freq().cpu()
        self._last_cache_key: (
            tuple[int, int, float, str, float, str, int, str] | None
        ) = None
        self._last_cache_value: tuple[torch.Tensor, torch.Tensor] | None = None

    def _build_inv_freq(self) -> torch.Tensor:
        if self.rope_type == "proportional":
            # Match HF proportional RoPE: inverse frequencies are computed over
            # the rotated sub-dimension (head_dim * partial_rotary_factor) but
            # the denominator is the full head_dim, then padded with zeros so
            # the non-rotated tail sees cos=1/sin=0.
            rot_dim = int(self.partial_rotary_factor * self.head_size)
            rot_dim = max(0, min(self.head_size, rot_dim))
            inv_rot = 1.0 / (
                self.base
                ** (
                    torch.arange(0, rot_dim, 2, dtype=torch.float32)
                    / float(self.head_size)
                )
            )
            no_rot = self.head_size // 2 - inv_rot.shape[0]
            if no_rot > 0:
                inv_freq = torch.cat(
                    (inv_rot, torch.zeros(no_rot, dtype=torch.float32)), dim=0
                )
            else:
                inv_freq = inv_rot
            return inv_freq
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.head_size, 2, dtype=torch.float32)
                / float(self.head_size)
            )
        )

    def _ensure_cache_len(self, required_len: int) -> None:
        if required_len <= self.max_position_embeddings:
            return
        new_len = min(int(required_len), int(self.max_position_embeddings_limit))
        if new_len <= self.max_position_embeddings:
            return
        self.max_position_embeddings = int(new_len)
        self._last_cache_key = None
        self._last_cache_value = None

    def _cache_pool_key(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[int, int, float, str, float, str, int, str]:
        return (
            int(self.max_position_embeddings),
            int(self.head_size),
            float(self.base),
            str(self.rope_type),
            float(self.partial_rotary_factor),
            str(device.type),
            int(device.index) if device.index is not None else -1,
            str(dtype),
        )

    def _get_or_build_cache(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._cache_pool_key(device=device, dtype=dtype)
        if self._last_cache_key == key and self._last_cache_value is not None:
            return self._last_cache_value
        pool = self._rope_cache_pool
        cached = pool.get(key)
        if cached is not None:
            pool.move_to_end(key)
            self._last_cache_key = key
            self._last_cache_value = cached
            return cached
        inv_freq = self._inv_freq_cpu.to(device=device, dtype=torch.float32)
        t = torch.arange(
            self.max_position_embeddings, device=device, dtype=torch.float32
        )
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        pool[key] = (cos, sin)
        self._last_cache_key = key
        self._last_cache_value = (cos, sin)
        pool_limit = int(self._rope_cache_pool_limit)
        while len(pool) > pool_limit:
            pool.popitem(last=False)
        return cos, sin

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        max_position_plus_one_cpu: int | None = None,
        inf_config: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Prefer a caller-provided CPU-side bound to avoid a GPU->CPU
        # sync (`positions.max().item()`) on every decoder layer.
        if max_position_plus_one_cpu is not None and int(max_position_plus_one_cpu) > 0:
            self._ensure_cache_len(int(max_position_plus_one_cpu))
        elif positions.numel() > 0:
            if _gemma4_model_policy_truthy(
                inf_config, "legacy_item_path", default=False
            ):
                required_len = int(positions.max().item()) + 1
                self._ensure_cache_len(required_len)
            else:
                # Ensure the cache covers the configured model limit once; then
                # growth happens lazily only when a hint explicitly exceeds it.
                self._ensure_cache_len(int(self.max_position_embeddings))
        if positions.device != query.device:
            positions = positions.to(query.device)
        cos_cached, sin_cached = self._get_or_build_cache(
            device=query.device,
            dtype=query.dtype,
        )
        cos = cos_cached[positions]
        sin = sin_cached[positions]
        return (
            self.apply_rotary_emb.forward_native(query, cos, sin),
            self.apply_rotary_emb.forward_native(key, cos, sin),
        )


def _get_rope(config: LiteConfig, head_size: int, layer_type: str):
    return _get_rope_with_runtime(config, head_size, layer_type, None)


def _get_rope_with_runtime(
    config: LiteConfig,
    head_size: int,
    layer_type: str,
    runtime_config: Any = None,
    layer_config: Gemma4LayerConfig | None = None,
):
    return Gemma4LayerRotaryEmbedding(
        config,
        head_size,
        layer_type,
        runtime_config=runtime_config,
        layer_config=layer_config,
    )
