# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.lite_config import LiteConfig

from .attention import Gemma4Attention
from .config import Gemma4LayerConfig
from .mlp import Gemma4MLP
from .moe import Gemma4SparseMoeBlock, _is_gemma4_moe_layer
from .policy_utils import (
    _get_eps,
    _meta_get,
    _reshape_hidden_to_2d,
    _restore_hidden_from_2d,
)
from .profiling import _gemma4_profile_span


def _residual_add_fp32(residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    return (residual.float() + update.float()).to(residual.dtype)


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: LiteConfig,
        quant_config: Any,
        prefix: str,
        layer_idx: int,
        fp32_residual_guard_enabled: bool = False,
        fp32_residual_guard_start: int = 8,
        fp32_residual_guard_span: int = 3,
        runtime_config: Any = None,
    ):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        self.self_attn = Gemma4Attention(
            config,
            quant_config,
            prefix,
            layer_idx,
            runtime_config=runtime_config,
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=_get_eps(config)
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=_get_eps(config)
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=_get_eps(config)
        )
        self.pre_feedforward_layernorm_2: RMSNorm | None = None
        self.post_feedforward_layernorm_1: RMSNorm | None = None
        self.post_feedforward_layernorm_2: RMSNorm | None = None
        self.layer_scalar = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self._fp32_residual_guard_enabled = bool(fp32_residual_guard_enabled)
        self._fp32_residual_guard_start = int(fp32_residual_guard_start)
        self._fp32_residual_guard_span = max(1, int(fp32_residual_guard_span))
        self.use_moe = _is_gemma4_moe_layer(config, layer_idx)
        if self.use_moe:
            # Gemma4-26B-A4B checkpoints expose dual pre-FFN norms and dual
            # branch post-FFN norms at layer root.
            self.pre_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=_get_eps(config)
            )
            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, eps=_get_eps(config)
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=_get_eps(config)
            )
            self.mlp = Gemma4SparseMoeBlock(
                config,
                quant_config,
                prefix,
                runtime_config=runtime_config,
            )
        else:
            self.mlp = Gemma4MLP(config, quant_config, prefix)
        self._layer_config = Gemma4LayerConfig()

    def set_config(self, config: Gemma4LayerConfig) -> None:
        """Install per-instance tuning/profile configuration."""
        self._layer_config = config
        # Propagate to sub-layers
        self.self_attn._layer_config = config
        self.mlp._layer_config = config

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Any,
        attn_metadata: Any,
        lora_mapping: Any = None,
    ) -> torch.Tensor:
        inf_config = _meta_get(attn_metadata, "config", None)
        residual = x
        h = self.input_layernorm(x)
        with _gemma4_profile_span("layer_self_attn", self._layer_config):
            h = self.self_attn(h, positions, kv_cache, attn_metadata, lora_mapping)
        h = self.post_attention_layernorm(h)
        guard_hit = (
            self._fp32_residual_guard_enabled
            and self._fp32_residual_guard_start
            <= self.layer_idx
            < (self._fp32_residual_guard_start + self._fp32_residual_guard_span)
        )
        x = _residual_add_fp32(residual, h) if guard_hit else residual + h

        residual = x
        h_dense = self.pre_feedforward_layernorm(x)
        if self.use_moe and isinstance(self.mlp, Gemma4SparseMoeBlock):
            # Match HF Gemma4 MoE flow:
            # - dense MLP branch consumes pre_feedforward_layernorm(residual)
            # - router consumes raw residual (before pre-FF norms)
            # - sparse experts consume pre_feedforward_layernorm_2(residual)
            with _gemma4_profile_span("layer_dense_mlp", self._layer_config):
                dense_out = self.mlp.shared_mlp(
                    h_dense, lora_mapping=lora_mapping, inf_config=inf_config
                )
            if self.post_feedforward_layernorm_1 is not None:
                dense_out = self.post_feedforward_layernorm_1(dense_out)

            router_in_2d, router_shape = _reshape_hidden_to_2d(residual)
            with _gemma4_profile_span("layer_moe_router", self._layer_config):
                router_logits, routing_weights, selected_experts = self.mlp.router(
                    router_in_2d
                )
            if self.pre_feedforward_layernorm_2 is not None:
                sparse_in = self.pre_feedforward_layernorm_2(residual)
            else:
                sparse_in = residual
            sparse_in_2d, _ = _reshape_hidden_to_2d(sparse_in)
            with _gemma4_profile_span("layer_moe_sparse_experts", self._layer_config):
                sparse_out_2d = self.mlp.experts(
                    sparse_in_2d,
                    router_logits,
                    topk_weights=routing_weights,
                    topk_ids=selected_experts,
                )
            sparse_out = _restore_hidden_from_2d(sparse_out_2d, router_shape)
            if self.post_feedforward_layernorm_2 is not None:
                sparse_out = self.post_feedforward_layernorm_2(sparse_out)
            h = dense_out + sparse_out
        else:
            with _gemma4_profile_span("layer_dense_mlp", self._layer_config):
                h = self.mlp(h_dense, lora_mapping, inf_config=inf_config)
        h = self.post_feedforward_layernorm(h)
        x = _residual_add_fp32(residual, h) if guard_hit else residual + h
        return x * self.layer_scalar
