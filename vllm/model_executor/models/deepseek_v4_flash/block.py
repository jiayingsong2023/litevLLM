# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .attention import (
    apply_rope_to_tail_reference,
    grouped_output_projection_reference,
    latent_kv_projection_reference,
    per_head_rms_norm_reference,
    q_lora_attention_projection_reference,
    shared_kv_swa_attention_reference,
)
from .config import DEEPSEEK_V4_FLASH_SHAPE, DeepSeekV4FlashShape
from .hyper_connection import (
    hyper_connection_post_reference,
    hyper_connection_pre_reference,
)
from .moe import (
    combined_shared_routed_moe_reference,
    grouped_expert_reference,
    hash_routed_expert_ids_reference,
    hash_routed_moe_reference,
)
from .ops import rms_norm_reference
from .weight_store import (
    DeepSeekV4FlashGroupedExpertTensors,
    DeepSeekV4FlashLayerSemanticBindings,
    DeepSeekV4FlashWeightStore,
)


@dataclass(frozen=True)
class DeepSeekV4FlashBlockReference:
    layer_idx: int
    hidden_size: int
    attention: Callable[[torch.Tensor, int, Any], torch.Tensor]
    moe: Callable[[torch.Tensor], torch.Tensor]
    attn_norm_weight: torch.Tensor
    ffn_norm_weight: torch.Tensor

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        token_idx: int,
        kv_cache: Any,
    ) -> torch.Tensor:
        if hidden.shape != (self.hidden_size,):
            raise ValueError(
                f"hidden shape must be ({self.hidden_size},); "
                f"got {tuple(hidden.shape)}"
            )
        if self.attn_norm_weight.shape != (self.hidden_size,):
            raise ValueError(
                f"attn_norm_weight shape must be ({self.hidden_size},); "
                f"got {tuple(self.attn_norm_weight.shape)}"
            )
        if self.ffn_norm_weight.shape != (self.hidden_size,):
            raise ValueError(
                f"ffn_norm_weight shape must be ({self.hidden_size},); "
                f"got {tuple(self.ffn_norm_weight.shape)}"
            )

        attn_input = rms_norm_reference(hidden, self.attn_norm_weight)
        attn_output = self.attention(attn_input, token_idx, kv_cache)
        if attn_output.shape != hidden.shape:
            raise ValueError(
                "attention output shape must match hidden shape; "
                f"got {tuple(attn_output.shape)} and {tuple(hidden.shape)}"
            )
        hidden = hidden.to(torch.float32) + attn_output.to(torch.float32)

        ffn_input = rms_norm_reference(hidden, self.ffn_norm_weight)
        ffn_output = self.moe(ffn_input)
        if ffn_output.shape != hidden.shape:
            raise ValueError(
                "MoE output shape must match hidden shape; "
                f"got {tuple(ffn_output.shape)} and {tuple(hidden.shape)}"
            )
        return hidden + ffn_output.to(torch.float32)


class DeepSeekV4FlashSlidingLayerReferenceRunner:
    """CPU reference runner for one sliding-only layer bring-up.

    Layers 0 and 1 are sliding-only and do not use the compressed/indexer
    attention path. The runner is for correctness bring-up, not for production
    latency.
    """

    def __init__(
        self,
        store: DeepSeekV4FlashWeightStore,
        *,
        layer_idx: int,
        shape: DeepSeekV4FlashShape = DEEPSEEK_V4_FLASH_SHAPE,
    ) -> None:
        if layer_idx < 0 or layer_idx >= 2:
            raise ValueError(
                "DeepSeekV4FlashSlidingLayerReferenceRunner supports only "
                f"sliding-only layers 0 and 1; got {layer_idx}"
            )
        self.store = store
        self.layer_idx = layer_idx
        self.shape = shape

    def forward(
        self,
        streams: torch.Tensor,
        *,
        token_id: int,
        token_idx: int,
        kv_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if streams.shape != (4, self.shape.hidden_size):
            raise ValueError(
                f"streams shape must be (4, {self.shape.hidden_size}); "
                f"got {tuple(streams.shape)}"
            )
        layer = self.store.bindings.layers[self.layer_idx]

        residual = streams.to(torch.float32)
        attn_pre = self._hyper_pre(layer, residual, attention=True)
        attn_input = self._norm(attn_pre.mixed, layer.attention_norm)
        attn_output, current_kv = self._attention(layer, attn_input, token_idx)
        attn_kv_rows = (
            current_kv.reshape(1, -1)
            if kv_rows is None
            else torch.cat([kv_rows.to(torch.float32), current_kv.reshape(1, -1)])
        )
        attn_output = self._attention_with_kv_rows(
            layer,
            attn_input,
            token_idx,
            attn_kv_rows,
        )
        streams = hyper_connection_post_reference(attn_output, residual, attn_pre)

        residual = streams.to(torch.float32)
        ffn_pre = self._hyper_pre(layer, residual, attention=False)
        ffn_input = self._norm(ffn_pre.mixed, layer.ffn_norm)
        ffn_output = self._moe(layer, ffn_input, token_id=token_id)
        return hyper_connection_post_reference(ffn_output, residual, ffn_pre)

    def _hyper_pre(
        self,
        layer: DeepSeekV4FlashLayerSemanticBindings,
        streams: torch.Tensor,
        *,
        attention: bool,
    ):
        hc = layer.attention_hyper_connection if attention else layer.ffn_hyper_connection
        if hc is None:
            raise RuntimeError("sliding layer requires hyper-connection tensors")
        return hyper_connection_pre_reference(
            streams,
            self.store.decode_matrix(hc.fn),
            self.store.tensor_to_torch(hc.base, dtype=torch.float32),
            self.store.tensor_to_torch(hc.scale, dtype=torch.float32),
        )

    def _norm(self, hidden: torch.Tensor, tensor: Any) -> torch.Tensor:
        if tensor is None:
            raise RuntimeError("missing layer norm tensor")
        return rms_norm_reference(
            hidden,
            self.store.tensor_to_torch(tensor, dtype=torch.float32),
        )

    def _attention(
        self,
        layer: DeepSeekV4FlashLayerSemanticBindings,
        hidden: torch.Tensor,
        token_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_kv = self._kv_latent(layer, hidden, token_idx)
        return self._attention_with_kv_rows(
            layer,
            hidden,
            token_idx,
            current_kv.reshape(1, -1),
        ), current_kv

    def _kv_latent(
        self,
        layer: DeepSeekV4FlashLayerSemanticBindings,
        hidden: torch.Tensor,
        token_idx: int,
    ) -> torch.Tensor:
        if layer.attention_key_value is None or layer.attention_key_value_a_norm is None:
            raise RuntimeError("sliding layer requires KV latent tensors")
        kv = latent_kv_projection_reference(
            hidden,
            self.store.decode_matrix(layer.attention_key_value).transpose(0, 1),
            self.store.tensor_to_torch(
                layer.attention_key_value_a_norm,
                dtype=torch.float32,
            ),
        )
        return apply_rope_to_tail_reference(
            kv,
            token_idx=token_idx,
            rotary_dim=self.shape.rotary_dim,
        )

    def _attention_with_kv_rows(
        self,
        layer: DeepSeekV4FlashLayerSemanticBindings,
        hidden: torch.Tensor,
        token_idx: int,
        kv_rows: torch.Tensor,
    ) -> torch.Tensor:
        required = (
            layer.attention_query_a,
            layer.attention_query_a_norm,
            layer.attention_query_b,
            layer.attention_sinks,
            layer.attention_output_a,
            layer.attention_output_b,
        )
        if any(tensor is None for tensor in required):
            raise RuntimeError("sliding layer requires complete attention tensors")
        assert layer.attention_query_a is not None
        assert layer.attention_query_a_norm is not None
        assert layer.attention_query_b is not None
        assert layer.attention_sinks is not None
        assert layer.attention_output_a is not None
        assert layer.attention_output_b is not None
        q = q_lora_attention_projection_reference(
            hidden,
            self.store.decode_matrix(layer.attention_query_a).transpose(0, 1),
            self.store.tensor_to_torch(
                layer.attention_query_a_norm,
                dtype=torch.float32,
            ),
            self.store.decode_matrix(layer.attention_query_b).transpose(0, 1),
        ).reshape(self.shape.num_attention_heads, self.shape.head_dim)
        q = per_head_rms_norm_reference(q)
        q = apply_rope_to_tail_reference(
            q,
            token_idx=token_idx,
            rotary_dim=self.shape.rotary_dim,
        )
        context = shared_kv_swa_attention_reference(
            q,
            kv_rows,
            self.store.tensor_to_torch(layer.attention_sinks, dtype=torch.float32),
        )
        context = apply_rope_to_tail_reference(
            context,
            token_idx=token_idx,
            rotary_dim=self.shape.rotary_dim,
            inverse=True,
        )
        return grouped_output_projection_reference(
            context,
            self.store.decode_matrix(layer.attention_output_a).transpose(0, 1),
            self.store.decode_matrix(layer.attention_output_b).transpose(0, 1),
            output_groups=self.shape.output_groups,
        )

    def _moe(
        self,
        layer: DeepSeekV4FlashLayerSemanticBindings,
        hidden: torch.Tensor,
        *,
        token_id: int,
    ) -> torch.Tensor:
        if layer.shared_experts is None or layer.grouped_experts is None:
            raise RuntimeError("sliding layer requires shared and routed experts")
        shared = self._run_expert_group(layer.shared_experts, hidden)
        if layer.expert_token_to_expert_ids is None:
            raise RuntimeError("sliding layer requires hash routing tensor")
        expert_ids = hash_routed_expert_ids_reference(
            self.store.tensor_to_torch(
                layer.expert_token_to_expert_ids,
                dtype=torch.int32,
            ),
            token_id=token_id,
        )
        routed = hash_routed_moe_reference(
            hidden,
            expert_ids,
            lambda expert_id, expert_hidden: self._run_expert_group(
                layer.grouped_experts,
                expert_hidden,
                expert_id=expert_id,
            ),
            routed_scaling_factor=1.5,
        )
        return combined_shared_routed_moe_reference(shared, routed)

    def _run_expert_group(
        self,
        tensors: DeepSeekV4FlashGroupedExpertTensors,
        hidden: torch.Tensor,
        *,
        expert_id: int | None = None,
    ) -> torch.Tensor:
        if expert_id is None:
            gate = self.store.decode_matrix(tensors.gate)
            down = self.store.decode_matrix(tensors.down)
            up = self.store.decode_matrix(tensors.up)
        else:
            gate = self.store.decode_grouped_expert_matrix(tensors.gate, expert_id)
            down = self.store.decode_grouped_expert_matrix(tensors.down, expert_id)
            up = self.store.decode_grouped_expert_matrix(tensors.up, expert_id)
        return grouped_expert_reference(hidden, gate, up, down)


class DeepSeekV4FlashLayer0ReferenceRunner(DeepSeekV4FlashSlidingLayerReferenceRunner):
    def __init__(
        self,
        store: DeepSeekV4FlashWeightStore,
        *,
        shape: DeepSeekV4FlashShape = DEEPSEEK_V4_FLASH_SHAPE,
    ) -> None:
        super().__init__(store, layer_idx=0, shape=shape)
