# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.models.lite_config import LiteConfig
from vllm.model_executor.models.multimodal_utils import replace_image_placeholders

from .layer import Gemma4DecoderLayer
from .policy_utils import _gemma4_fp32_residual_guard_policy, _get_eps
from .vision import Gemma4VisionProjector, Gemma4VisionTower


def _replace_image_placeholders(
    *,
    input_ids: torch.Tensor,
    text_embeddings: torch.Tensor,
    multimodal_embeddings: torch.Tensor,
    image_token_id: int,
    image_token_count: int,
) -> torch.Tensor:
    return replace_image_placeholders(
        input_ids=input_ids,
        text_embeddings=text_embeddings,
        multimodal_embeddings=multimodal_embeddings,
        image_token_id=image_token_id,
        image_token_count=image_token_count,
    )


class Gemma4TextModel(nn.Module):
    def __init__(
        self,
        hf_config: Any,
        quant_config: Any,
        prefix: str = "model",
        runtime_config: Any = None,
    ):
        super().__init__()
        self.config = LiteConfig(hf_config)
        (
            fp32_residual_guard_enabled,
            fp32_residual_guard_start,
            fp32_residual_guard_span,
        ) = _gemma4_fp32_residual_guard_policy(runtime_config)
        padding_idx = int(getattr(hf_config, "pad_token_id", 0) or 0)
        self.embed_scale = float(self.config.hidden_size) ** 0.5
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            padding_idx=padding_idx,
        )
        # Per-Layer Embeddings (PLE) for Gemma4 E2B/E4B.
        self.hidden_size_per_layer_input = int(
            getattr(self.config, "hidden_size_per_layer_input", 0) or 0
        )
        if self.config.ple_enabled():
            total_ple_dim = (
                self.config.num_hidden_layers * self.hidden_size_per_layer_input
            )
            self.embed_tokens_per_layer = nn.Embedding(
                self.config.vocab_size_per_layer_input,
                total_ple_dim,
            )
            self.per_layer_model_projection = LiteLinear(
                self.config.hidden_size,
                total_ple_dim,
                bias=False,
                quant_config=quant_config,
                prefix="model.per_layer_model_projection",
            )
            self.per_layer_projection_norm = RMSNorm(
                self.hidden_size_per_layer_input,
                eps=_get_eps(self.config),
            )
            self.register_buffer(
                "embed_scale_per_layer",
                torch.tensor(self.hidden_size_per_layer_input**0.5),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_input_scale",
                torch.rsqrt(torch.tensor(2.0)),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_projection_scale",
                torch.tensor(self.config.hidden_size**-0.5),
                persistent=False,
            )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None
            self.embed_scale_per_layer = None
            self.per_layer_input_scale = None
            self.per_layer_projection_scale = None
        layers: list[Gemma4DecoderLayer] = []
        for i in range(self.config.num_hidden_layers):
            donor_attn = None
            if self.config.is_kv_shared_layer(i):
                layer_types = self.config.layer_types
                own_type = layer_types[i] if layer_types else "global"
                # Find offset of this layer within its attention group.
                group_start = 0
                if layer_types:
                    for j in range(i - 1, -1, -1):
                        if layer_types[j] == "full_attention":
                            group_start = j + 1
                            break
                offset = i - group_start
                for j in range(i - 1, -1, -1):
                    if (
                        not self.config.is_kv_shared_layer(j)
                        and layer_types
                        and layer_types[j] == own_type
                    ):
                        # Match the same position within the donor group.
                        donor_group_start = 0
                        for k in range(j - 1, -1, -1):
                            if layer_types[k] == "full_attention":
                                donor_group_start = k + 1
                                break
                        if j - donor_group_start == offset:
                            donor_attn = layers[j].self_attn
                            break
            layers.append(
                Gemma4DecoderLayer(
                    self.config,
                    quant_config,
                    prefix=f"layers.{i}",
                    layer_idx=i,
                    fp32_residual_guard_enabled=fp32_residual_guard_enabled,
                    fp32_residual_guard_start=fp32_residual_guard_start,
                    fp32_residual_guard_span=fp32_residual_guard_span,
                    runtime_config=runtime_config,
                    kv_shared_with=donor_attn,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(self.config.hidden_size, eps=_get_eps(self.config))
        vision_config = getattr(hf_config, "vision_config", None)
        if vision_config is not None:
            vision_lite_config = LiteConfig(vision_config)
            self.vision_tower = Gemma4VisionTower(vision_config)
            self.embed_vision = Gemma4VisionProjector(
                int(vision_lite_config.hidden_size),
                int(self.config.hidden_size),
            )

    def _compute_per_layer_inputs(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.config.ple_enabled():
            return None
        if input_ids.dtype != torch.long:
            return None
        bsz, seqlen = input_ids.shape
        num_layers = self.config.num_hidden_layers
        ple_dim = self.hidden_size_per_layer_input

        # Token-identity component.
        token_part = self.embed_tokens_per_layer(input_ids) * self.embed_scale_per_layer
        token_part = token_part.view(bsz, seqlen, num_layers, ple_dim)

        # Context-aware projection component.
        proj = self.per_layer_model_projection(inputs_embeds)
        proj = proj * self.per_layer_projection_scale
        proj = proj.view(bsz, seqlen, num_layers, ple_dim)
        proj = self.per_layer_projection_norm(proj)

        return (token_part + proj) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Any,
        attn_metadata: Any,
        lora_mapping: Any = None,
        multimodal_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dtype == torch.long:
            x = self.embed_tokens(input_ids) * self.embed_scale
            if multimodal_embeddings is not None:
                x = _replace_image_placeholders(
                    input_ids=input_ids,
                    text_embeddings=x,
                    multimodal_embeddings=multimodal_embeddings,
                    image_token_id=int(attn_metadata.get("image_token_id", -1)),
                    image_token_count=int(attn_metadata.get("image_token_count", 0)),
                )
        else:
            x = input_ids
        per_layer_inputs = self._compute_per_layer_inputs(input_ids, x)
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[layer.self_attn.kv_scale_cache_idx]
            ple_input = (
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            )
            x = layer(
                x,
                positions,
                kv_cache,
                attn_metadata,
                lora_mapping,
                per_layer_input=ple_input,
            )
        return self.norm(x)


def _assert_text_only_kwargs(kwargs: dict[str, Any]) -> None:
    banned = (
        "pixel_values",
        "image_embeds",
        "audio_values",
        "input_features",
    )
    for k in banned:
        if k in kwargs and kwargs[k] is not None:
            raise NotImplementedError(
                f"Gemma4 text-only path does not support multimodal input: {k}"
            )


class Gemma4ForConditionalGeneration(nn.Module):
    supports_multimodal = True

    def __init__(self, vllm_config: Any, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        self.config = hf_config
        self.model = Gemma4TextModel(
            hf_config,
            vllm_config.quant_config,
            prefix="model",
            runtime_config=getattr(vllm_config, "runtime_config", None),
        )
        self.lm_head = LiteLinear(
            self.model.config.hidden_size,
            self.model.config.vocab_size,
            bias=False,
            prefix="lm_head",
        )

    def get_multimodal_embeddings(self, **kwargs: Any) -> torch.Tensor:
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            raise ValueError("Gemma4 multimodal input requires pixel_values")
        vision_tower = getattr(self.model, "vision_tower", None)
        embed_vision = getattr(self.model, "embed_vision", None)
        if vision_tower is None or embed_vision is None:
            raise ValueError("Gemma4 model was built without vision_config")
        try:
            vision_param = next(vision_tower.parameters())
        except StopIteration:
            vision_param = None
        if vision_param is not None and vision_param.numel() > 0:
            pixel_values = pixel_values.to(
                device=vision_param.device,
                dtype=vision_param.dtype,
            )
        image_position_ids = kwargs.get("image_position_ids")
        if image_position_ids is None:
            hidden = vision_tower(pixel_values)
        else:
            hidden = vision_tower(pixel_values, image_position_ids)
        projected = embed_vision(hidden, kwargs.get("lora_mapping"))
        image_token_count = int(kwargs.get("image_token_count", 0) or 0)
        if image_token_count <= 0:
            return projected
        if projected.shape[0] == 1:
            if projected.shape[1] < image_token_count:
                raise ValueError(
                    "Gemma4 vision embeddings shorter than image_token_count"
                )
            return projected[:, :image_token_count, :]
        flattened = projected.reshape(-1, projected.shape[-1])
        if flattened.shape[0] < image_token_count:
            raise ValueError("Gemma4 vision embeddings shorter than image_token_count")
        return flattened[:image_token_count, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Any,
        attn_metadata: Any,
        lora_mapping: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        multimodal_embeddings = kwargs.pop("multimodal_embeddings", None)
        _assert_text_only_kwargs(kwargs)
        hidden = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            lora_mapping,
            multimodal_embeddings=multimodal_embeddings,
        )
        if getattr(self.model.config, "tie_word_embeddings", False):
            logits = torch.nn.functional.linear(
                hidden[:, -1:, :], self.model.embed_tokens.weight
            )
        else:
            logits = self.lm_head(hidden[:, -1:, :], lora_mapping)
        final_softcap = getattr(self.model.config, "final_logit_softcapping", None)
        if final_softcap is not None and float(final_softcap) > 0:
            logits = torch.tanh(logits / float(final_softcap)) * float(final_softcap)
        return logits


class Gemma4ForCausalLM(Gemma4ForConditionalGeneration):
    pass
