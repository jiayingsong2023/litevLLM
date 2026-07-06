# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.models.lite_config import LiteConfig

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
    if image_token_count <= 0:
        return text_embeddings
    image_mask = input_ids.eq(int(image_token_id))
    if int(image_mask.sum().item()) != int(image_token_count):
        raise ValueError("image placeholder count does not match image_token_count")
    if multimodal_embeddings.dim() == 2:
        multimodal_embeddings = multimodal_embeddings.unsqueeze(0)
    image_embeddings = multimodal_embeddings.reshape(-1, text_embeddings.shape[-1])
    if image_embeddings.shape[0] != int(image_token_count):
        raise ValueError("image embedding count does not match image_token_count")
    output = text_embeddings.clone()
    output[image_mask] = image_embeddings.to(dtype=output.dtype, device=output.device)
    return output


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
        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    self.config,
                    quant_config,
                    prefix=f"layers.{i}",
                    layer_idx=i,
                    fp32_residual_guard_enabled=fp32_residual_guard_enabled,
                    fp32_residual_guard_start=fp32_residual_guard_start,
                    fp32_residual_guard_span=fp32_residual_guard_span,
                    runtime_config=runtime_config,
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=_get_eps(self.config))
        vision_config = getattr(hf_config, "vision_config", None)
        if vision_config is not None:
            vision_lite_config = LiteConfig(vision_config)
            self.vision_tower = Gemma4VisionTower(vision_config)
            self.embed_vision = Gemma4VisionProjector(
                int(vision_lite_config.hidden_size),
                int(self.config.hidden_size),
            )

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
        for i, layer in enumerate(self.layers):
            x = layer(x, positions, kv_caches[i], attn_metadata, lora_mapping)
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
        hidden = vision_tower(pixel_values)
        projected = embed_vision(hidden)
        image_token_count = int(kwargs.get("image_token_count", 0) or 0)
        if image_token_count <= 0:
            return projected
        if projected.shape[1] < image_token_count:
            raise ValueError("Gemma4 vision embeddings shorter than image_token_count")
        return projected[:, :image_token_count, :]

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
