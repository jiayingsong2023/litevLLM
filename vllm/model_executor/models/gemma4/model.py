# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear

from .layer import Gemma4DecoderLayer
from vllm.model_executor.models.lite_config import LiteConfig
from .policy_utils import _gemma4_fp32_residual_guard_policy, _get_eps


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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Any,
        attn_metadata: Any,
        lora_mapping: Any = None,
    ) -> torch.Tensor:
        if input_ids.dtype == torch.long:
            x = self.embed_tokens(input_ids) * self.embed_scale
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
        "multimodal_embeddings",
    )
    for k in banned:
        if k in kwargs and kwargs[k] is not None:
            raise NotImplementedError(
                f"Gemma4 text-only path does not support multimodal input: {k}"
            )


class Gemma4ForConditionalGeneration(nn.Module):
    def __init__(self, vllm_config: Any, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Any,
        attn_metadata: Any,
        lora_mapping: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        _assert_text_only_kwargs(kwargs)
        hidden = self.model(
            input_ids, positions, kv_caches, attn_metadata, lora_mapping
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
