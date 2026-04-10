# SPDX-License-Identifier: Apache-2.0
from typing import Any

from .base import ModelAdapter, ModelCapabilities


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


class Gemma4Adapter(ModelAdapter):
    model_type = "gemma4"

    def detect(self, model: Any, model_config: Any) -> ModelCapabilities:
        hf_config = getattr(model_config, "hf_config", None)
        inner_model = getattr(model, "model", model)
        layers = getattr(inner_model, "layers", None)

        num_attention_heads = _int_or(
            getattr(hf_config, "num_attention_heads", 0)
            if hf_config is not None
            else 0,
            0,
        )
        max_kv_heads = 0
        max_head_dim = 0
        if layers is not None:
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                if attn is None:
                    continue
                max_kv_heads = max(max_kv_heads, _int_or(getattr(attn, "num_kv_heads", 0), 0))
                max_head_dim = max(max_head_dim, _int_or(getattr(attn, "head_dim", 0), 0))
                if num_attention_heads <= 0:
                    num_attention_heads = _int_or(getattr(attn, "num_heads", 0), 0)

        if max_kv_heads <= 0:
            max_kv_heads = _int_or(model_config.get_num_kv_heads(None), 1)
        if max_head_dim <= 0:
            hd = getattr(hf_config, "global_head_dim", None) if hf_config is not None else None
            if hd is None:
                hd = getattr(hf_config, "head_dim", None) if hf_config is not None else None
            max_head_dim = _int_or(hd, _int_or(model_config.get_head_size(), 1))
        if num_attention_heads <= 0:
            num_attention_heads = _int_or(
                getattr(hf_config, "num_attention_heads", max_kv_heads)
                if hf_config is not None
                else max_kv_heads,
                max_kv_heads,
            )

        return ModelCapabilities(
            model_type=self.model_type,
            num_layers=int(model_config.get_num_layers(None)),
            num_attention_heads=num_attention_heads,
            num_kv_heads=max_kv_heads,
            head_dim=max_head_dim,
            max_model_len=int(model_config.get_max_model_len()),
            supports_moe=False,
            supports_fp8_kv=True,
            supports_int4_kv=True,
            supports_paged_prefill=True,
            preferred_kv_dtype="bfloat16",
        )

