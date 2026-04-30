# SPDX-License-Identifier: Apache-2.0
from typing import Any

from .base import ModelAdapter, ModelCapabilities, RuntimeModelPolicy


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if value % multiple == 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


class Qwen35Adapter(ModelAdapter):
    model_type = "qwen3_5"

    def runtime_policy(
        self,
        model_config: Any,
        runtime_config: Any,
    ) -> RuntimeModelPolicy:
        return RuntimeModelPolicy(
            prefill_chunk_size_high_end=2048,
            prefill_chunk_size_standard=1024,
        )

    def install_tuning_config(self, tuning_env: dict[str, str]) -> None:
        from vllm.model_executor.models.qwen3_5 import set_qwen35_tuning_config

        set_qwen35_tuning_config(tuning_env, locked=True)

    def detect(self, model: Any, model_config: Any) -> ModelCapabilities:
        hf_config = getattr(model_config, "hf_config", None)
        inner_model = getattr(model, "model", model)
        first_layer = None
        if hasattr(inner_model, "layers") and len(inner_model.layers) > 0:
            first_layer = inner_model.layers[0]

        num_attention_heads = 0
        num_kv_heads = 0
        head_dim = 0
        if first_layer is not None:
            num_attention_heads = int(getattr(first_layer, "num_heads", 0) or 0)
            num_kv_heads = int(getattr(first_layer, "num_kv_heads", 0) or 0)
            head_dim = int(getattr(first_layer, "head_dim", 0) or 0)
            if num_attention_heads == 0 and hasattr(first_layer, "self_attn"):
                num_attention_heads = int(
                    getattr(first_layer.self_attn, "num_heads", 0) or 0
                )
                num_kv_heads = int(
                    getattr(first_layer.self_attn, "num_kv_heads", 0) or 0
                )
                head_dim = int(getattr(first_layer.self_attn, "head_dim", 0) or 0)

        if num_attention_heads == 0 or head_dim == 0:
            num_kv_heads = int(model_config.get_num_kv_heads(None))
            head_dim = int(model_config.get_head_size())
            num_attention_heads = int(
                getattr(hf_config, "num_attention_heads", num_kv_heads)
                if hf_config is not None
                else num_kv_heads
            )
            hd = getattr(hf_config, "head_dim", None) if hf_config is not None else None
            if hd is None and isinstance(hf_config, dict):
                hd = hf_config.get("head_dim")
            if hd:
                head_dim = int(hd)

        head_dim = _round_up_to_multiple(int(head_dim), 8)
        supports_moe = "moe" in type(model).__name__.lower()

        return ModelCapabilities(
            model_type=self.model_type,
            num_layers=int(model_config.get_num_layers(None)),
            num_attention_heads=num_attention_heads,
            num_kv_heads=int(num_kv_heads),
            head_dim=head_dim,
            max_model_len=int(model_config.get_max_model_len()),
            supports_moe=supports_moe,
            supports_fp8_kv=True,
            supports_int4_kv=not supports_moe,
            supports_paged_prefill=True,
            preferred_kv_dtype="bfloat16",
        )
