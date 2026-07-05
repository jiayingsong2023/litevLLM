# SPDX-License-Identifier: Apache-2.0
from typing import Any

from .base import ModelAdapter, ModelCapabilities, RuntimeModelPolicy


class LlamaAdapter(ModelAdapter):
    model_type = "llama"

    def runtime_policy(
        self,
        model_config: Any,
        runtime_config: Any,
    ) -> RuntimeModelPolicy:
        return RuntimeModelPolicy()

    def install_tuning_config(self, tuning_env: dict[str, str]) -> None:
        return None

    def build_direct_runtime(
        self,
        *,
        model: Any,
        model_config: Any,
        runtime_config: Any,
        tokenizer: Any | None,
        device: Any,
        observer: Any | None,
    ) -> None:
        return None

    def detect(self, model: Any, model_config: Any) -> ModelCapabilities:
        hf_config = getattr(model_config, "hf_config", None)
        num_layers = int(model_config.get_num_layers(None))
        num_kv_heads = int(model_config.get_num_kv_heads(None))
        head_dim = int(model_config.get_head_size())
        num_attention_heads = int(
            getattr(hf_config, "num_attention_heads", num_kv_heads)
            if hf_config is not None
            else num_kv_heads
        )
        max_model_len = int(model_config.get_max_model_len())
        return ModelCapabilities(
            model_type=self.model_type,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_model_len=max_model_len,
            supports_moe=False,
            supports_fp8_kv=True,
            supports_int4_kv=True,
            supports_paged_prefill=True,
            supports_lora=True,
            preferred_kv_dtype="float16",
        )
