# SPDX-License-Identifier: Apache-2.0
from typing import Any

from .base import ModelCapabilities, RuntimeModelPolicy
from .llama import LlamaAdapter


class Qwen2VLAdapter(LlamaAdapter):
    model_type = "qwen2_vl"

    def runtime_policy(
        self,
        model_config: Any,
        runtime_config: Any,
    ) -> RuntimeModelPolicy:
        return RuntimeModelPolicy()

    def detect(self, model: Any, model_config: Any) -> ModelCapabilities:
        caps = super().detect(model, model_config)
        return ModelCapabilities(
            model_type=self.model_type,
            num_layers=caps.num_layers,
            num_attention_heads=caps.num_attention_heads,
            num_kv_heads=caps.num_kv_heads,
            head_dim=caps.head_dim,
            max_model_len=caps.max_model_len,
            supports_moe=caps.supports_moe,
            supports_fp8_kv=caps.supports_fp8_kv,
            supports_int4_kv=caps.supports_int4_kv,
            supports_paged_prefill=caps.supports_paged_prefill,
            preferred_kv_dtype="bfloat16",
            supports_chunked_prefill=caps.supports_chunked_prefill,
            supports_lora=caps.supports_lora,
            supports_multimodal=True,
        )
