# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from vllm.model_executor.models.deepseek_v4_flash.config import (
    DEEPSEEK_V4_FLASH_SHAPE,
    DeepSeekV4FlashMemoryPolicy,
)

from .base import ModelAdapter, ModelCapabilities, RuntimeModelPolicy


class DeepSeekV4FlashAdapter(ModelAdapter):
    model_type = "deepseek_v4_flash"

    def runtime_policy(
        self,
        model_config: Any,
        runtime_config: Any,
    ) -> RuntimeModelPolicy:
        return RuntimeModelPolicy(
            model_policy={
                "experimental": True,
                "kv_layout": "deepseek_v4_compressed_paged",
                "max_tested_context": (
                    DeepSeekV4FlashMemoryPolicy.max_first_release_context
                ),
                "requires_greedy_batch1_first_release": True,
            },
            kernel_policy={
                "compressed_attention_uses_page_tables": True,
                "q8_linear_kernel": "correctness_first_wrapper",
            },
        )

    def install_tuning_config(self, tuning_env: dict[str, str]) -> None:
        return None

    def detect(self, model: Any, model_config: Any) -> ModelCapabilities:
        shape = DEEPSEEK_V4_FLASH_SHAPE
        max_model_len = int(getattr(model_config, "get_max_model_len", lambda: 8192)())
        return ModelCapabilities(
            model_type=self.model_type,
            num_layers=shape.num_layers,
            num_attention_heads=shape.num_attention_heads,
            num_kv_heads=shape.num_kv_heads,
            head_dim=shape.head_dim,
            max_model_len=min(
                max_model_len,
                DeepSeekV4FlashMemoryPolicy.max_first_release_context,
            ),
            supports_moe=True,
            supports_fp8_kv=False,
            supports_int4_kv=False,
            supports_paged_prefill=False,
            preferred_kv_dtype="deepseek_v4_compressed",
        )
