# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from vllm.engine.custom_runtime_components import CustomRuntimeComponents
from vllm.model_executor.models.deepseek_v4_flash.config import (
    DEEPSEEK_V4_FLASH_SHAPE,
    DeepSeekV4FlashMemoryPolicy,
)
from vllm.model_executor.models.deepseek_v4_flash.executors import (
    DeepSeekDecodeExecutor,
    DeepSeekPrefillExecutor,
)
from vllm.model_executor.models.deepseek_v4_flash.kv_lifecycle import (
    DeepSeekKVLifecycleAdapter,
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

    def build_executors(
        self,
        *,
        model: Any,
        model_config: Any,
        runtime_config: Any,
        observer: Any | None,
        **kwargs: Any,
    ) -> CustomRuntimeComponents:
        device = torch.device(
            kwargs.get("device") or getattr(model, "device", lambda: "cuda:0")()
        )
        context_length = int(
            getattr(runtime_config, "kv_max_model_len", None)
            or getattr(runtime_config, "max_model_len", None)
            or getattr(model_config, "get_max_model_len", lambda: 8192)()
        )
        max_active_requests = int(
            kwargs.get("max_active_requests")
            or getattr(runtime_config, "kv_max_active_requests", 1)
            or 1
        )
        kv_block_manager = DeepSeekKVLifecycleAdapter(
            model=model,
            device=device,
            context_length=context_length,
            max_active_requests=max_active_requests,
            observer=observer,
        )
        return CustomRuntimeComponents(
            prefill_executor=DeepSeekPrefillExecutor(model=model, observer=observer),
            decode_executor=DeepSeekDecodeExecutor(model=model, observer=observer),
            kv_block_manager=kv_block_manager,
            multimodal_processor=None,
        )

    def estimate_kv_bytes(
        self,
        *,
        max_active_requests: int,
        context_length: int,
    ) -> int:
        block_size = 16
        raw_blocks = (int(context_length) + block_size - 1) // block_size
        raw_bytes = (
            int(max_active_requests)
            * DEEPSEEK_V4_FLASH_SHAPE.num_layers
            * raw_blocks
            * block_size
            * DEEPSEEK_V4_FLASH_SHAPE.num_kv_heads
            * DEEPSEEK_V4_FLASH_SHAPE.head_dim
            * 2
        )
        # ponytail: rough static cap; replace with model-reported compressed KV bytes
        # when DeepSeek exposes exact pool allocation sizing.
        return raw_bytes

    def estimate_staging_bytes(self, *, max_active_requests: int) -> int:
        del max_active_requests
        return int(DeepSeekV4FlashMemoryPolicy.default_expert_cache_bytes)

    def admission_cap(
        self,
        *,
        current_max_active_requests: int,
        max_model_len: int,
        runtime_config: Any,
    ) -> int:
        """Keep the model-owned compressed-KV runtime inside its memory budget."""
        budget = int(
            (getattr(runtime_config, "model_policy", {}) or {}).get(
                "runtime_budget_bytes", 0
            )
            or 0
        )
        if budget <= 0:
            return current_max_active_requests
        per_request = self.estimate_kv_bytes(
            max_active_requests=1,
            context_length=max_model_len,
        ) + self.estimate_staging_bytes(max_active_requests=1)
        if per_request <= 0:
            return current_max_active_requests
        return max(1, min(current_max_active_requests, budget // per_request))

    def validate_request(
        self,
        *,
        sampling_params: Any,
        lora_id: str | None,
        lora_request: Any | None,
        multi_modal_data: dict[str, Any] | None,
    ) -> None:
        if lora_id is not None or lora_request is not None:
            raise ValueError("DeepSeek V4 Flash LiteEngine path does not support LoRA")
        if multi_modal_data:
            raise ValueError(
                "DeepSeek V4 Flash LiteEngine path does not support multimodal inputs"
            )
        if int(getattr(sampling_params, "n", 1) or 1) != 1:
            raise ValueError("DeepSeek V4 Flash LiteEngine path supports n=1 only")
        max_tokens = getattr(sampling_params, "max_tokens", None)
        if max_tokens is None or int(max_tokens) <= 0:
            raise ValueError(
                "DeepSeek V4 Flash LiteEngine path requires max_tokens > 0"
            )
        if float(getattr(sampling_params, "temperature", 0.0) or 0.0) != 0.0:
            raise ValueError(
                "DeepSeek V4 Flash LiteEngine path supports greedy sampling only"
            )
        if float(getattr(sampling_params, "top_p", 1.0) or 1.0) != 1.0:
            raise ValueError(
                "DeepSeek V4 Flash LiteEngine path supports greedy sampling only"
            )
        top_k = int(getattr(sampling_params, "top_k", -1) or -1)
        if top_k not in (-1, 0):
            raise ValueError(
                "DeepSeek V4 Flash LiteEngine path supports greedy sampling only"
            )
        if getattr(sampling_params, "structured_outputs", None) is not None:
            raise ValueError(
                "DeepSeek V4 Flash LiteEngine path does not support structured outputs"
            )

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
            supports_chunked_prefill=False,
        )
