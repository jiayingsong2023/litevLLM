# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any


class PrefillExecutor:
    def __init__(
        self,
        *,
        model: Any,
        input_batch_builder: Any,
        kv_caches: Any,
        multimodal_processor: Any | None = None,
    ) -> None:
        self.model = model
        self.input_batch_builder = input_batch_builder
        self.kv_caches = kv_caches
        self.multimodal_processor = multimodal_processor

    def execute(
        self,
        request_ids: list[str],
        scheduler: Any,
        chunk_len: int,
    ) -> tuple[Any, list[dict[str, Any]], list[bool]]:
        (
            curr_input,
            positions,
            attn_metadata,
            req_dicts_prefill,
            is_last_chunk_flags,
        ) = self.input_batch_builder.build_prefill(request_ids, scheduler, chunk_len)
        model_kwargs: dict[str, Any] = {
            "lora_mapping": attn_metadata.get("lora_mapping"),
        }
        if self.multimodal_processor is not None:
            mm_inputs = self.multimodal_processor.build_prefill_inputs(
                req_dicts_prefill
            )
            if mm_inputs:
                attn_metadata["image_token_count"] = int(
                    mm_inputs.get("image_token_count", 0) or 0
                )
                attn_metadata["image_token_counts"] = list(
                    mm_inputs.get("image_token_counts", [])
                )
                attn_metadata["image_token_id"] = int(
                    mm_inputs.get("image_token_id", -1)
                )
                if mm_inputs.get("image_grid_thw") is not None:
                    attn_metadata["image_grid_thw"] = mm_inputs["image_grid_thw"]
                multimodal_embeddings = (
                    self.multimodal_processor.get_multimodal_embeddings(mm_inputs)
                )
                attn_metadata["multimodal_embeddings"] = multimodal_embeddings
                if multimodal_embeddings is not None:
                    model_kwargs["multimodal_embeddings"] = multimodal_embeddings
        logits = self.model(
            curr_input,
            positions,
            self.kv_caches,
            attn_metadata,
            **model_kwargs,
        )
        self.input_batch_builder.split_per_layer_carries(
            attn_metadata, req_dicts_prefill
        )
        return logits, req_dicts_prefill, is_last_chunk_flags
