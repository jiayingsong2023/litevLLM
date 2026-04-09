# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch


class PrefillExecutor:
    def __init__(
        self,
        *,
        model: Any,
        input_batch_builder: Any,
        kv_caches: Any,
    ) -> None:
        self.model = model
        self.input_batch_builder = input_batch_builder
        self.kv_caches = kv_caches

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
        lora_mapping = [scheduler.get_request(rid).get("lora_id") for rid in request_ids]
        logits = self.model(
            curr_input,
            positions,
            self.kv_caches,
            attn_metadata,
            lora_mapping=lora_mapping,
        )
        self.input_batch_builder.split_per_layer_carries(attn_metadata, req_dicts_prefill)
        return logits, req_dicts_prefill, is_last_chunk_flags
