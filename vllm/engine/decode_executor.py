# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch


class DecodeExecutor:
    def __init__(
        self,
        *,
        model: Any,
        input_batch_builder: Any,
        kv_caches: Any,
        fast_input_ids: torch.Tensor,
        fast_positions: torch.Tensor,
        fast_slot_mapping: torch.Tensor,
        fast_seq_lens: torch.Tensor,
    ) -> None:
        self.model = model
        self.input_batch_builder = input_batch_builder
        self.kv_caches = kv_caches
        self._fast_input_ids = fast_input_ids
        self._fast_positions = fast_positions
        self._fast_slot_mapping = fast_slot_mapping
        self._fast_seq_lens = fast_seq_lens

    def execute_sync_fast(
        self,
        request_ids: list[str],
        scheduler: Any,
    ) -> tuple[Any, list[dict[str, Any]]]:
        input_ids, positions, attn_metadata, req_dicts = (
            self.input_batch_builder.build_decode_fast(
                request_ids,
                scheduler,
                fast_input_ids=self._fast_input_ids,
                fast_positions=self._fast_positions,
                fast_slot_mapping=self._fast_slot_mapping,
                fast_seq_lens=self._fast_seq_lens,
            )
        )
        logits = self.model(
            input_ids,
            positions,
            self.kv_caches,
            attn_metadata,
            lora_mapping=attn_metadata.get("lora_mapping"),
        )
        self.input_batch_builder.split_per_layer_carries(attn_metadata, req_dicts)
        return logits, req_dicts

    def execute_batch(
        self,
        request_ids: list[str],
        scheduler: Any,
    ) -> tuple[Any, list[dict[str, Any]]]:
        curr_input, positions, attn_metadata, req_dicts = (
            self.input_batch_builder.build_decode_batch(request_ids, scheduler)
        )
        logits = self.model(
            curr_input,
            positions,
            self.kv_caches,
            attn_metadata,
            lora_mapping=attn_metadata.get("lora_mapping"),
        )
        self.input_batch_builder.split_per_layer_carries(attn_metadata, req_dicts)
        return logits, req_dicts
