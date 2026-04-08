# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch


class PrefillExecutor:
    def __init__(
        self,
        *,
        model: Any,
        device: torch.device,
        num_layers: int,
        num_blocks_per_seq: int,
        max_model_len: int,
        kv_caches: Any,
        kv_scale_caches: Any,
        inf_config: Any,
        stack_per_layer_carries: Any,
        split_per_layer_carries: Any,
    ) -> None:
        self.model = model
        self.device = device
        self.num_layers = num_layers
        self.num_blocks_per_seq = num_blocks_per_seq
        self.max_model_len = max_model_len
        self.kv_caches = kv_caches
        self.kv_scale_caches = kv_scale_caches
        self.inf_config = inf_config
        self._stack_per_layer_carries = stack_per_layer_carries
        self._split_per_layer_carries = split_per_layer_carries

    def execute(
        self,
        request_ids: list[str],
        scheduler: Any,
        chunk_len: int,
    ) -> tuple[Any, list[dict[str, Any]], list[bool]]:
        curr_input_rows = []
        position_rows = []
        slot_mapping_rows = []
        block_tables = []
        seq_lens_prefill = []
        kv_start_indices = []
        req_dicts_prefill = [scheduler.get_request(rid) for rid in request_ids]
        is_last_chunk_flags = []

        for rid in request_ids:
            req = scheduler.get_request(rid)
            slot_idx = req["slot_idx"]
            all_input_ids = req["input_ids"]
            processed_len = req["seq_len"]
            is_last_chunk = processed_len + chunk_len >= len(all_input_ids)
            is_last_chunk_flags.append(is_last_chunk)

            curr_chunk_ids = all_input_ids[processed_len : processed_len + chunk_len]
            curr_input_rows.append(curr_chunk_ids)
            position_rows.append(
                torch.arange(
                    processed_len,
                    processed_len + chunk_len,
                    device=self.device,
                    dtype=torch.long,
                )
            )
            slot_mapping_rows.append(
                slot_idx * self.max_model_len
                + torch.arange(
                    processed_len,
                    processed_len + chunk_len,
                    device=self.device,
                    dtype=torch.long,
                )
            )
            start_block = slot_idx * self.num_blocks_per_seq
            block_tables.append(
                torch.arange(
                    start_block,
                    start_block + self.num_blocks_per_seq,
                    device=self.device,
                    dtype=torch.int32,
                )
            )
            seq_lens_prefill.append(processed_len + chunk_len)
            kv_start_indices.append(processed_len)

        curr_input = torch.tensor(curr_input_rows, device=self.device)
        positions = torch.stack(position_rows, dim=0).to(self.device)
        slot_mapping = torch.cat(slot_mapping_rows, dim=0).to(self.device)
        block_tables_t = torch.stack(block_tables).to(self.device)
        attn_carry_prefill = self._stack_per_layer_carries(
            req_dicts_prefill, self.num_layers, "linear_attn_carry"
        )
        conv_carry_prefill = self._stack_per_layer_carries(
            req_dicts_prefill, self.num_layers, "linear_conv_carry"
        )

        attn_metadata = {
            "slot_mapping": slot_mapping,
            "seq_lens": torch.tensor(
                seq_lens_prefill, device=self.device, dtype=torch.int32
            ),
            "is_prefill": True,
            "kv_start_indices": torch.tensor(
                kv_start_indices, device=self.device, dtype=torch.int32
            ),
            "block_tables": block_tables_t,
            "linear_attn_carry": attn_carry_prefill,
            "linear_conv_carry": conv_carry_prefill,
            "kv_scale_cache": self.kv_scale_caches,
            "kv_cache_dtype": self.inf_config.kv_type,
            "k_scale": self.inf_config.k_scale,
            "v_scale": self.inf_config.v_scale,
            "config": self.inf_config,
        }
        lora_mapping = [scheduler.get_request(rid).get("lora_id") for rid in request_ids]
        logits = self.model(
            curr_input,
            positions,
            self.kv_caches,
            attn_metadata,
            lora_mapping=lora_mapping,
        )
        self._split_per_layer_carries(
            attn_metadata["linear_attn_carry"],
            req_dicts_prefill,
            "linear_attn_carry",
        )
        self._split_per_layer_carries(
            attn_metadata["linear_conv_carry"],
            req_dicts_prefill,
            "linear_conv_carry",
        )
        return logits, req_dicts_prefill, is_last_chunk_flags
