# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch


class DecodeExecutor:
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
        fast_input_ids: torch.Tensor,
        fast_positions: torch.Tensor,
        fast_slot_mapping: torch.Tensor,
        fast_seq_lens: torch.Tensor,
        fast_block_tables: torch.Tensor,
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
        self._fast_input_ids = fast_input_ids
        self._fast_positions = fast_positions
        self._fast_slot_mapping = fast_slot_mapping
        self._fast_seq_lens = fast_seq_lens
        self._fast_block_tables = fast_block_tables
        self._stack_per_layer_carries = stack_per_layer_carries
        self._split_per_layer_carries = split_per_layer_carries

    def execute_sync_fast(
        self,
        request_ids: list[str],
        scheduler: Any,
    ) -> tuple[Any, list[dict[str, Any]]]:
        bs = len(request_ids)
        input_ids = self._fast_input_ids[:bs]
        positions = self._fast_positions[:bs]
        slot_mapping = self._fast_slot_mapping[:bs]
        seq_lens = self._fast_seq_lens[:bs]

        req_dicts = []
        lora_mapping = []
        for i, rid in enumerate(request_ids):
            req = scheduler.get_request(rid)
            req_dicts.append(req)
            input_ids[i, 0] = req["generated_ids"][-1]
            positions[i, 0] = req["seq_len"]
            slot_mapping[i] = req["slot_idx"] * self.max_model_len + req["seq_len"]
            seq_lens[i] = req["seq_len"] + 1
            lora_mapping.append(req.get("lora_id"))

        slots_t = torch.tensor([req["slot_idx"] for req in req_dicts], device=self.device)
        block_tables = self._fast_block_tables.index_select(0, slots_t)
        attn_carry_batch = self._stack_per_layer_carries(
            req_dicts, self.num_layers, "linear_attn_carry"
        )
        conv_carry_batch = self._stack_per_layer_carries(
            req_dicts, self.num_layers, "linear_conv_carry"
        )

        attn_metadata = {
            "slot_mapping": slot_mapping,
            "seq_lens": seq_lens,
            "block_tables": block_tables,
            "is_prefill": False,
            "kv_start_indices": positions.squeeze(1).to(torch.int32),
            "linear_attn_carry": attn_carry_batch,
            "linear_conv_carry": conv_carry_batch,
            "kv_cache_dtype": self.inf_config.kv_type,
            "k_scale": self.inf_config.k_scale,
            "v_scale": self.inf_config.v_scale,
            "config": self.inf_config,
        }

        logits = self.model(
            input_ids, positions, self.kv_caches, attn_metadata, lora_mapping=lora_mapping
        )
        self._split_per_layer_carries(
            attn_metadata["linear_attn_carry"], req_dicts, "linear_attn_carry"
        )
        self._split_per_layer_carries(
            attn_metadata["linear_conv_carry"], req_dicts, "linear_conv_carry"
        )
        return logits, req_dicts

    def execute_batch(
        self,
        request_ids: list[str],
        scheduler: Any,
    ) -> tuple[Any, list[dict[str, Any]]]:
        input_tokens = []
        slot_indices = []
        seq_lens = []
        pos_indices = []

        for rid in request_ids:
            req = scheduler.get_request(rid)
            last_token = req["generated_ids"][-1]
            input_tokens.append([last_token])
            slot_indices.append(req["slot_idx"])
            current_len = req["seq_len"]
            seq_lens.append(current_len + 1)
            pos_indices.append(current_len)

        curr_input = torch.tensor(input_tokens, device=self.device)
        positions = torch.tensor(pos_indices, device=self.device).unsqueeze(1)
        batch_block_tables = []
        for slot_idx in slot_indices:
            start_block = slot_idx * self.num_blocks_per_seq
            batch_block_tables.append(
                torch.arange(
                    start_block,
                    start_block + self.num_blocks_per_seq,
                    dtype=torch.int32,
                )
            )
        block_tables = torch.stack(batch_block_tables).to(self.device)
        slot_mapping = torch.tensor(
            [s * self.max_model_len + p for s, p in zip(slot_indices, pos_indices)],
            device=self.device,
            dtype=torch.long,
        )
        req_dicts = [scheduler.get_request(rid) for rid in request_ids]
        attn_carry_batch = self._stack_per_layer_carries(
            req_dicts, self.num_layers, "linear_attn_carry"
        )
        conv_carry_batch = self._stack_per_layer_carries(
            req_dicts, self.num_layers, "linear_conv_carry"
        )
        attn_metadata = {
            "slot_mapping": slot_mapping,
            "seq_lens": torch.tensor(seq_lens, device=self.device, dtype=torch.int32),
            "block_tables": block_tables,
            "is_prefill": False,
            "kv_start_indices": torch.tensor(
                pos_indices, device=self.device, dtype=torch.int32
            ),
            "linear_attn_carry": attn_carry_batch,
            "linear_conv_carry": conv_carry_batch,
            "kv_scale_cache": self.kv_scale_caches,
            "kv_cache_dtype": self.inf_config.kv_type,
            "k_scale": self.inf_config.k_scale,
            "v_scale": self.inf_config.v_scale,
            "config": self.inf_config,
        }
        lora_mapping = [scheduler.get_request(rid).get("lora_id") for rid in request_ids]
        logits = self.model(
            curr_input, positions, self.kv_caches, attn_metadata, lora_mapping=lora_mapping
        )
        self._split_per_layer_carries(
            attn_metadata["linear_attn_carry"], req_dicts, "linear_attn_carry"
        )
        self._split_per_layer_carries(
            attn_metadata["linear_conv_carry"], req_dicts, "linear_conv_carry"
        )
        return logits, req_dicts
