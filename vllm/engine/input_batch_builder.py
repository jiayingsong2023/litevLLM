# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from vllm.engine.request_state import RequestState
from vllm.kernels.triton.compute_slot_mapping import compute_slot_mapping


class InputBatchBuilder:
    def __init__(
        self,
        *,
        device: torch.device,
        max_model_len: int,
        num_layers: int,
        kv_block_manager: Any,
        inf_config: Any,
        stack_per_layer_carries: Any,
        split_per_layer_carries: Any,
    ) -> None:
        self.device = device
        self.max_model_len = max_model_len
        self.num_layers = num_layers
        self.kv_block_manager = kv_block_manager
        self.inf_config = inf_config
        self._stack_per_layer_carries = stack_per_layer_carries
        self._split_per_layer_carries = split_per_layer_carries
        self._decode_input_ids: torch.Tensor | None = None
        self._decode_positions: torch.Tensor | None = None
        self._decode_slot_mapping: torch.Tensor | None = None
        self._decode_seq_lens: torch.Tensor | None = None
        self._decode_block_tables: torch.Tensor | None = None

    def build_prefill(
        self,
        request_ids: list[str],
        scheduler: Any,
        chunk_len: int,
    ) -> tuple[
        torch.Tensor, torch.Tensor, dict[str, Any], list[RequestState], list[bool]
    ]:
        curr_input_rows = []
        position_rows = []
        block_tables = []
        chunk_lengths = []
        seq_lens_prefill = []
        kv_start_indices = []
        req_dicts_prefill = [scheduler.get_request(rid) for rid in request_ids]
        is_last_chunk_flags = []

        for rid in request_ids:
            req = scheduler.get_request(rid)
            slot_idx = int(req.slot_idx)
            all_input_ids = req.input_ids
            processed_len = req.seq_len
            prefix_hit_len = int(req.prefix_hit_len or req._prefix_cache_hit_len or 0)
            start_pos = max(processed_len, prefix_hit_len)

            actual_chunk_len = min(chunk_len, len(all_input_ids) - start_pos)
            if actual_chunk_len <= 0:
                raise ValueError(
                    "zero-token prefill for request "
                    f"{rid}: start_pos={start_pos}, prompt_len={len(all_input_ids)}, "
                    f"seq_len={processed_len}, prefix_hit_len={prefix_hit_len}, "
                    f"chunk_len={chunk_len}"
                )
            is_last_chunk = start_pos + actual_chunk_len >= len(all_input_ids)
            is_last_chunk_flags.append(is_last_chunk)

            curr_chunk_ids = all_input_ids[start_pos : start_pos + actual_chunk_len]
            curr_input_rows.append(curr_chunk_ids)
            position_rows.append(
                torch.arange(
                    start_pos,
                    start_pos + actual_chunk_len,
                    device=self.device,
                    dtype=torch.long,
                )
            )
            chunk_lengths.append(actual_chunk_len)
            self.kv_block_manager.update_block_table_row(slot_idx, rid)
            block_tables.append(self.kv_block_manager.block_table_for_slot(slot_idx))
            seq_lens_prefill.append(start_pos + actual_chunk_len)
            kv_start_indices.append(start_pos)

        curr_input = torch.tensor(curr_input_rows, device=self.device)
        positions = torch.stack(position_rows, dim=0).to(self.device)
        total_tokens = sum(chunk_lengths)
        query_start_loc = torch.zeros(
            len(chunk_lengths) + 1, device=self.device, dtype=torch.int32
        )
        query_start_loc[1:] = torch.cumsum(
            torch.tensor(chunk_lengths, device=self.device, dtype=torch.int32), dim=0
        )
        slot_mapping = torch.empty(total_tokens, device=self.device, dtype=torch.long)
        block_tables_t = torch.stack(block_tables, dim=0).to(self.device)
        compute_slot_mapping(
            query_start_loc,
            positions.view(-1),
            block_tables_t,
            self.kv_block_manager.block_size,
            slot_mapping,
            pad_id=-1,
        )
        attn_carry_prefill = self._stack_per_layer_carries(
            req_dicts_prefill, self.num_layers, "linear_attn_carry"
        )
        conv_carry_prefill = self._stack_per_layer_carries(
            req_dicts_prefill, self.num_layers, "linear_conv_carry"
        )
        lora_mapping = [req.lora_id for req in req_dicts_prefill]
        self._assert_single_lora_adapter(lora_mapping)
        multimodal_flags = [
            self._is_multimodal_request(req) for req in req_dicts_prefill
        ]
        # CPU-side scalars / lists carried through attn_metadata so hot-path
        # model code (e.g. Gemma4 60-layer loop) can avoid `.item()` D->H syncs.
        seq_lens_cpu_list = [int(v) for v in seq_lens_prefill]
        kv_start_cpu_list = [int(v) for v in kv_start_indices]
        max_seq_len_cpu = max(seq_lens_cpu_list) if seq_lens_cpu_list else 0
        max_kv_start_cpu = max(kv_start_cpu_list) if kv_start_cpu_list else 0
        attn_metadata = {
            "slot_mapping": slot_mapping,
            "seq_lens": torch.tensor(
                seq_lens_prefill, device=self.device, dtype=torch.int32
            ),
            "seq_lens_cpu": seq_lens_cpu_list,
            "max_seq_len_cpu": int(max_seq_len_cpu),
            "max_kv_start_cpu": int(max_kv_start_cpu),
            "is_prefill": True,
            "kv_start_indices": torch.tensor(
                kv_start_indices, device=self.device, dtype=torch.int32
            ),
            "kv_start_indices_cpu": kv_start_cpu_list,
            "block_tables": block_tables_t,
            "linear_attn_carry": attn_carry_prefill,
            "linear_conv_carry": conv_carry_prefill,
            "kv_scale_cache": self.kv_block_manager.kv_scale_caches,
            "kv_cache_dtype": self.inf_config.kv_type,
            "k_scale": self.inf_config.k_scale,
            "v_scale": self.inf_config.v_scale,
            "config": self.inf_config,
            "lora_mapping": lora_mapping,
            "lora_adapter_count": self._lora_adapter_count(lora_mapping),
            "mixed_lora_batch": self._is_mixed_lora_batch(lora_mapping),
            "multimodal_request_count": self._multimodal_request_count(
                multimodal_flags
            ),
            "has_multimodal_requests": any(multimodal_flags),
            "mixed_multimodal_batch": self._is_mixed_multimodal_batch(multimodal_flags),
            "multimodal_lora_request_count": self._multimodal_lora_request_count(
                req_dicts_prefill
            ),
            "has_multimodal_lora_requests": any(
                self._is_multimodal_lora_request(req) for req in req_dicts_prefill
            ),
        }
        return (
            curr_input,
            positions,
            attn_metadata,
            req_dicts_prefill,
            is_last_chunk_flags,
        )

    def build_decode_batch(
        self,
        request_ids: list[str],
        scheduler: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], list[RequestState]]:
        bs = len(request_ids)
        self._ensure_decode_scratch(bs)
        assert self._decode_input_ids is not None
        assert self._decode_positions is not None
        assert self._decode_slot_mapping is not None
        assert self._decode_seq_lens is not None
        curr_input = self._decode_input_ids[:bs]
        positions = self._decode_positions[:bs]
        slot_mapping = self._decode_slot_mapping[:bs]
        seq_lens_t = self._decode_seq_lens[:bs]

        slot_indices = []
        seq_lens_cpu_list = []
        positions_cpu_list = []
        req_dicts = []

        for i, rid in enumerate(request_ids):
            req = scheduler.get_request(rid)
            req_dicts.append(req)
            last_token = req.generated_ids[-1]
            current_len = int(req.seq_len)
            slot_idx = int(req.slot_idx)
            curr_input[i, 0] = last_token
            positions[i, 0] = current_len
            seq_lens_t[i] = current_len + 1
            slot_indices.append(slot_idx)
            seq_lens_cpu_list.append(current_len + 1)
            positions_cpu_list.append(current_len)
            self.kv_block_manager.update_block_table_row(slot_idx, rid)

        block_tables = torch.stack(
            [
                self.kv_block_manager.block_table_for_slot(slot_idx)
                for slot_idx in slot_indices
            ]
        ).to(self.device)
        query_start_loc = torch.arange(bs + 1, device=self.device, dtype=torch.int32)
        compute_slot_mapping(
            query_start_loc,
            positions.view(-1),
            block_tables,
            self.kv_block_manager.block_size,
            slot_mapping,
            pad_id=-1,
        )
        attn_carry_batch = self._stack_per_layer_carries(
            req_dicts, self.num_layers, "linear_attn_carry"
        )
        conv_carry_batch = self._stack_per_layer_carries(
            req_dicts, self.num_layers, "linear_conv_carry"
        )
        lora_mapping = [req.lora_id for req in req_dicts]
        self._assert_single_lora_adapter(lora_mapping)
        multimodal_flags = [self._is_multimodal_request(req) for req in req_dicts]
        max_seq_len_cpu = max(seq_lens_cpu_list) if seq_lens_cpu_list else 0
        attn_metadata = {
            "slot_mapping": slot_mapping,
            "seq_lens": seq_lens_t,
            "seq_lens_cpu": seq_lens_cpu_list,
            "max_seq_len_cpu": int(max_seq_len_cpu),
            "positions_cpu": positions_cpu_list,
            "block_tables": block_tables,
            "is_prefill": False,
            "kv_start_indices": positions.squeeze(1).to(torch.int32),
            "kv_start_indices_cpu": positions_cpu_list,
            "linear_attn_carry": attn_carry_batch,
            "linear_conv_carry": conv_carry_batch,
            "kv_scale_cache": self.kv_block_manager.kv_scale_caches,
            "kv_cache_dtype": self.inf_config.kv_type,
            "k_scale": self.inf_config.k_scale,
            "v_scale": self.inf_config.v_scale,
            "config": self.inf_config,
            "lora_mapping": lora_mapping,
            "lora_adapter_count": self._lora_adapter_count(lora_mapping),
            "mixed_lora_batch": self._is_mixed_lora_batch(lora_mapping),
            "multimodal_request_count": self._multimodal_request_count(
                multimodal_flags
            ),
            "has_multimodal_requests": any(multimodal_flags),
            "mixed_multimodal_batch": self._is_mixed_multimodal_batch(multimodal_flags),
            "multimodal_lora_request_count": self._multimodal_lora_request_count(
                req_dicts
            ),
            "has_multimodal_lora_requests": any(
                self._is_multimodal_lora_request(req) for req in req_dicts
            ),
        }
        return curr_input, positions, attn_metadata, req_dicts

    def _ensure_decode_scratch(self, batch_size: int) -> None:
        if (
            self._decode_input_ids is not None
            and self._decode_input_ids.shape[0] >= batch_size
        ):
            return
        self._decode_input_ids = torch.empty(
            (batch_size, 1), device=self.device, dtype=torch.long
        )
        self._decode_positions = torch.empty(
            (batch_size, 1), device=self.device, dtype=torch.long
        )
        self._decode_slot_mapping = torch.empty(
            (batch_size,), device=self.device, dtype=torch.long
        )
        self._decode_seq_lens = torch.empty(
            (batch_size,), device=self.device, dtype=torch.int32
        )
        self._decode_block_tables = torch.empty(
            (batch_size, self.kv_block_manager.num_blocks_per_seq),
            device=self.device,
            dtype=torch.int32,
        )

    def build_decode_fast(
        self,
        request_ids: list[str],
        scheduler: Any,
        *,
        fast_input_ids: torch.Tensor,
        fast_positions: torch.Tensor,
        fast_slot_mapping: torch.Tensor,
        fast_seq_lens: torch.Tensor,
        fast_block_tables: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], list[RequestState]]:
        bs = len(request_ids)
        input_ids = fast_input_ids[:bs]
        positions = fast_positions[:bs]
        slot_mapping = fast_slot_mapping[:bs]
        seq_lens = fast_seq_lens[:bs]
        if fast_block_tables is not None:
            block_tables = fast_block_tables[:bs]
        else:
            self._ensure_decode_scratch(bs)
            assert self._decode_block_tables is not None
            block_tables = self._decode_block_tables[:bs]

        req_dicts = []
        # CPU-side scalars built during the same loop that writes device tensors,
        # so downstream models can consume Python ints without triggering
        # hipMemcpyWithStream / `.item()` syncs on the decode hot path.
        seq_lens_cpu_list: list[int] = []
        positions_cpu_list: list[int] = []
        slot_indices = []
        for i, rid in enumerate(request_ids):
            req = scheduler.get_request(rid)
            req_dicts.append(req)
            prev_seq_len = int(req.seq_len)
            slot_idx = int(req.slot_idx)
            last_token_tensor = req._last_token_tensor
            if isinstance(last_token_tensor, torch.Tensor):
                input_ids[i, 0].copy_(last_token_tensor.reshape(()))
            else:
                last_token = req.generated_ids[-1]
                input_ids[i, 0] = last_token
            positions[i, 0] = prev_seq_len
            seq_lens[i] = prev_seq_len + 1
            positions_cpu_list.append(prev_seq_len)
            seq_lens_cpu_list.append(prev_seq_len + 1)
            slot_indices.append(slot_idx)
            self.kv_block_manager.update_block_table_row(slot_idx, rid)

        # Copy the dynamic block-table rows from KVBlockManager into the
        # caller-provided fast buffer so the model reuses the pre-allocated
        # device tensor instead of allocating a new one each decode step.
        for i, slot_idx in enumerate(slot_indices):
            block_tables[i].copy_(self.kv_block_manager.block_table_for_slot(slot_idx))
        query_start_loc = torch.arange(bs + 1, device=self.device, dtype=torch.int32)
        compute_slot_mapping(
            query_start_loc,
            positions.view(-1),
            block_tables,
            self.kv_block_manager.block_size,
            slot_mapping,
            pad_id=-1,
        )
        lora_mapping = [req.lora_id for req in req_dicts]
        self._assert_single_lora_adapter(lora_mapping)
        multimodal_flags = [self._is_multimodal_request(req) for req in req_dicts]
        attn_carry_batch = self._stack_per_layer_carries(
            req_dicts, self.num_layers, "linear_attn_carry"
        )
        conv_carry_batch = self._stack_per_layer_carries(
            req_dicts, self.num_layers, "linear_conv_carry"
        )
        max_seq_len_cpu = max(seq_lens_cpu_list) if seq_lens_cpu_list else 0
        attn_metadata = {
            "slot_mapping": slot_mapping,
            "seq_lens": seq_lens,
            "seq_lens_cpu": seq_lens_cpu_list,
            "max_seq_len_cpu": int(max_seq_len_cpu),
            "positions_cpu": positions_cpu_list,
            "block_tables": block_tables,
            "is_prefill": False,
            "kv_start_indices": positions.squeeze(1).to(torch.int32),
            "kv_start_indices_cpu": positions_cpu_list,
            "linear_attn_carry": attn_carry_batch,
            "linear_conv_carry": conv_carry_batch,
            "kv_scale_cache": self.kv_block_manager.kv_scale_caches,
            "kv_cache_dtype": self.inf_config.kv_type,
            "k_scale": self.inf_config.k_scale,
            "v_scale": self.inf_config.v_scale,
            "config": self.inf_config,
            "lora_mapping": lora_mapping,
            "lora_adapter_count": self._lora_adapter_count(lora_mapping),
            "mixed_lora_batch": self._is_mixed_lora_batch(lora_mapping),
            "multimodal_request_count": self._multimodal_request_count(
                multimodal_flags
            ),
            "has_multimodal_requests": any(multimodal_flags),
            "mixed_multimodal_batch": self._is_mixed_multimodal_batch(multimodal_flags),
            "multimodal_lora_request_count": self._multimodal_lora_request_count(
                req_dicts
            ),
            "has_multimodal_lora_requests": any(
                self._is_multimodal_lora_request(req) for req in req_dicts
            ),
        }
        return input_ids, positions, attn_metadata, req_dicts

    @staticmethod
    def _lora_adapter_count(lora_mapping: list[str | None]) -> int:
        return len({str(item) for item in lora_mapping if item})

    @classmethod
    def _is_mixed_lora_batch(cls, lora_mapping: list[str | None]) -> bool:
        active = {str(item) for item in lora_mapping if item}
        has_base = any(not item for item in lora_mapping)
        return len(active) > 1 or (bool(active) and has_base)

    @classmethod
    def _assert_single_lora_adapter(cls, lora_mapping: list[str | None]) -> None:
        if cls._is_mixed_lora_batch(lora_mapping):
            raise ValueError("Phase 1 does not support mixed LoRA batches")

    @staticmethod
    def _is_multimodal_request(request: RequestState) -> bool:
        return bool(
            request.is_multimodal or (request.multi_modal_data or {}).get("image")
        )

    @classmethod
    def _is_multimodal_lora_request(cls, request: RequestState) -> bool:
        return cls._is_multimodal_request(request) and bool(request.lora_id)

    @classmethod
    def _multimodal_request_count(cls, multimodal_flags: list[bool]) -> int:
        return sum(1 for flag in multimodal_flags if flag)

    @classmethod
    def _is_mixed_multimodal_batch(cls, multimodal_flags: list[bool]) -> bool:
        multimodal_count = cls._multimodal_request_count(multimodal_flags)
        return 0 < multimodal_count < len(multimodal_flags)

    @classmethod
    def _multimodal_lora_request_count(cls, req_dicts: list[RequestState]) -> int:
        return sum(1 for req in req_dicts if cls._is_multimodal_lora_request(req))

    def split_per_layer_carries(
        self,
        attn_metadata: dict[str, Any],
        req_dicts: list[RequestState],
    ) -> None:
        self._split_per_layer_carries(
            attn_metadata["linear_attn_carry"], req_dicts, "linear_attn_carry"
        )
        self._split_per_layer_carries(
            attn_metadata["linear_conv_carry"], req_dicts, "linear_conv_carry"
        )
