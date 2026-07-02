# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm.engine.block_allocator import BlockAllocator
from vllm.engine.runtime_factory import LiteRuntimeFactory, RuntimeAssemblyContext


class LiteRuntimeAssembler:
    """Assembles runtime components from a prepared initialization context."""

    @classmethod
    def assemble(
        cls,
        *,
        block_allocator: BlockAllocator,
        kv_caches: list[torch.Tensor],
        kv_scale_caches: list[torch.Tensor],
        num_blocks_per_seq: int,
        block_size: int,
        device: torch.device,
        max_model_len: int,
        num_layers: int,
        inf_config: Any,
        stack_per_layer_carries: Any,
        split_per_layer_carries: Any,
        model: nn.Module,
        fast_input_ids: torch.Tensor,
        fast_positions: torch.Tensor,
        fast_slot_mapping: torch.Tensor,
        fast_seq_lens: torch.Tensor,
        fast_block_tables: torch.Tensor,
        step_token_budget: int,
        decode_priority_enabled: bool,
        prefill_chunk_size: int,
        prefill_reserved_tokens: int,
        prefill_reserve_backlog: int,
        prefill_catchup_ratio: float,
        prefill_microbatch_size: int,
        min_prefill_chunk_size: int,
        max_prefill_chunk_size: int | None,
        prefill_sla_ttft_ms: float,
        max_active_requests: int,
        scheduler_policy: Any,
        backend_policy: Any,
        scheduler: Any,
        observer: Any,
        lora_registry: Any,
        queue_timeout_s: float,
    ) -> dict[str, Any]:
        runtime_context = RuntimeAssemblyContext(
            block_allocator=block_allocator,
            kv_caches=kv_caches,
            kv_scale_caches=kv_scale_caches,
            num_blocks_per_seq=num_blocks_per_seq,
            block_size=block_size,
            device=device,
            max_model_len=max_model_len,
            num_layers=num_layers,
            inf_config=inf_config,
            stack_per_layer_carries=stack_per_layer_carries,
            split_per_layer_carries=split_per_layer_carries,
            model=model,
            fast_input_ids=fast_input_ids,
            fast_positions=fast_positions,
            fast_slot_mapping=fast_slot_mapping,
            fast_seq_lens=fast_seq_lens,
            fast_block_tables=fast_block_tables,
            step_token_budget=step_token_budget,
            decode_priority_enabled=decode_priority_enabled,
            prefill_chunk_size=prefill_chunk_size,
            prefill_reserved_tokens=prefill_reserved_tokens,
            prefill_reserve_backlog=prefill_reserve_backlog,
            prefill_catchup_ratio=prefill_catchup_ratio,
            prefill_microbatch_size=prefill_microbatch_size,
            min_prefill_chunk_size=min_prefill_chunk_size,
            max_prefill_chunk_size=max_prefill_chunk_size,
            prefill_sla_ttft_ms=prefill_sla_ttft_ms,
            max_active_requests=max_active_requests,
            scheduler_policy=scheduler_policy,
            backend_policy=backend_policy,
            scheduler=scheduler,
            observer=observer,
            lora_registry=lora_registry,
            sampling_driver=None,
            output_pipeline=None,
            queue_timeout_s=queue_timeout_s,
        )
        return LiteRuntimeFactory.build(runtime_context)
