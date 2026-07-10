# SPDX-License-Identifier: Apache-2.0
"""Cached verifier runner for Gemma4 speculative decoding microbenchmarks.

Builds prefill-style attention metadata that reuses an existing request's KV
block table, runs the target model with
``attn_metadata["verifier_return_all_logits"] == True``, and returns per-position
logits for the new tokens.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm import LLM
from vllm.kernels.triton.compute_slot_mapping import compute_slot_mapping
from vllm.lora.mapping import LoRAMapping


def build_verifier_metadata(
    kv_block_manager: Any,
    inf_config: Any,
    slot_idx: int,
    request_id: str,
    prefix_len: int,
    input_ids: torch.Tensor,
    num_layers: int,
) -> dict[str, Any]:
    device = input_ids.device
    bsz, seqlen = input_ids.shape
    total_len = prefix_len + seqlen

    # Grow the block table if the verifier will write past the current end.
    kv_block_manager.ensure_blocks(request_id, total_len)
    kv_block_manager.update_block_table_row(slot_idx, request_id)

    positions = torch.arange(
        prefix_len, total_len, device=device, dtype=torch.long
    ).unsqueeze(0)
    seq_lens = torch.tensor([total_len], device=device, dtype=torch.int32)
    block_table = kv_block_manager.block_table_for_slot(slot_idx).unsqueeze(0)
    query_start_loc = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
    slot_mapping = torch.empty(seqlen, device=device, dtype=torch.long)
    compute_slot_mapping(
        query_start_loc,
        positions.view(-1),
        block_table,
        kv_block_manager.block_size,
        slot_mapping,
        pad_id=-1,
    )
    return {
        "slot_mapping": slot_mapping,
        "seq_lens": seq_lens,
        "seq_lens_cpu": [int(total_len)],
        "max_seq_len_cpu": int(total_len),
        "kv_start_indices": torch.tensor(
            [prefix_len], device=device, dtype=torch.int32
        ),
        "kv_start_indices_cpu": [prefix_len],
        "block_tables": block_table,
        "is_prefill": True,
        "verifier_return_all_logits": True,
        "kv_scale_cache": kv_block_manager.kv_scale_caches,
        "kv_cache_dtype": inf_config.kv_type,
        "k_scale": inf_config.k_scale,
        "v_scale": inf_config.v_scale,
        "config": inf_config,
        "lora_mapping": LoRAMapping.from_ids([None]),
        "linear_attn_carry": [None] * num_layers,
        "linear_conv_carry": [None] * num_layers,
    }


def run_cached_verifier(
    target_llm: LLM,
    request_id: str,
    prefix_len: int,
    input_ids: torch.Tensor,  # (1, K+1)
) -> torch.Tensor:
    engine = target_llm.engine
    kvbm = engine.kv_block_manager
    req = engine.scheduler.get_request(request_id)
    slot_idx = int(req.slot_idx)
    attn_metadata = build_verifier_metadata(
        kvbm,
        engine.inf_config,
        slot_idx,
        request_id,
        prefix_len,
        input_ids,
        len(engine.model.model.layers),
    )
    positions = torch.arange(
        prefix_len,
        prefix_len + input_ids.shape[1],
        device=input_ids.device,
        dtype=torch.long,
    ).unsqueeze(0)
    with torch.inference_mode():
        logits = engine.model(
            input_ids,
            positions,
            kvbm.kv_caches,
            attn_metadata,
        )
    return logits
