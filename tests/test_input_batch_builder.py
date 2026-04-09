# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm.engine.input_batch_builder import InputBatchBuilder
from vllm.engine.kv_block_manager import KVBlockManager
from vllm.engine.request_scheduler import RequestScheduler


def _stack(req_dicts, num_layers: int, key: str):
    out = []
    for li in range(num_layers):
        parts = [r[key][li] for r in req_dicts]
        out.append(None if all(p is None for p in parts) else torch.cat(parts, dim=0))
    return out


def _split(stacked, req_dicts, key: str):
    for li, t in enumerate(stacked):
        for i, r in enumerate(req_dicts):
            r[key][li] = None if t is None else t[i : i + 1].contiguous()


def _scheduler_with_request() -> RequestScheduler:
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.add_request(
        "r1",
        {
            "slot_idx": 0,
            "is_prefill": True,
            "seq_len": 1,
            "input_ids": [11, 12, 13, 14],
            "generated_ids": [99],
            "linear_attn_carry": [None],
            "linear_conv_carry": [None],
            "lora_id": None,
        },
    )
    return scheduler


def test_input_batch_builder_build_prefill() -> None:
    scheduler = _scheduler_with_request()
    builder = InputBatchBuilder(
        device=torch.device("cpu"),
        max_model_len=8,
        num_layers=1,
        kv_block_manager=KVBlockManager(
            kv_caches=[],
            kv_scale_caches=[],
            num_blocks_per_seq=2,
            block_size=2,
        ),
        inf_config=type("Cfg", (), {"kv_type": "fp16", "k_scale": 1.0, "v_scale": 1.0})(),
        stack_per_layer_carries=_stack,
        split_per_layer_carries=_split,
    )

    curr_input, positions, attn_metadata, req_dicts, last_flags = builder.build_prefill(
        ["r1"], scheduler, 2
    )

    assert curr_input.tolist() == [[12, 13]]
    assert positions.tolist() == [[1, 2]]
    assert attn_metadata["slot_mapping"].tolist() == [1, 2]
    assert attn_metadata["block_tables"].tolist() == [[0, 1]]
    assert attn_metadata["seq_lens"].tolist() == [3]
    assert req_dicts[0]["slot_idx"] == 0
    assert last_flags == [False]


def test_input_batch_builder_build_decode_batch() -> None:
    scheduler = _scheduler_with_request()
    req = scheduler.get_request("r1")
    req["is_prefill"] = False
    req["generated_ids"] = [42]
    req["seq_len"] = 3

    builder = InputBatchBuilder(
        device=torch.device("cpu"),
        max_model_len=8,
        num_layers=1,
        kv_block_manager=KVBlockManager(
            kv_caches=[],
            kv_scale_caches=[],
            num_blocks_per_seq=2,
            block_size=2,
        ),
        inf_config=type("Cfg", (), {"kv_type": "fp16", "k_scale": 1.0, "v_scale": 1.0})(),
        stack_per_layer_carries=_stack,
        split_per_layer_carries=_split,
    )

    curr_input, positions, attn_metadata, req_dicts = builder.build_decode_batch(
        ["r1"], scheduler
    )

    assert curr_input.tolist() == [[42]]
    assert positions.tolist() == [[3]]
    assert attn_metadata["slot_mapping"].tolist() == [3]
    assert attn_metadata["block_tables"].tolist() == [[0, 1]]
    assert attn_metadata["seq_lens"].tolist() == [4]
    assert req_dicts[0]["seq_len"] == 3
