# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
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
        inf_config=type(
            "Cfg", (), {"kv_type": "fp16", "k_scale": 1.0, "v_scale": 1.0}
        )(),
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
    assert attn_metadata["lora_mapping"] == [None]
    assert attn_metadata["lora_adapter_count"] == 0
    assert attn_metadata["mixed_lora_batch"] is False
    assert req_dicts[0]["slot_idx"] == 0
    assert last_flags == [False]


def test_input_batch_builder_rejects_zero_token_prefill() -> None:
    scheduler = _scheduler_with_request()
    scheduler.get_request("r1")["seq_len"] = 4
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
        inf_config=type(
            "Cfg", (), {"kv_type": "fp16", "k_scale": 1.0, "v_scale": 1.0}
        )(),
        stack_per_layer_carries=_stack,
        split_per_layer_carries=_split,
    )

    with pytest.raises(ValueError, match="zero-token prefill"):
        builder.build_prefill(["r1"], scheduler, 1)


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
        inf_config=type(
            "Cfg", (), {"kv_type": "fp16", "k_scale": 1.0, "v_scale": 1.0}
        )(),
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
    assert attn_metadata["lora_mapping"] == [None]
    assert attn_metadata["lora_adapter_count"] == 0
    assert attn_metadata["mixed_lora_batch"] is False
    assert req_dicts[0]["seq_len"] == 3


def test_input_batch_builder_reuses_decode_batch_tensors() -> None:
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
        inf_config=type(
            "Cfg", (), {"kv_type": "fp16", "k_scale": 1.0, "v_scale": 1.0}
        )(),
        stack_per_layer_carries=_stack,
        split_per_layer_carries=_split,
    )

    curr_input, positions, attn_metadata, _ = builder.build_decode_batch(
        ["r1"], scheduler
    )
    ptrs = (
        curr_input.data_ptr(),
        positions.data_ptr(),
        attn_metadata["slot_mapping"].data_ptr(),
        attn_metadata["seq_lens"].data_ptr(),
    )

    req["generated_ids"] = [43]
    req["seq_len"] = 4
    curr_input2, positions2, attn_metadata2, _ = builder.build_decode_batch(
        ["r1"], scheduler
    )

    assert (
        curr_input2.data_ptr(),
        positions2.data_ptr(),
        attn_metadata2["slot_mapping"].data_ptr(),
        attn_metadata2["seq_lens"].data_ptr(),
    ) == ptrs
    assert curr_input2.tolist() == [[43]]
    assert positions2.tolist() == [[4]]
    assert attn_metadata2["slot_mapping"].tolist() == [4]
    assert attn_metadata2["seq_lens"].tolist() == [5]


def test_input_batch_builder_marks_mixed_lora_decode_batch() -> None:
    scheduler = RequestScheduler(max_active_requests=2)
    scheduler.add_request(
        "r1",
        {
            "slot_idx": 0,
            "is_prefill": False,
            "seq_len": 3,
            "input_ids": [11, 12, 13, 14],
            "generated_ids": [42],
            "linear_attn_carry": [None],
            "linear_conv_carry": [None],
            "lora_id": "adapter-a",
        },
    )
    scheduler.add_request(
        "r2",
        {
            "slot_idx": 1,
            "is_prefill": False,
            "seq_len": 3,
            "input_ids": [11, 12, 13, 14],
            "generated_ids": [43],
            "linear_attn_carry": [None],
            "linear_conv_carry": [None],
            "lora_id": "adapter-b",
        },
    )
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
        inf_config=type(
            "Cfg", (), {"kv_type": "fp16", "k_scale": 1.0, "v_scale": 1.0}
        )(),
        stack_per_layer_carries=_stack,
        split_per_layer_carries=_split,
    )

    _, _, attn_metadata, _ = builder.build_decode_batch(["r1", "r2"], scheduler)

    assert attn_metadata["lora_mapping"] == ["adapter-a", "adapter-b"]
    assert attn_metadata["lora_adapter_count"] == 2
    assert attn_metadata["mixed_lora_batch"] is True


def test_input_batch_builder_tracks_multimodal_lora_prefill_contract() -> None:
    scheduler = RequestScheduler(max_active_requests=2)
    scheduler.add_request(
        "r1",
        {
            "slot_idx": 0,
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [11, 12, 13],
            "generated_ids": [],
            "linear_attn_carry": [None],
            "linear_conv_carry": [None],
            "lora_id": "adapter-a",
            "multi_modal_data": {"image": [{"image": "file:///tmp/cat.png"}]},
            "is_multimodal": True,
            "is_multimodal_lora": True,
        },
    )
    scheduler.add_request(
        "r2",
        {
            "slot_idx": 1,
            "is_prefill": True,
            "seq_len": 0,
            "input_ids": [21, 22, 23],
            "generated_ids": [],
            "linear_attn_carry": [None],
            "linear_conv_carry": [None],
            "lora_id": None,
            "multi_modal_data": None,
            "is_multimodal": False,
            "is_multimodal_lora": False,
        },
    )
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
        inf_config=type(
            "Cfg", (), {"kv_type": "fp16", "k_scale": 1.0, "v_scale": 1.0}
        )(),
        stack_per_layer_carries=_stack,
        split_per_layer_carries=_split,
    )

    _, _, attn_metadata, _, _ = builder.build_prefill(["r1", "r2"], scheduler, 2)

    assert attn_metadata["lora_mapping"] == ["adapter-a", None]
    assert attn_metadata["lora_adapter_count"] == 1
    assert attn_metadata["mixed_lora_batch"] is False
    assert attn_metadata["multimodal_request_count"] == 1
    assert attn_metadata["has_multimodal_requests"] is True
    assert attn_metadata["mixed_multimodal_batch"] is True
    assert attn_metadata["multimodal_lora_request_count"] == 1
    assert attn_metadata["has_multimodal_lora_requests"] is True
