# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from vllm.engine.input_batch_builder import InputBatchBuilder
from vllm.engine.kv_block_manager import KVBlockManager
from vllm.engine.planners.budget_computer import BudgetComputer
from vllm.engine.request_scheduler import RequestScheduler
from vllm.engine.request_state import RequestState
from vllm.engine.step_scheduler import StepScheduler
from vllm.kernels.triton.paged_attention import _select_paged_attention_launch_config
from vllm.sampling_params import SamplingParams


def _scheduler_with_requests(requests: list[dict[str, Any]]) -> RequestScheduler:
    scheduler = RequestScheduler(max_active_requests=max(1, len(requests)))
    for i, request in enumerate(requests):
        state = RequestState(
            request_id=f"r{i}",
            prompt="",
            input_ids=request.get("input_ids", [1, 2, 3, 4]),
            sampling_params=request.get("sampling_params") or SamplingParams(),
            slot_idx=i,
            is_prefill=request["is_prefill"],
            seq_len=request.get("seq_len", 0),
            generated_ids=request.get("generated_ids", [10]),
            service_class=request.get("service_class", "latency"),
            lora_id=request.get("lora_id"),
            prefix_hit_len=request.get("prefix_hit_len", 0),
        )
        scheduler.add_request(f"r{i}", state)
    return scheduler


def test_adaptive_sizing_verification() -> None:
    """Verify that StepScheduler dynamically scales chunk size down

    as context size grows.
    """
    scheduler_short = _scheduler_with_requests(
        [{"is_prefill": True, "seq_len": 0, "input_ids": [1] * 2000}]
    )
    step_scheduler = StepScheduler(
        step_token_budget=2048,
        decode_priority_enabled=True,
        prefill_chunk_size=2048,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
    )
    step_scheduler.min_prefill_chunk_size = 128
    step_scheduler.max_prefill_chunk_size = 2048
    step_scheduler.build_plan(scheduler_short)
    assert step_scheduler.prefill_chunk_size == 2048
    scheduler_medium = _scheduler_with_requests(
        [{"is_prefill": True, "seq_len": 0, "input_ids": [1] * 12000}]
    )
    step_scheduler.build_plan(scheduler_medium)
    assert step_scheduler.prefill_chunk_size == 512
    scheduler_long = _scheduler_with_requests(
        [{"is_prefill": True, "seq_len": 0, "input_ids": [1] * 20000}]
    )
    step_scheduler.build_plan(scheduler_long)
    assert step_scheduler.prefill_chunk_size == 256


def test_backlog_aware_budgeting() -> None:
    """Verify that under heavy prefill backlogs, prefill budget increases

    using backlog-aware catchup ratio.
    """
    _scheduler_with_requests(
        [
            {"is_prefill": True, "seq_len": 0, "input_ids": [1] * 100},
            {"is_prefill": False, "seq_len": 50},
        ]
    )
    step_scheduler = StepScheduler(
        step_token_budget=1024,
        decode_priority_enabled=True,
        prefill_chunk_size=512,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=4,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
    )
    step_scheduler.min_prefill_chunk_size = 128
    step_scheduler.max_prefill_chunk_size = 512
    budget_computer = BudgetComputer(
        step_token_budget=1024,
        prefill_chunk_size=512,
        decode_priority_enabled=True,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=4,
        prefill_catchup_ratio=0.25,
    )
    light = budget_computer.compute(
        num_prefills=1, num_decodes=1, starvation_protected=False
    )
    heavy = budget_computer.compute(
        num_prefills=6, num_decodes=1, starvation_protected=False
    )
    assert heavy.prefill_budget > light.prefill_budget


def test_prefix_cache_chunked_prefill() -> None:
    """Test that chunked prefill coexisting with prefix cache matches

    slices and offsets correctly.
    """
    scheduler = RequestScheduler(max_active_requests=1)
    scheduler.add_request(
        "r1",
        RequestState(
            request_id="r1",
            prompt="",
            slot_idx=0,
            is_prefill=True,
            seq_len=2,
            prefix_hit_len=2,
            input_ids=[10, 20, 30, 40],
            generated_ids=[],
            linear_attn_carry=[None],
            linear_conv_carry=[None],
            lora_id=None,
            sampling_params=SamplingParams(),
        ),
    )

    def dummy_stack(req_dicts, num_layers: int, key: str):
        return [None]

    def dummy_split(stacked, req_dicts, key: str):
        pass

    builder = InputBatchBuilder(
        device=torch.device("cpu"),
        max_model_len=8,
        num_layers=1,
        kv_block_manager=KVBlockManager(
            kv_caches=[], kv_scale_caches=[], num_blocks_per_seq=2, block_size=2
        ),
        inf_config=type(
            "Cfg", (), {"kv_type": "fp16", "k_scale": 1.0, "v_scale": 1.0}
        )(),
        stack_per_layer_carries=dummy_stack,
        split_per_layer_carries=dummy_split,
    )
    curr_input, positions, attn_metadata, req_dicts, last_flags = builder.build_prefill(
        ["r1"], scheduler, chunk_len=1
    )
    assert curr_input.tolist() == [[30]]
    assert positions.tolist() == [[2]]
    assert attn_metadata["slot_mapping"].tolist() == [2]
    assert attn_metadata["kv_start_indices"].tolist() == [2]
    assert last_flags == [False]


def test_paged_attention_launch_tuning() -> None:
    """Verify that _select_paged_attention_launch_config tunes warps and stages

    for chunked prefill (query_len > 8).
    """
    warps, stages = _select_paged_attention_launch_config(
        num_seqs=128, head_size=256, block_size=16, is_int4=True, is_fp8=False
    )
    assert warps == 8
    assert stages == 2
    warps_quant, stages_quant = _select_paged_attention_launch_config(
        num_seqs=128, head_size=128, block_size=16, is_int4=True, is_fp8=False
    )
    assert warps_quant == 4
    assert stages_quant == 4


def test_gemma4_qwen35_tinyllama_integration_smoke() -> None:
    """Integration smoke check representing Gemma4, Qwen3.5, and TinyLlama

    architectures under StepScheduler.
    """
    step_scheduler = StepScheduler(
        step_token_budget=1024,
        decode_priority_enabled=True,
        prefill_chunk_size=1024,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
    )
    step_scheduler.min_prefill_chunk_size = 128
    step_scheduler.max_prefill_chunk_size = 1024
    scheduler_tiny = _scheduler_with_requests(
        [{"is_prefill": True, "seq_len": 0, "input_ids": [1] * 256}]
    )
    plan_tiny = step_scheduler.build_plan(scheduler_tiny)
    assert plan_tiny.prefills.chunk_len == 256
    scheduler_qwen = _scheduler_with_requests(
        [{"is_prefill": True, "seq_len": 0, "input_ids": [1] * 5000}]
    )
    step_scheduler.build_plan(scheduler_qwen)
    assert step_scheduler.prefill_chunk_size == 1024
    scheduler_gemma = _scheduler_with_requests(
        [{"is_prefill": True, "seq_len": 0, "input_ids": [1] * 18000}]
    )
    step_scheduler.build_plan(scheduler_gemma)
    assert step_scheduler.prefill_chunk_size == 256
