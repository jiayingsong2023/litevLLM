# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.engine.runtime_config import BackendRuntimePolicy, SchedulerRuntimePolicy
from vllm.engine.runtime_factory import LiteRuntimeFactory, RuntimeAssemblyContext


def test_lite_runtime_factory_builds_expected_components() -> None:
    context = RuntimeAssemblyContext(
        kv_caches=[],
        kv_scale_caches=[],
        num_blocks_per_seq=2,
        block_size=16,
        device="cuda:0",
        max_model_len=128,
        num_layers=4,
        inf_config=object(),
        stack_per_layer_carries=lambda *args, **kwargs: None,
        split_per_layer_carries=lambda *args, **kwargs: None,
        model=object(),
        fast_input_ids=object(),
        fast_positions=object(),
        fast_slot_mapping=object(),
        fast_seq_lens=object(),
        fast_block_tables=object(),
        step_token_budget=16,
        decode_priority_enabled=True,
        prefill_chunk_size=8,
        prefill_reserved_tokens=2,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
        prefill_microbatch_size=2,
        min_prefill_chunk_size=4,
        max_prefill_chunk_size=None,
        prefill_sla_ttft_ms=1500.0,
        max_active_requests=4,
        scheduler_policy=SchedulerRuntimePolicy(max_decode_streak=9),
        backend_policy=BackendRuntimePolicy(
            max_prefix_cache_entries=11,
            preemption_mode="off",
            preemption_min_backlog=3,
            preemption_min_decodes=2,
            preemption_max_queue_wait_s=1.5,
            preemptible_service_classes={"throughput"},
            preempt_multimodal_prefills=True,
            preempt_multimodal_max_queue_wait_s=2.0,
            multimodal_prefix_cache_protect_threshold=0.6,
            gpu_greedy_sampling=True,
            gpu_greedy_max_tokens_only=True,
            gpu_greedy_bypass_cpu_policies=True,
            gpu_greedy_ignore_eos=True,
        ),
        scheduler=object(),
        observer=object(),
        lora_registry=None,
        sampling_driver=None,
        output_pipeline=None,
        queue_timeout_s=30.0,
    )

    runtime_components = LiteRuntimeFactory.build(context)

    assert set(runtime_components) == {
        "kv_block_manager",
        "input_batch_builder",
        "multimodal_processor",
        "prefill_executor",
        "decode_executor",
        "step_scheduler",
        "execution_backend",
        "runtime_controller",
    }

    step_scheduler = runtime_components["step_scheduler"]
    assert step_scheduler.max_decode_streak == 9
    assert step_scheduler.min_prefill_chunk_size == 4
    assert step_scheduler.max_prefill_chunk_size == 8
    assert step_scheduler.prefill_sla_ttft_ms == 1500.0

    backend = runtime_components["execution_backend"]
    stats = backend.stats()
    assert stats["prefix_cache"]["capacity"] == 11
    assert stats["preemption_mode"] == "off"
    assert stats["preemption_min_backlog"] == 3
    assert stats["preemption_min_decodes"] == 2
    assert stats["preemption_max_queue_wait_s"] == 1.5
    assert stats["preemptible_service_classes"] == ["throughput"]
    assert stats["preempt_multimodal_prefills"] is True
    assert stats["preempt_multimodal_max_queue_wait_s"] == 2.0
    assert stats["multimodal_prefix_cache_protect_threshold"] == 0.6
    assert stats["gpu_greedy_sampling"] is True
    assert stats["gpu_greedy_max_tokens_only"] is True
    assert stats["gpu_greedy_bypass_cpu_policies"] is True
    assert stats["gpu_greedy_ignore_eos"] is True
