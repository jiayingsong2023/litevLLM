# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from vllm.engine.runtime_config import BackendRuntimePolicy, SchedulerRuntimePolicy
from vllm.engine.runtime_factory import LiteRuntimeFactory


def test_lite_runtime_factory_builds_expected_components() -> None:
    engine = SimpleNamespace(
        kv_caches=[],
        kv_scale_caches=[],
        num_blocks_per_seq=2,
        block_size=16,
        device="cuda:0",
        max_model_len=128,
        num_layers=4,
        inf_config=object(),
        _stack_per_layer_carries=lambda *args, **kwargs: None,
        _split_per_layer_carries=lambda *args, **kwargs: None,
        model=object(),
        _fast_input_ids=object(),
        _fast_positions=object(),
        _fast_slot_mapping=object(),
        _fast_seq_lens=object(),
        _fast_block_tables=object(),
        _step_token_budget=16,
        _decode_priority_enabled=True,
        _prefill_chunk_size=8,
        _prefill_reserved_tokens=2,
        _prefill_reserve_backlog=2,
        _prefill_catchup_ratio=0.25,
        _prefill_microbatch_size=2,
        max_active_requests=4,
        runtime_config=SimpleNamespace(
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
            ),
        ),
        scheduler=object(),
        observer=object(),
        lora_registry=None,
        sampling_driver=None,
        output_pipeline=None,
        _queue_timeout_s=30.0,
    )

    runtime_components = LiteRuntimeFactory.build(engine)

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
