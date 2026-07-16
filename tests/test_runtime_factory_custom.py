from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.engine.custom_runtime_components import CustomRuntimeComponents
from vllm.engine.multimodal_processor import NullMultiModalProcessor
from vllm.engine.runtime_factory import LiteRuntimeFactory, RuntimeAssemblyContext


class _KV:
    block_size = 16
    num_blocks_per_seq = 8
    num_layers = 1

    def ensure_blocks_for_requests(
        self,
        request_ids: list[str],
        token_counts: list[int],
    ) -> None:
        del request_ids, token_counts

    def free_request_blocks(self, request_id: str) -> None:
        del request_id


def _policy():
    return SimpleNamespace(
        max_admit_lora_adapters_per_step=0,
        max_prefill_lora_adapters_per_batch=0,
        max_decode_lora_adapters_per_batch=0,
        lora_fairness_relax_threshold=0.0,
        lora_locality_tighten_threshold=0.0,
        lora_limit_relax_delta=1,
        lora_limit_tighten_delta=1,
        max_admit_multimodal_per_step=0,
        max_prefill_multimodal_requests_per_batch=0,
        max_decode_multimodal_requests_per_batch=0,
        max_admit_multimodal_lora_per_step=0,
        max_prefill_multimodal_lora_requests_per_batch=0,
        max_decode_multimodal_lora_requests_per_batch=0,
        multimodal_prefix_cache_relax_threshold=0.0,
        multimodal_prefix_cache_tighten_threshold=0.0,
        multimodal_prefill_limit_relax_delta=1,
        multimodal_prefill_limit_tighten_delta=1,
        multimodal_lora_prefill_limit_relax_delta=1,
        multimodal_lora_prefill_limit_tighten_delta=1,
        multimodal_lora_fairness_relax_threshold=0.0,
        multimodal_lora_locality_tighten_threshold=0.0,
        max_decode_streak=1,
        queue_aging_threshold_s=0.0,
        max_prefill_deferrals=1,
        service_class_weights={},
        admission_service_class_quotas={},
        decode_service_class_quotas={},
        fairness_guardrail_queue_wait_s=0.0,
        fairness_guardrail_service_classes=set(),
    )


def _backend_policy():
    return SimpleNamespace(
        max_prefix_cache_entries=0,
        preemption_mode="off",
        preemption_min_backlog=1,
        preemption_min_decodes=1,
        preemption_max_queue_wait_s=0.0,
        preemptible_service_classes=set(),
        preempt_multimodal_prefills=False,
        preempt_multimodal_max_queue_wait_s=0.0,
        multimodal_prefix_cache_protect_threshold=0.0,
        gpu_greedy_sampling=False,
        gpu_greedy_max_tokens_only=False,
        gpu_greedy_bypass_cpu_policies=False,
        gpu_greedy_ignore_eos=False,
    )


def _context(**overrides):
    values = dict(
        block_allocator=None,
        kv_caches=[],
        kv_scale_caches=[],
        num_blocks_per_seq=0,
        block_size=16,
        device=torch.device("cpu"),
        max_model_len=128,
        num_layers=1,
        inf_config=None,
        stack_per_layer_carries=None,
        split_per_layer_carries=None,
        model=SimpleNamespace(),
        fast_input_ids=torch.empty((1, 1), dtype=torch.long),
        fast_positions=torch.empty((1, 1), dtype=torch.long),
        fast_slot_mapping=torch.empty((1,), dtype=torch.long),
        fast_seq_lens=torch.empty((1,), dtype=torch.int32),
        step_token_budget=1,
        decode_priority_enabled=False,
        prefill_chunk_size=1,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=1,
        prefill_catchup_ratio=0.0,
        prefill_microbatch_size=1,
        min_prefill_chunk_size=1,
        max_prefill_chunk_size=1,
        prefill_sla_ttft_ms=0.0,
        max_active_requests=1,
        scheduler_policy=_policy(),
        backend_policy=_backend_policy(),
        scheduler=SimpleNamespace(),
        observer=SimpleNamespace(),
        lora_registry=None,
        sampling_driver=None,
        output_pipeline=None,
        queue_timeout_s=0.0,
        custom_runtime_components=None,
    )
    values.update(overrides)
    return RuntimeAssemblyContext(**values)


def test_runtime_factory_uses_custom_components_with_null_multimodal() -> None:
    prefill = object()
    decode = object()
    components = CustomRuntimeComponents(
        prefill_executor=prefill,
        decode_executor=decode,
        kv_block_manager=_KV(),
    )

    components_out = LiteRuntimeFactory.build(
        _context(custom_runtime_components=components)
    )

    assert components_out.prefill_executor is prefill
    assert components_out.decode_executor is decode
    assert components_out.input_batch_builder is None
    assert isinstance(components_out.multimodal_processor, NullMultiModalProcessor)


def test_runtime_factory_rejects_standard_runtime_without_block_allocator() -> None:
    with pytest.raises(RuntimeError, match="block allocator"):
        LiteRuntimeFactory.build(_context(custom_runtime_components=None))
