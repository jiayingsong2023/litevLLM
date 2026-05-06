# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from vllm.engine.backend.lite_single_gpu import LiteSingleGpuBackend
from vllm.engine.decode_executor import DecodeExecutor
from vllm.engine.input_batch_builder import InputBatchBuilder
from vllm.engine.kv_block_manager import KVBlockManager
from vllm.engine.multimodal_processor import LiteMultiModalProcessor
from vllm.engine.prefill_executor import PrefillExecutor
from vllm.engine.runtime_config import BackendRuntimePolicy, SchedulerRuntimePolicy
from vllm.engine.runtime_controller import RuntimeController
from vllm.engine.step_scheduler import StepScheduler


class LiteRuntimeFactory:
    @classmethod
    def build(cls, engine: Any) -> dict[str, Any]:
        runtime_config = getattr(engine, "runtime_config", None)
        scheduler_policy = getattr(
            runtime_config,
            "scheduler_policy",
            SchedulerRuntimePolicy(),
        )
        backend_policy = getattr(
            runtime_config,
            "backend_policy",
            BackendRuntimePolicy(),
        )
        kv_block_manager = KVBlockManager(
            kv_caches=engine.kv_caches,
            kv_scale_caches=engine.kv_scale_caches,
            num_blocks_per_seq=engine.num_blocks_per_seq,
            block_size=engine.block_size,
        )
        input_batch_builder = InputBatchBuilder(
            device=engine.device,
            max_model_len=engine.max_model_len,
            num_layers=engine.num_layers,
            kv_block_manager=kv_block_manager,
            inf_config=engine.inf_config,
            stack_per_layer_carries=engine._stack_per_layer_carries,
            split_per_layer_carries=engine._split_per_layer_carries,
            sig_caches=getattr(engine, "sig_caches", None),
            sig_temp_buffers=getattr(engine, "_sig_temp_buffers", None),
        )
        multimodal_processor = LiteMultiModalProcessor(
            model=engine.model,
            device=engine.device,
        )
        prefill_executor = PrefillExecutor(
            model=engine.model,
            input_batch_builder=input_batch_builder,
            kv_caches=engine.kv_caches,
            multimodal_processor=multimodal_processor,
        )
        decode_executor = DecodeExecutor(
            model=engine.model,
            input_batch_builder=input_batch_builder,
            kv_caches=engine.kv_caches,
            fast_input_ids=engine._fast_input_ids,
            fast_positions=engine._fast_positions,
            fast_slot_mapping=engine._fast_slot_mapping,
            fast_seq_lens=engine._fast_seq_lens,
            fast_block_tables=engine._fast_block_tables,
        )
        step_scheduler = StepScheduler(
            step_token_budget=engine._step_token_budget,
            decode_priority_enabled=engine._decode_priority_enabled,
            prefill_chunk_size=engine._prefill_chunk_size,
            prefill_reserved_tokens=engine._prefill_reserved_tokens,
            prefill_reserve_backlog=engine._prefill_reserve_backlog,
            prefill_catchup_ratio=engine._prefill_catchup_ratio,
            prefill_microbatch_size=engine._prefill_microbatch_size,
            max_admit_per_step=max(1, min(4, engine.max_active_requests)),
            max_decode_streak=scheduler_policy.max_decode_streak,
            queue_aging_threshold_s=scheduler_policy.queue_aging_threshold_s,
            max_prefill_deferrals=scheduler_policy.max_prefill_deferrals,
            service_class_weights=scheduler_policy.service_class_weights,
            admission_service_class_quotas=(
                scheduler_policy.admission_service_class_quotas
            ),
            decode_service_class_quotas=scheduler_policy.decode_service_class_quotas,
            fairness_guardrail_queue_wait_s=(
                scheduler_policy.fairness_guardrail_queue_wait_s
            ),
            fairness_guardrail_service_classes=(
                scheduler_policy.fairness_guardrail_service_classes
            ),
            max_admit_lora_adapters_per_step=(
                scheduler_policy.max_admit_lora_adapters_per_step
            ),
            max_prefill_lora_adapters_per_batch=(
                scheduler_policy.max_prefill_lora_adapters_per_batch
            ),
            max_decode_lora_adapters_per_batch=(
                scheduler_policy.max_decode_lora_adapters_per_batch
            ),
            lora_fairness_relax_threshold=(
                scheduler_policy.lora_fairness_relax_threshold
            ),
            lora_locality_tighten_threshold=(
                scheduler_policy.lora_locality_tighten_threshold
            ),
            lora_limit_relax_delta=scheduler_policy.lora_limit_relax_delta,
            lora_limit_tighten_delta=scheduler_policy.lora_limit_tighten_delta,
            max_admit_multimodal_per_step=(
                scheduler_policy.max_admit_multimodal_per_step
            ),
            max_prefill_multimodal_requests_per_batch=(
                scheduler_policy.max_prefill_multimodal_requests_per_batch
            ),
            max_decode_multimodal_requests_per_batch=(
                scheduler_policy.max_decode_multimodal_requests_per_batch
            ),
            max_admit_multimodal_lora_per_step=(
                scheduler_policy.max_admit_multimodal_lora_per_step
            ),
            max_prefill_multimodal_lora_requests_per_batch=(
                scheduler_policy.max_prefill_multimodal_lora_requests_per_batch
            ),
            max_decode_multimodal_lora_requests_per_batch=(
                scheduler_policy.max_decode_multimodal_lora_requests_per_batch
            ),
            multimodal_prefix_cache_relax_threshold=(
                scheduler_policy.multimodal_prefix_cache_relax_threshold
            ),
            multimodal_prefix_cache_tighten_threshold=(
                scheduler_policy.multimodal_prefix_cache_tighten_threshold
            ),
            multimodal_prefill_limit_relax_delta=(
                scheduler_policy.multimodal_prefill_limit_relax_delta
            ),
            multimodal_prefill_limit_tighten_delta=(
                scheduler_policy.multimodal_prefill_limit_tighten_delta
            ),
            multimodal_lora_prefill_limit_relax_delta=(
                scheduler_policy.multimodal_lora_prefill_limit_relax_delta
            ),
            multimodal_lora_prefill_limit_tighten_delta=(
                scheduler_policy.multimodal_lora_prefill_limit_tighten_delta
            ),
            multimodal_lora_fairness_relax_threshold=(
                scheduler_policy.multimodal_lora_fairness_relax_threshold
            ),
            multimodal_lora_locality_tighten_threshold=(
                scheduler_policy.multimodal_lora_locality_tighten_threshold
            ),
        )
        execution_backend = LiteSingleGpuBackend(
            scheduler=engine.scheduler,
            observer=engine.observer,
            prefill_executor=prefill_executor,
            decode_executor=decode_executor,
            sampling_driver=engine.sampling_driver,
            output_coordinator=engine.output_pipeline,
            kv_block_manager=kv_block_manager,
            lora_registry=engine.lora_registry,
            max_prefix_cache_entries=backend_policy.max_prefix_cache_entries,
            preemption_mode=backend_policy.preemption_mode,
            preemption_min_backlog=backend_policy.preemption_min_backlog,
            preemption_min_decodes=backend_policy.preemption_min_decodes,
            preemption_max_queue_wait_s=backend_policy.preemption_max_queue_wait_s,
            preemptible_service_classes=backend_policy.preemptible_service_classes,
            preempt_multimodal_prefills=backend_policy.preempt_multimodal_prefills,
            preempt_multimodal_max_queue_wait_s=(
                backend_policy.preempt_multimodal_max_queue_wait_s
            ),
            multimodal_prefix_cache_protect_threshold=(
                backend_policy.multimodal_prefix_cache_protect_threshold
            ),
        )
        runtime_controller = RuntimeController(
            scheduler=engine.scheduler,
            step_scheduler=step_scheduler,
            observer=engine.observer,
            backend=execution_backend,
            queue_timeout_s=engine._queue_timeout_s,
            lora_registry=engine.lora_registry,
        )
        return {
            "kv_block_manager": kv_block_manager,
            "input_batch_builder": input_batch_builder,
            "multimodal_processor": multimodal_processor,
            "prefill_executor": prefill_executor,
            "decode_executor": decode_executor,
            "step_scheduler": step_scheduler,
            "execution_backend": execution_backend,
            "runtime_controller": runtime_controller,
        }
