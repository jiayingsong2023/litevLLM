# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.engine.backend.lite_single_gpu import LiteSingleGpuBackend
from vllm.engine.decode_executor import DecodeExecutor
from vllm.engine.input_batch_builder import InputBatchBuilder
from vllm.engine.kv_block_manager import KVBlockManager
from vllm.engine.multimodal_processor import LiteMultiModalProcessor
from vllm.engine.prefill_executor import PrefillExecutor
from vllm.engine.runtime_controller import RuntimeController
from vllm.engine.runtime_policy import BackendRuntimePolicy, SchedulerRuntimePolicy
from vllm.engine.step_scheduler import StepScheduler


@dataclass(frozen=True)
class RuntimeAssemblyContext:
    kv_caches: Any
    kv_scale_caches: Any
    num_blocks_per_seq: int
    block_size: int
    device: Any
    max_model_len: int
    num_layers: int
    inf_config: Any
    stack_per_layer_carries: Any
    split_per_layer_carries: Any
    model: Any
    fast_input_ids: Any
    fast_positions: Any
    fast_slot_mapping: Any
    fast_seq_lens: Any
    fast_block_tables: Any
    step_token_budget: int
    decode_priority_enabled: bool
    prefill_chunk_size: int
    prefill_reserved_tokens: int
    prefill_reserve_backlog: int
    prefill_catchup_ratio: float
    prefill_microbatch_size: int
    min_prefill_chunk_size: int
    max_prefill_chunk_size: int
    prefill_sla_ttft_ms: float
    max_active_requests: int
    scheduler_policy: SchedulerRuntimePolicy
    backend_policy: BackendRuntimePolicy
    scheduler: Any
    observer: Any
    lora_registry: Any
    sampling_driver: Any
    output_pipeline: Any
    queue_timeout_s: float
    sig_caches: Any | None = None
    sig_temp_buffers: Any | None = None


class LiteRuntimeFactory:
    @classmethod
    def build(cls, context: RuntimeAssemblyContext) -> dict[str, Any]:
        scheduler_policy = context.scheduler_policy
        backend_policy = context.backend_policy
        kv_block_manager = KVBlockManager(
            kv_caches=context.kv_caches,
            kv_scale_caches=context.kv_scale_caches,
            num_blocks_per_seq=context.num_blocks_per_seq,
            block_size=context.block_size,
        )
        input_batch_builder = InputBatchBuilder(
            device=context.device,
            max_model_len=context.max_model_len,
            num_layers=context.num_layers,
            kv_block_manager=kv_block_manager,
            inf_config=context.inf_config,
            stack_per_layer_carries=context.stack_per_layer_carries,
            split_per_layer_carries=context.split_per_layer_carries,
            sig_caches=context.sig_caches,
            sig_temp_buffers=context.sig_temp_buffers,
        )
        multimodal_processor = LiteMultiModalProcessor(
            model=context.model,
            device=context.device,
        )
        prefill_executor = PrefillExecutor(
            model=context.model,
            input_batch_builder=input_batch_builder,
            kv_caches=context.kv_caches,
            multimodal_processor=multimodal_processor,
        )
        decode_executor = DecodeExecutor(
            model=context.model,
            input_batch_builder=input_batch_builder,
            kv_caches=context.kv_caches,
            fast_input_ids=context.fast_input_ids,
            fast_positions=context.fast_positions,
            fast_slot_mapping=context.fast_slot_mapping,
            fast_seq_lens=context.fast_seq_lens,
            fast_block_tables=context.fast_block_tables,
        )
        step_scheduler = StepScheduler(
            step_token_budget=context.step_token_budget,
            decode_priority_enabled=context.decode_priority_enabled,
            prefill_chunk_size=context.prefill_chunk_size,
            prefill_reserved_tokens=context.prefill_reserved_tokens,
            prefill_reserve_backlog=context.prefill_reserve_backlog,
            prefill_catchup_ratio=context.prefill_catchup_ratio,
            prefill_microbatch_size=context.prefill_microbatch_size,
            min_prefill_chunk_size=context.min_prefill_chunk_size,
            max_prefill_chunk_size=context.max_prefill_chunk_size,
            prefill_sla_ttft_ms=context.prefill_sla_ttft_ms,
            max_admit_per_step=max(1, min(4, context.max_active_requests)),
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
            scheduler=context.scheduler,
            observer=context.observer,
            prefill_executor=prefill_executor,
            decode_executor=decode_executor,
            sampling_driver=context.sampling_driver,
            output_coordinator=context.output_pipeline,
            kv_block_manager=kv_block_manager,
            lora_registry=context.lora_registry,
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
            gpu_greedy_sampling=backend_policy.gpu_greedy_sampling,
            gpu_greedy_max_tokens_only=backend_policy.gpu_greedy_max_tokens_only,
            gpu_greedy_bypass_cpu_policies=(
                backend_policy.gpu_greedy_bypass_cpu_policies
            ),
            gpu_greedy_ignore_eos=backend_policy.gpu_greedy_ignore_eos,
        )
        runtime_controller = RuntimeController(
            scheduler=context.scheduler,
            step_scheduler=step_scheduler,
            observer=context.observer,
            backend=execution_backend,
            queue_timeout_s=context.queue_timeout_s,
            lora_registry=context.lora_registry,
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
