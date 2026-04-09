# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from typing import Any

from vllm.engine.backend.lite_single_gpu import LiteSingleGpuBackend
from vllm.engine.decode_executor import DecodeExecutor
from vllm.engine.input_batch_builder import InputBatchBuilder
from vllm.engine.kv_block_manager import KVBlockManager
from vllm.engine.prefill_executor import PrefillExecutor
from vllm.engine.runtime_controller import RuntimeController
from vllm.engine.step_scheduler import StepScheduler


def _parse_service_class_quotas_env(name: str) -> dict[str, int] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    quotas: dict[str, int] = {}
    for item in raw.split(","):
        part = item.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        try:
            quotas[key] = max(0, int(value))
        except ValueError:
            continue
    return quotas or None


def _parse_service_class_weights_env(name: str) -> dict[str, int] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    weights: dict[str, int] = {}
    for item in raw.split(","):
        part = item.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        try:
            weights[key] = max(1, int(value))
        except ValueError:
            continue
    return weights or None


def _parse_service_class_list_env(name: str) -> set[str] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


class LiteRuntimeFactory:
    @classmethod
    def build(cls, engine: Any) -> dict[str, Any]:
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
        )
        prefill_executor = PrefillExecutor(
            model=engine.model,
            input_batch_builder=input_batch_builder,
            kv_caches=engine.kv_caches,
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
            max_decode_streak=int(
                os.environ.get("FASTINFERENCE_MAX_DECODE_STREAK", "4")
            ),
            queue_aging_threshold_s=float(
                os.environ.get("FASTINFERENCE_QUEUE_AGING_THRESHOLD_S", "2.0")
            ),
            max_prefill_deferrals=int(
                os.environ.get("FASTINFERENCE_MAX_PREFILL_DEFERRALS", "2")
            ),
            service_class_weights=_parse_service_class_weights_env(
                "FASTINFERENCE_SERVICE_CLASS_WEIGHTS"
            ),
            admission_service_class_quotas=_parse_service_class_quotas_env(
                "FASTINFERENCE_ADMISSION_SERVICE_CLASS_QUOTAS"
            ),
            decode_service_class_quotas=_parse_service_class_quotas_env(
                "FASTINFERENCE_DECODE_SERVICE_CLASS_QUOTAS"
            ),
            fairness_guardrail_queue_wait_s=float(
                os.environ.get("FASTINFERENCE_FAIRNESS_GUARDRAIL_QUEUE_WAIT_S", "0.0")
            ),
            fairness_guardrail_service_classes=_parse_service_class_list_env(
                "FASTINFERENCE_FAIRNESS_GUARDRAIL_SERVICE_CLASSES"
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
            max_prefix_cache_entries=int(
                os.environ.get("FASTINFERENCE_PREFIX_CACHE_MAX_ENTRIES", "8")
            ),
            preemption_mode=os.environ.get(
                "FASTINFERENCE_PREEMPTION_MODE", "defer_prefill"
            ),
            preemption_min_backlog=int(
                os.environ.get("FASTINFERENCE_PREEMPT_MIN_BACKLOG", "1")
            ),
            preemption_min_decodes=int(
                os.environ.get("FASTINFERENCE_PREEMPT_MIN_DECODES", "1")
            ),
            preemption_max_queue_wait_s=float(
                os.environ.get("FASTINFERENCE_PREEMPT_MAX_QUEUE_WAIT_S", "0.0")
            ),
            preemptible_service_classes=_parse_service_class_list_env(
                "FASTINFERENCE_PREEMPTIBLE_SERVICE_CLASSES"
            ),
        )
        runtime_controller = RuntimeController(
            scheduler=engine.scheduler,
            step_scheduler=step_scheduler,
            observer=engine.observer,
            backend=execution_backend,
            queue_timeout_s=engine._queue_timeout_s,
        )
        return {
            "kv_block_manager": kv_block_manager,
            "input_batch_builder": input_batch_builder,
            "prefill_executor": prefill_executor,
            "decode_executor": decode_executor,
            "step_scheduler": step_scheduler,
            "execution_backend": execution_backend,
            "runtime_controller": runtime_controller,
        }
