# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.engine.request_state import RequestState
from vllm.logger import init_logger

logger = init_logger(__name__)


class _BoundedList(list[Any]):
    def __init__(self, limit: int, values: list[Any] | None = None) -> None:
        self._limit = max(1, int(limit))
        super().__init__((values or [])[-self._limit :])

    def append(self, value: Any) -> None:
        if len(self) >= self._limit:
            del self[0]
        super().append(value)


class RuntimeObserver:
    def on_request_added(self, request_id: str, request: RequestState) -> None:
        pass

    def on_request_admitted(
        self,
        request_id: str,
        queue_wait_s: float,
        service_class: str | None = None,
    ) -> None:
        pass

    def on_prefix_cache_event(
        self,
        request_id: str,
        *,
        hit: bool,
        exact: bool,
        prefix_len: int,
        saved_prefill_tokens: int,
        is_multimodal: bool = False,
    ) -> None:
        pass

    def on_preemption_event(
        self,
        *,
        preempted_prefill_requests: int,
        queued_backlog: int,
        multimodal_prefill_requests: int = 0,
    ) -> None:
        pass

    def on_multimodal_preemption_guard(
        self,
        *,
        protected_prefill_requests: int,
        prefix_cache_hit_rate: float,
    ) -> None:
        pass

    def on_request_rejected(self, request_id: str, reason: str) -> None:
        pass

    def on_step_started(self, plan: Any) -> None:
        pass

    def on_prefill_executed(self, plan: Any, num_outputs: int) -> None:
        pass

    def on_decode_executed(self, plan: Any, num_outputs: int) -> None:
        pass

    def on_request_finished(self, request_id: str, reason: str) -> None:
        pass

    def on_first_token(self, request_id: str, ttft_s: float) -> None:
        pass

    def on_request_aborted(self, request_id: str) -> None:
        pass

    def on_background_error(self, exc: BaseException, request_ids: list[str]) -> None:
        pass

    def on_model_surface_resolved(
        self,
        *,
        event_name: str,
        model_name: str,
        model_type: str,
        status: str,
        reason: str,
    ) -> None:
        pass

    def on_deepseek_event(self, event: str, **payload: Any) -> None:
        pass

    def stats(self) -> dict[str, Any]:
        return {}

    def reset_stats(self) -> None:
        return None


class NullRuntimeObserver(RuntimeObserver):
    pass


@dataclass
class InMemoryRuntimeObserver(RuntimeObserver):
    history_limit: int = 4096
    added: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    rejection_reason_counts: dict[str, int] = field(default_factory=dict)
    admitted: list[tuple[str, float, str]] = field(default_factory=list)
    prefix_cache_events: list[tuple[str, bool, bool, int, int]] = field(
        default_factory=list
    )
    prefix_cache_hits: int = 0
    prefix_cache_misses: int = 0
    prefix_cache_exact_hits: int = 0
    prefix_cache_partial_hits: int = 0
    prefix_cache_saved_prefill_tokens: int = 0
    multimodal_prefix_cache_hits: int = 0
    multimodal_prefix_cache_misses: int = 0
    multimodal_prefix_cache_saved_prefill_tokens: int = 0
    preemption_events: list[tuple[int, int]] = field(default_factory=list)
    preempted_steps: int = 0
    preempted_prefill_requests: int = 0
    preempted_multimodal_prefill_requests: int = 0
    protected_multimodal_prefix_steps: int = 0
    protected_multimodal_prefix_prefill_requests: int = 0
    fairness_admitted_service_classes: dict[str, int] = field(default_factory=dict)
    fairness_prefill_service_classes: dict[str, int] = field(default_factory=dict)
    fairness_decode_service_classes: dict[str, int] = field(default_factory=dict)
    fairness_backlog_service_classes: dict[str, int] = field(default_factory=dict)
    lora_admitted_adapters: dict[str, int] = field(default_factory=dict)
    lora_prefill_adapters: dict[str, int] = field(default_factory=dict)
    lora_decode_adapters: dict[str, int] = field(default_factory=dict)
    lora_backlog_adapters: dict[str, int] = field(default_factory=dict)
    lora_admit_relaxed_steps: int = 0
    lora_admit_tightened_steps: int = 0
    lora_prefill_relaxed_steps: int = 0
    lora_prefill_tightened_steps: int = 0
    lora_decode_relaxed_steps: int = 0
    lora_decode_tightened_steps: int = 0
    starvation_protected_steps: int = 0
    fairness_guardrail_triggered_steps: int = 0
    prefill_step_count: int = 0
    decode_step_count: int = 0
    mixed_lora_prefill_steps: int = 0
    mixed_lora_decode_steps: int = 0
    aged_admissions: int = 0
    max_queue_wait_s: float = 0.0
    admitted_queue_wait_sum_s: float = 0.0
    admitted_queue_wait_count: int = 0
    per_class_queue_wait_sum_s: dict[str, float] = field(default_factory=dict)
    per_class_queue_wait_count: dict[str, int] = field(default_factory=dict)
    per_class_max_queue_wait_s: dict[str, float] = field(default_factory=dict)
    queue_wait_samples_s: list[float] = field(default_factory=list)
    per_class_queue_wait_samples_s: dict[str, list[float]] = field(default_factory=dict)
    finished: list[tuple[str, str]] = field(default_factory=list)
    first_tokens: list[tuple[str, float]] = field(default_factory=list)
    aborted: list[str] = field(default_factory=list)
    background_errors: list[str] = field(default_factory=list)
    model_surface_events: list[dict[str, str]] = field(default_factory=list)
    deepseek_events: dict[str, int] = field(default_factory=dict)
    deepseek_decode_batch_tokens: int = 0
    deepseek_decode_batch_max_size: int = 0
    deepseek_decode_batch_latency_ms_sum: float = 0.0
    deepseek_stager_cache_hits: int = 0
    deepseek_stager_cache_misses: int = 0
    deepseek_kv_family_allocations: int = 0
    step_count: int = 0
    multimodal_requests: int = 0
    multimodal_images: int = 0
    multimodal_lora_requests: int = 0
    step_multimodal_admitted_requests: int = 0
    step_multimodal_prefill_requests: int = 0
    step_multimodal_decode_requests: int = 0
    step_multimodal_queued_requests: int = 0
    step_multimodal_lora_admitted_requests: int = 0
    step_multimodal_lora_prefill_requests: int = 0
    step_multimodal_lora_decode_requests: int = 0
    step_multimodal_lora_queued_requests: int = 0
    mixed_multimodal_lora_prefill_steps: int = 0
    multimodal_prefill_limit_sum: int = 0
    multimodal_prefill_limit_relaxed_steps: int = 0
    multimodal_prefill_limit_tightened_steps: int = 0
    multimodal_prefill_limit_triggered_steps: int = 0
    multimodal_lora_admit_limit_sum: int = 0
    multimodal_lora_prefill_limit_sum: int = 0
    multimodal_lora_decode_limit_sum: int = 0
    multimodal_lora_admit_limit_triggered_steps: int = 0
    multimodal_lora_prefill_limit_relaxed_steps: int = 0
    multimodal_lora_prefill_limit_tightened_steps: int = 0
    multimodal_lora_prefill_limit_triggered_steps: int = 0
    multimodal_lora_prefill_limit_relaxed_by_fairness_steps: int = 0
    multimodal_lora_prefill_limit_tightened_by_locality_steps: int = 0
    multimodal_lora_prefill_max_fairness_gap_sum: float = 0.0
    multimodal_lora_decode_limit_relaxed_steps: int = 0
    multimodal_lora_decode_limit_tightened_steps: int = 0
    multimodal_lora_decode_limit_triggered_steps: int = 0
    multimodal_lora_decode_limit_relaxed_by_fairness_steps: int = 0
    multimodal_lora_decode_limit_tightened_by_locality_steps: int = 0
    multimodal_lora_decode_max_fairness_gap_sum: float = 0.0
    multimodal_max_queue_wait_s: float = 0.0
    multimodal_queue_wait_samples_s: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in (
            "added",
            "rejected",
            "admitted",
            "prefix_cache_events",
            "preemption_events",
            "queue_wait_samples_s",
            "finished",
            "first_tokens",
            "aborted",
            "background_errors",
            "model_surface_events",
            "multimodal_queue_wait_samples_s",
        ):
            values = getattr(self, name)
            setattr(self, name, _BoundedList(self.history_limit, values))

    def on_request_added(self, request_id: str, request: RequestState) -> None:
        mm_data = request.multi_modal_data or {}
        images = mm_data.get("image") if isinstance(mm_data, dict) else None
        self.added.append(request_id)
        if isinstance(images, list) and images:
            self.multimodal_requests += 1
            self.multimodal_images += len(images)
            if request.lora_id:
                self.multimodal_lora_requests += 1

    def on_request_rejected(self, request_id: str, reason: str) -> None:
        self.rejected.append((request_id, reason))
        reason_key = self._rejection_reason_key(reason)
        self.rejection_reason_counts[reason_key] = (
            self.rejection_reason_counts.get(reason_key, 0) + 1
        )

    def on_request_admitted(
        self,
        request_id: str,
        queue_wait_s: float,
        service_class: str | None = None,
    ) -> None:
        service_class = str(service_class or "latency")
        self.admitted.append((request_id, queue_wait_s, service_class))
        self.max_queue_wait_s = max(self.max_queue_wait_s, queue_wait_s)
        self.admitted_queue_wait_sum_s += queue_wait_s
        self.admitted_queue_wait_count += 1
        self.per_class_queue_wait_sum_s[service_class] = (
            self.per_class_queue_wait_sum_s.get(service_class, 0.0) + queue_wait_s
        )
        self.per_class_queue_wait_count[service_class] = (
            self.per_class_queue_wait_count.get(service_class, 0) + 1
        )
        self.per_class_max_queue_wait_s[service_class] = max(
            self.per_class_max_queue_wait_s.get(service_class, 0.0),
            queue_wait_s,
        )

    def on_prefix_cache_event(
        self,
        request_id: str,
        *,
        hit: bool,
        exact: bool,
        prefix_len: int,
        saved_prefill_tokens: int,
        is_multimodal: bool = False,
    ) -> None:
        self.prefix_cache_events.append(
            (request_id, hit, exact, prefix_len, saved_prefill_tokens)
        )
        if hit:
            self.prefix_cache_hits += 1
            if exact:
                self.prefix_cache_exact_hits += 1
            else:
                self.prefix_cache_partial_hits += 1
        else:
            self.prefix_cache_misses += 1
        self.prefix_cache_saved_prefill_tokens += max(0, int(saved_prefill_tokens))
        if is_multimodal:
            if hit:
                self.multimodal_prefix_cache_hits += 1
                self.multimodal_prefix_cache_saved_prefill_tokens += max(
                    0, int(saved_prefill_tokens)
                )
            else:
                self.multimodal_prefix_cache_misses += 1

    def on_preemption_event(
        self,
        *,
        preempted_prefill_requests: int,
        queued_backlog: int,
        multimodal_prefill_requests: int = 0,
    ) -> None:
        self.preemption_events.append((preempted_prefill_requests, queued_backlog))
        self.preempted_steps += 1
        self.preempted_prefill_requests += max(0, int(preempted_prefill_requests))
        self.preempted_multimodal_prefill_requests += max(
            0, int(multimodal_prefill_requests)
        )

    def on_multimodal_preemption_guard(
        self,
        *,
        protected_prefill_requests: int,
        prefix_cache_hit_rate: float,
    ) -> None:
        del prefix_cache_hit_rate
        self.protected_multimodal_prefix_steps += 1
        self.protected_multimodal_prefix_prefill_requests += max(
            0, int(protected_prefill_requests)
        )

    def on_step_started(self, plan: Any) -> None:
        self.step_count += 1
        plan = getattr(plan, "metrics", None) or plan
        if getattr(plan, "prefill_starvation_protected", False):
            self.starvation_protected_steps += 1
        if getattr(plan, "fairness_guardrail_triggered", False):
            self.fairness_guardrail_triggered_steps += 1
        self.aged_admissions += int(getattr(plan, "aged_admission_count", 0) or 0)
        self._merge_counts(
            self.fairness_admitted_service_classes,
            getattr(plan, "admitted_service_classes", None) or {},
        )
        self._merge_counts(
            self.fairness_prefill_service_classes,
            getattr(plan, "prefill_service_classes", None) or {},
        )
        self._merge_counts(
            self.fairness_decode_service_classes,
            getattr(plan, "decode_service_classes", None) or {},
        )
        self.step_multimodal_admitted_requests += int(
            getattr(plan, "admitted_multimodal_requests", 0) or 0
        )
        self.step_multimodal_lora_admitted_requests += int(
            getattr(plan, "admitted_multimodal_lora_requests", 0) or 0
        )
        self.step_multimodal_prefill_requests += int(
            getattr(plan, "prefill_multimodal_requests", 0) or 0
        )
        self.step_multimodal_lora_prefill_requests += int(
            getattr(plan, "prefill_multimodal_lora_requests", 0) or 0
        )
        self.step_multimodal_decode_requests += int(
            getattr(plan, "decode_multimodal_requests", 0) or 0
        )
        self.step_multimodal_lora_decode_requests += int(
            getattr(plan, "decode_multimodal_lora_requests", 0) or 0
        )
        self.step_multimodal_queued_requests += int(
            getattr(plan, "queued_multimodal_requests", 0) or 0
        )
        self.multimodal_prefill_limit_sum += int(
            getattr(plan, "effective_prefill_multimodal_request_limit", 0) or 0
        )
        if getattr(plan, "prefill_multimodal_limit_relaxed", False):
            self.multimodal_prefill_limit_relaxed_steps += 1
        if getattr(plan, "prefill_multimodal_limit_tightened", False):
            self.multimodal_prefill_limit_tightened_steps += 1
        if getattr(plan, "prefill_multimodal_limit_triggered", False):
            self.multimodal_prefill_limit_triggered_steps += 1
        self.step_multimodal_lora_queued_requests += int(
            getattr(plan, "queued_multimodal_lora_requests", 0) or 0
        )
        self.multimodal_lora_admit_limit_sum += int(
            getattr(plan, "effective_admit_multimodal_lora_request_limit", 0) or 0
        )
        self.multimodal_lora_prefill_limit_sum += int(
            getattr(plan, "effective_prefill_multimodal_lora_request_limit", 0) or 0
        )
        self.multimodal_lora_decode_limit_sum += int(
            getattr(plan, "effective_decode_multimodal_lora_request_limit", 0) or 0
        )
        if getattr(plan, "admit_multimodal_lora_limit_triggered", False):
            self.multimodal_lora_admit_limit_triggered_steps += 1
        if getattr(plan, "prefill_multimodal_lora_limit_relaxed", False):
            self.multimodal_lora_prefill_limit_relaxed_steps += 1
        if getattr(plan, "prefill_multimodal_lora_limit_relaxed_by_fairness", False):
            self.multimodal_lora_prefill_limit_relaxed_by_fairness_steps += 1
        if getattr(plan, "prefill_multimodal_lora_limit_tightened", False):
            self.multimodal_lora_prefill_limit_tightened_steps += 1
        if getattr(plan, "prefill_multimodal_lora_limit_tightened_by_locality", False):
            self.multimodal_lora_prefill_limit_tightened_by_locality_steps += 1
        if getattr(plan, "prefill_multimodal_lora_limit_triggered", False):
            self.multimodal_lora_prefill_limit_triggered_steps += 1
        self.multimodal_lora_prefill_max_fairness_gap_sum += float(
            getattr(plan, "prefill_multimodal_lora_max_fairness_gap", 0.0) or 0.0
        )
        if getattr(plan, "decode_multimodal_lora_limit_relaxed", False):
            self.multimodal_lora_decode_limit_relaxed_steps += 1
        if getattr(plan, "decode_multimodal_lora_limit_tightened", False):
            self.multimodal_lora_decode_limit_tightened_steps += 1
        if getattr(plan, "decode_multimodal_lora_limit_triggered", False):
            self.multimodal_lora_decode_limit_triggered_steps += 1
        if getattr(plan, "decode_multimodal_lora_limit_relaxed_by_fairness", False):
            self.multimodal_lora_decode_limit_relaxed_by_fairness_steps += 1
        if getattr(plan, "decode_multimodal_lora_limit_tightened_by_locality", False):
            self.multimodal_lora_decode_limit_tightened_by_locality_steps += 1
        self.multimodal_lora_decode_max_fairness_gap_sum += float(
            getattr(plan, "decode_multimodal_lora_max_fairness_gap", 0.0) or 0.0
        )
        if int(getattr(plan, "prefill_multimodal_lora_requests", 0) or 0) > 0 and int(
            getattr(plan, "prefill_multimodal_requests", 0) or 0
        ) > int(getattr(plan, "prefill_multimodal_lora_requests", 0) or 0):
            self.mixed_multimodal_lora_prefill_steps += 1
        self.multimodal_max_queue_wait_s = max(
            self.multimodal_max_queue_wait_s,
            float(getattr(plan, "queued_multimodal_max_wait_s", 0.0) or 0.0),
        )
        self.multimodal_queue_wait_samples_s.append(
            float(getattr(plan, "queued_multimodal_p95_wait_s", 0.0) or 0.0)
        )
        self._merge_counts(
            self.fairness_backlog_service_classes,
            getattr(plan, "queued_service_classes", None) or {},
        )
        self._merge_counts(
            self.lora_admitted_adapters,
            getattr(plan, "admitted_lora_adapters", None) or {},
        )
        self._merge_counts(
            self.lora_prefill_adapters,
            getattr(plan, "prefill_lora_adapters", None) or {},
        )
        self._merge_counts(
            self.lora_decode_adapters,
            getattr(plan, "decode_lora_adapters", None) or {},
        )
        self._merge_counts(
            self.lora_backlog_adapters,
            getattr(plan, "queued_lora_adapters", None) or {},
        )
        if getattr(plan, "admit_lora_limit_relaxed", False):
            self.lora_admit_relaxed_steps += 1
        if getattr(plan, "admit_lora_limit_tightened", False):
            self.lora_admit_tightened_steps += 1
        if getattr(plan, "prefill_lora_limit_relaxed", False):
            self.lora_prefill_relaxed_steps += 1
        if getattr(plan, "prefill_lora_limit_tightened", False):
            self.lora_prefill_tightened_steps += 1
        if getattr(plan, "decode_lora_limit_relaxed", False):
            self.lora_decode_relaxed_steps += 1
        if getattr(plan, "decode_lora_limit_tightened", False):
            self.lora_decode_tightened_steps += 1
        if len(getattr(plan, "prefill_lora_adapters", None) or {}) > 1:
            self.mixed_lora_prefill_steps += 1
        if len(getattr(plan, "decode_lora_adapters", None) or {}) > 1:
            self.mixed_lora_decode_steps += 1
        self.max_queue_wait_s = max(
            self.max_queue_wait_s,
            float(getattr(plan, "queued_max_wait_s", 0.0) or 0.0),
        )
        self._merge_float_sums(
            self.per_class_queue_wait_sum_s,
            self.per_class_queue_wait_count,
            getattr(plan, "queued_service_class_avg_wait_s", None) or {},
            getattr(plan, "queued_service_classes", None) or {},
        )
        self._merge_float_max(
            self.per_class_max_queue_wait_s,
            getattr(plan, "queued_service_class_max_wait_s", None) or {},
        )
        self.queue_wait_samples_s.append(
            float(getattr(plan, "queued_p95_wait_s", 0.0) or 0.0)
        )
        self._merge_wait_samples(
            self.per_class_queue_wait_samples_s,
            getattr(plan, "queued_service_class_p95_wait_s", None) or {},
        )

    def on_request_finished(self, request_id: str, reason: str) -> None:
        self.finished.append((request_id, reason))

    def on_prefill_executed(self, plan: Any, num_outputs: int) -> None:
        del plan, num_outputs
        self.prefill_step_count += 1

    def on_decode_executed(self, plan: Any, num_outputs: int) -> None:
        del plan, num_outputs
        self.decode_step_count += 1

    def on_first_token(self, request_id: str, ttft_s: float) -> None:
        self.first_tokens.append((request_id, ttft_s))

    def on_request_aborted(self, request_id: str) -> None:
        self.aborted.append(request_id)

    def on_background_error(self, exc: BaseException, request_ids: list[str]) -> None:
        self.background_errors.append(
            f"{type(exc).__name__}:{exc}:{','.join(request_ids)}"
        )

    def on_model_surface_resolved(
        self,
        *,
        event_name: str,
        model_name: str,
        model_type: str,
        status: str,
        reason: str,
    ) -> None:
        self.model_surface_events.append(
            {
                "event_name": str(event_name),
                "model_name": str(model_name),
                "model_type": str(model_type),
                "status": str(status),
                "reason": str(reason),
            }
        )


    def on_deepseek_event(self, event: str, **payload: Any) -> None:
        event = str(event)
        self.deepseek_events[event] = self.deepseek_events.get(event, 0) + 1
        if event == "decode_batch":
            batch_size = int(payload.get("batch_size", 0) or 0)
            self.deepseek_decode_batch_tokens += max(0, batch_size)
            self.deepseek_decode_batch_max_size = max(
                self.deepseek_decode_batch_max_size, batch_size
            )
            self.deepseek_decode_batch_latency_ms_sum += float(
                payload.get("latency_ms", 0.0) or 0.0
            )
        elif event == "stager_cache_hit":
            self.deepseek_stager_cache_hits += 1
        elif event == "stager_cache_miss":
            self.deepseek_stager_cache_misses += 1
        elif event == "kv_family_allocation":
            self.deepseek_kv_family_allocations += 1

    def stats(self) -> dict[str, Any]:
        prefix_cache_hits = self.prefix_cache_hits
        prefix_cache_events = len(self.prefix_cache_events)
        model_surface_last = (
            self.model_surface_events[-1] if self.model_surface_events else {}
        )
        return {
            "added_requests": len(self.added),
            "rejected_requests": len(self.rejected),
            "admitted_requests": len(self.admitted),
            "finished_requests": len(self.finished),
            "aborted_requests": len(self.aborted),
            "background_error_count": len(self.background_errors),
            "step_count": self.step_count,
            "first_token_count": len(self.first_tokens),
            "rejections": {
                "reasons": dict(self.rejection_reason_counts),
                "queue_timeout": self.rejection_reason_counts.get("queue_timeout", 0),
            },
            "deepseek": {
                "events": dict(self.deepseek_events),
                "decode_batch_tokens": self.deepseek_decode_batch_tokens,
                "decode_batch_max_size": self.deepseek_decode_batch_max_size,
                "decode_batch_latency_ms_sum": (
                    self.deepseek_decode_batch_latency_ms_sum
                ),
                "stager_cache_hits": self.deepseek_stager_cache_hits,
                "stager_cache_misses": self.deepseek_stager_cache_misses,
                "kv_family_allocations": self.deepseek_kv_family_allocations,
            },
            "model_surface": {
                "events": len(self.model_surface_events),
                "experimental_events": sum(
                    1
                    for event in self.model_surface_events
                    if event["status"] == "experimental"
                ),
                "supported_events": sum(
                    1
                    for event in self.model_surface_events
                    if event["status"] == "supported"
                ),
                "last_event_name": model_surface_last.get("event_name", ""),
                "last_model_name": model_surface_last.get("model_name", ""),
                "last_model_type": model_surface_last.get("model_type", ""),
                "last_status": model_surface_last.get("status", ""),
                "last_reason": model_surface_last.get("reason", ""),
            },
            "multimodal": {
                "requests": self.multimodal_requests,
                "images": self.multimodal_images,
                "multimodal_lora_requests": self.multimodal_lora_requests,
                "admitted_requests": self.step_multimodal_admitted_requests,
                "prefill_requests": self.step_multimodal_prefill_requests,
                "decode_requests": self.step_multimodal_decode_requests,
                "queued_requests": self.step_multimodal_queued_requests,
                "admitted_multimodal_lora_requests": (
                    self.step_multimodal_lora_admitted_requests
                ),
                "prefill_multimodal_lora_requests": (
                    self.step_multimodal_lora_prefill_requests
                ),
                "decode_multimodal_lora_requests": (
                    self.step_multimodal_lora_decode_requests
                ),
                "queued_multimodal_lora_requests": (
                    self.step_multimodal_lora_queued_requests
                ),
                "mixed_multimodal_lora_prefill_steps": (
                    self.mixed_multimodal_lora_prefill_steps
                ),
                "prefix_cache_hits": self.multimodal_prefix_cache_hits,
                "prefix_cache_misses": self.multimodal_prefix_cache_misses,
                "prefix_cache_saved_prefill_tokens": (
                    self.multimodal_prefix_cache_saved_prefill_tokens
                ),
                "prefix_cache_hit_rate": (
                    self.multimodal_prefix_cache_hits
                    / (
                        self.multimodal_prefix_cache_hits
                        + self.multimodal_prefix_cache_misses
                    )
                    if (
                        self.multimodal_prefix_cache_hits
                        + self.multimodal_prefix_cache_misses
                    )
                    else 0.0
                ),
                "mixed_multimodal_lora_prefill_ratio": (
                    self.mixed_multimodal_lora_prefill_steps / self.prefill_step_count
                    if self.prefill_step_count
                    else 0.0
                ),
                "avg_effective_prefill_multimodal_limit": (
                    self.multimodal_prefill_limit_sum / self.step_count
                    if self.step_count
                    else 0.0
                ),
                "prefill_multimodal_limit_relaxed_steps": (
                    self.multimodal_prefill_limit_relaxed_steps
                ),
                "prefill_multimodal_limit_tightened_steps": (
                    self.multimodal_prefill_limit_tightened_steps
                ),
                "prefill_multimodal_limit_triggered_steps": (
                    self.multimodal_prefill_limit_triggered_steps
                ),
                "avg_effective_admit_multimodal_lora_limit": (
                    self.multimodal_lora_admit_limit_sum / self.step_count
                    if self.step_count
                    else 0.0
                ),
                "avg_effective_prefill_multimodal_lora_limit": (
                    self.multimodal_lora_prefill_limit_sum / self.step_count
                    if self.step_count
                    else 0.0
                ),
                "avg_effective_decode_multimodal_lora_limit": (
                    self.multimodal_lora_decode_limit_sum / self.step_count
                    if self.step_count
                    else 0.0
                ),
                "admit_multimodal_lora_limit_triggered_steps": (
                    self.multimodal_lora_admit_limit_triggered_steps
                ),
                "prefill_multimodal_lora_limit_relaxed_steps": (
                    self.multimodal_lora_prefill_limit_relaxed_steps
                ),
                "prefill_multimodal_lora_limit_tightened_steps": (
                    self.multimodal_lora_prefill_limit_tightened_steps
                ),
                "prefill_multimodal_lora_limit_relaxed_by_fairness_steps": (
                    self.multimodal_lora_prefill_limit_relaxed_by_fairness_steps
                ),
                "prefill_multimodal_lora_limit_tightened_by_locality_steps": (
                    self.multimodal_lora_prefill_limit_tightened_by_locality_steps
                ),
                "prefill_multimodal_lora_limit_triggered_steps": (
                    self.multimodal_lora_prefill_limit_triggered_steps
                ),
                "avg_prefill_multimodal_lora_max_fairness_gap": (
                    self.multimodal_lora_prefill_max_fairness_gap_sum / self.step_count
                    if self.step_count
                    else 0.0
                ),
                "decode_multimodal_lora_limit_relaxed_steps": (
                    self.multimodal_lora_decode_limit_relaxed_steps
                ),
                "decode_multimodal_lora_limit_tightened_steps": (
                    self.multimodal_lora_decode_limit_tightened_steps
                ),
                "decode_multimodal_lora_limit_triggered_steps": (
                    self.multimodal_lora_decode_limit_triggered_steps
                ),
                "decode_multimodal_lora_limit_relaxed_by_fairness_steps": (
                    self.multimodal_lora_decode_limit_relaxed_by_fairness_steps
                ),
                "decode_multimodal_lora_limit_tightened_by_locality_steps": (
                    self.multimodal_lora_decode_limit_tightened_by_locality_steps
                ),
                "avg_decode_multimodal_lora_max_fairness_gap": (
                    self.multimodal_lora_decode_max_fairness_gap_sum / self.step_count
                    if self.step_count
                    else 0.0
                ),
                "max_queue_wait_s": self.multimodal_max_queue_wait_s,
                "p95_queue_wait_s": self._percentile(
                    self.multimodal_queue_wait_samples_s, 0.95
                ),
            },
            "prefix_cache": {
                "events": prefix_cache_events,
                "hits": prefix_cache_hits,
                "misses": self.prefix_cache_misses,
                "exact_hits": self.prefix_cache_exact_hits,
                "partial_hits": self.prefix_cache_partial_hits,
                "saved_prefill_tokens": self.prefix_cache_saved_prefill_tokens,
                "hit_rate": (
                    prefix_cache_hits / prefix_cache_events
                    if prefix_cache_events
                    else 0.0
                ),
                "exact_hit_rate": (
                    self.prefix_cache_exact_hits / prefix_cache_hits
                    if prefix_cache_hits
                    else 0.0
                ),
                "partial_hit_rate": (
                    self.prefix_cache_partial_hits / prefix_cache_hits
                    if prefix_cache_hits
                    else 0.0
                ),
                "avg_saved_prefill_tokens_per_hit": (
                    self.prefix_cache_saved_prefill_tokens / prefix_cache_hits
                    if prefix_cache_hits
                    else 0.0
                ),
                "avg_saved_prefill_tokens_per_request": (
                    self.prefix_cache_saved_prefill_tokens / prefix_cache_events
                    if prefix_cache_events
                    else 0.0
                ),
            },
            "preemption": {
                "events": len(self.preemption_events),
                "preempted_steps": self.preempted_steps,
                "preempted_prefill_requests": self.preempted_prefill_requests,
                "preempted_multimodal_prefill_requests": (
                    self.preempted_multimodal_prefill_requests
                ),
                "protected_multimodal_prefix_steps": (
                    self.protected_multimodal_prefix_steps
                ),
                "protected_multimodal_prefix_prefill_requests": (
                    self.protected_multimodal_prefix_prefill_requests
                ),
            },
            "fairness": {
                "aged_admissions": self.aged_admissions,
                "starvation_protected_steps": self.starvation_protected_steps,
                "fairness_guardrail_triggered_steps": (
                    self.fairness_guardrail_triggered_steps
                ),
                "backlog_service_classes": dict(self.fairness_backlog_service_classes),
                "admitted_service_classes": dict(
                    self.fairness_admitted_service_classes
                ),
                "prefill_service_classes": dict(self.fairness_prefill_service_classes),
                "decode_service_classes": dict(self.fairness_decode_service_classes),
                "max_queue_wait_s": self.max_queue_wait_s,
                "p95_queue_wait_s": self._percentile(self.queue_wait_samples_s, 0.95),
                "avg_admitted_queue_wait_s": (
                    self.admitted_queue_wait_sum_s / self.admitted_queue_wait_count
                    if self.admitted_queue_wait_count
                    else 0.0
                ),
                "per_class_avg_queue_wait_s": {
                    key: self.per_class_queue_wait_sum_s[key]
                    / max(1, self.per_class_queue_wait_count.get(key, 0))
                    for key in self.per_class_queue_wait_sum_s
                },
                "per_class_max_queue_wait_s": dict(self.per_class_max_queue_wait_s),
                "per_class_p95_queue_wait_s": {
                    key: self._percentile(values, 0.95)
                    for key, values in self.per_class_queue_wait_samples_s.items()
                },
            },
            "lora": {
                "admitted_adapters": dict(self.lora_admitted_adapters),
                "prefill_adapters": dict(self.lora_prefill_adapters),
                "decode_adapters": dict(self.lora_decode_adapters),
                "backlog_adapters": dict(self.lora_backlog_adapters),
                "admit_relaxed_steps": self.lora_admit_relaxed_steps,
                "admit_tightened_steps": self.lora_admit_tightened_steps,
                "prefill_relaxed_steps": self.lora_prefill_relaxed_steps,
                "prefill_tightened_steps": self.lora_prefill_tightened_steps,
                "decode_relaxed_steps": self.lora_decode_relaxed_steps,
                "decode_tightened_steps": self.lora_decode_tightened_steps,
                "prefill_step_count": self.prefill_step_count,
                "decode_step_count": self.decode_step_count,
                "mixed_lora_prefill_steps": self.mixed_lora_prefill_steps,
                "mixed_lora_decode_steps": self.mixed_lora_decode_steps,
                "prefill_locality_rate": (
                    1.0 - (self.mixed_lora_prefill_steps / self.prefill_step_count)
                    if self.prefill_step_count
                    else 0.0
                ),
                "decode_locality_rate": (
                    1.0 - (self.mixed_lora_decode_steps / self.decode_step_count)
                    if self.decode_step_count
                    else 0.0
                ),
            },
        }

    def reset_stats(self) -> None:
        self.added.clear()
        self.rejected.clear()
        self.rejection_reason_counts.clear()
        self.admitted.clear()
        self.prefix_cache_events.clear()
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0
        self.prefix_cache_exact_hits = 0
        self.prefix_cache_partial_hits = 0
        self.prefix_cache_saved_prefill_tokens = 0
        self.multimodal_prefix_cache_hits = 0
        self.multimodal_prefix_cache_misses = 0
        self.multimodal_prefix_cache_saved_prefill_tokens = 0
        self.preemption_events.clear()
        self.preempted_steps = 0
        self.preempted_prefill_requests = 0
        self.preempted_multimodal_prefill_requests = 0
        self.protected_multimodal_prefix_steps = 0
        self.protected_multimodal_prefix_prefill_requests = 0
        self.fairness_admitted_service_classes.clear()
        self.fairness_prefill_service_classes.clear()
        self.fairness_decode_service_classes.clear()
        self.fairness_backlog_service_classes.clear()
        self.lora_admitted_adapters.clear()
        self.lora_prefill_adapters.clear()
        self.lora_decode_adapters.clear()
        self.lora_backlog_adapters.clear()
        self.lora_admit_relaxed_steps = 0
        self.lora_admit_tightened_steps = 0
        self.lora_prefill_relaxed_steps = 0
        self.lora_prefill_tightened_steps = 0
        self.lora_decode_relaxed_steps = 0
        self.lora_decode_tightened_steps = 0
        self.starvation_protected_steps = 0
        self.fairness_guardrail_triggered_steps = 0
        self.prefill_step_count = 0
        self.decode_step_count = 0
        self.mixed_lora_prefill_steps = 0
        self.mixed_lora_decode_steps = 0
        self.aged_admissions = 0
        self.max_queue_wait_s = 0.0
        self.admitted_queue_wait_sum_s = 0.0
        self.admitted_queue_wait_count = 0
        self.per_class_queue_wait_sum_s.clear()
        self.per_class_queue_wait_count.clear()
        self.per_class_max_queue_wait_s.clear()
        self.queue_wait_samples_s.clear()
        self.per_class_queue_wait_samples_s.clear()
        self.finished.clear()
        self.first_tokens.clear()
        self.aborted.clear()
        self.background_errors.clear()
        self.model_surface_events.clear()
        self.deepseek_events.clear()
        self.deepseek_decode_batch_tokens = 0
        self.deepseek_decode_batch_max_size = 0
        self.deepseek_decode_batch_latency_ms_sum = 0.0
        self.deepseek_stager_cache_hits = 0
        self.deepseek_stager_cache_misses = 0
        self.deepseek_kv_family_allocations = 0
        self.step_count = 0
        self.multimodal_requests = 0
        self.multimodal_images = 0
        self.multimodal_lora_requests = 0
        self.step_multimodal_admitted_requests = 0
        self.step_multimodal_prefill_requests = 0
        self.step_multimodal_decode_requests = 0
        self.step_multimodal_queued_requests = 0
        self.step_multimodal_lora_admitted_requests = 0
        self.step_multimodal_lora_prefill_requests = 0
        self.step_multimodal_lora_decode_requests = 0
        self.step_multimodal_lora_queued_requests = 0
        self.mixed_multimodal_lora_prefill_steps = 0
        self.multimodal_prefill_limit_sum = 0
        self.multimodal_prefill_limit_relaxed_steps = 0
        self.multimodal_prefill_limit_tightened_steps = 0
        self.multimodal_prefill_limit_triggered_steps = 0
        self.multimodal_lora_admit_limit_sum = 0
        self.multimodal_lora_prefill_limit_sum = 0
        self.multimodal_lora_decode_limit_sum = 0
        self.multimodal_lora_admit_limit_triggered_steps = 0
        self.multimodal_lora_prefill_limit_relaxed_steps = 0
        self.multimodal_lora_prefill_limit_tightened_steps = 0
        self.multimodal_lora_prefill_limit_triggered_steps = 0
        self.multimodal_lora_prefill_limit_relaxed_by_fairness_steps = 0
        self.multimodal_lora_prefill_limit_tightened_by_locality_steps = 0
        self.multimodal_lora_prefill_max_fairness_gap_sum = 0.0
        self.multimodal_lora_decode_limit_relaxed_steps = 0
        self.multimodal_lora_decode_limit_tightened_steps = 0
        self.multimodal_lora_decode_limit_triggered_steps = 0
        self.multimodal_lora_decode_limit_relaxed_by_fairness_steps = 0
        self.multimodal_lora_decode_limit_tightened_by_locality_steps = 0
        self.multimodal_lora_decode_max_fairness_gap_sum = 0.0
        self.multimodal_max_queue_wait_s = 0.0
        self.multimodal_queue_wait_samples_s.clear()

    @staticmethod
    def _rejection_reason_key(reason: str) -> str:
        normalized = str(reason or "unknown").strip().lower()
        if normalized.startswith("queue timeout"):
            return "queue_timeout"
        return normalized.replace(" ", "_") or "unknown"

    @staticmethod
    def _merge_counts(target: dict[str, int], delta: dict[str, int]) -> None:
        for key, value in delta.items():
            target[key] = target.get(key, 0) + int(value)

    @staticmethod
    def _merge_float_max(target: dict[str, float], delta: dict[str, float]) -> None:
        for key, value in delta.items():
            target[key] = max(target.get(key, 0.0), float(value))

    @staticmethod
    def _merge_float_sums(
        target_sum: dict[str, float],
        target_count: dict[str, int],
        avg_delta: dict[str, float],
        count_delta: dict[str, int],
    ) -> None:
        for key, count in count_delta.items():
            if count <= 0:
                continue
            avg_value = float(avg_delta.get(key, 0.0))
            target_sum[key] = target_sum.get(key, 0.0) + (avg_value * int(count))
            target_count[key] = target_count.get(key, 0) + int(count)

    def _merge_wait_samples(
        self,
        target: dict[str, list[float]],
        delta: dict[str, float],
    ) -> None:
        for key, value in delta.items():
            samples = target.setdefault(key, _BoundedList(self.history_limit))
            samples.append(float(value))

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        sorted_values = sorted(float(v) for v in values)
        idx = int(round((len(sorted_values) - 1) * max(0.0, min(1.0, q))))
        return sorted_values[idx]


class LoggingRuntimeObserver(RuntimeObserver):
    def on_request_added(self, request_id: str, request: RequestState) -> None:
        mm_data = request.multi_modal_data or {}
        images = mm_data.get("image") if isinstance(mm_data, dict) else None
        logger.info(
            "runtime request added id=%s prompt_tokens=%s "
            "is_prefill=%s multimodal_images=%s",
            request_id,
            len(request.input_ids),
            request.is_prefill,
            len(images) if isinstance(images, list) else 0,
        )

    def on_request_rejected(self, request_id: str, reason: str) -> None:
        logger.warning("runtime request rejected id=%s reason=%s", request_id, reason)

    def on_request_admitted(
        self,
        request_id: str,
        queue_wait_s: float,
        service_class: str | None = None,
    ) -> None:
        logger.debug(
            "runtime request admitted id=%s queue_wait_s=%.6f service_class=%s",
            request_id,
            queue_wait_s,
            service_class or "latency",
        )

    def on_prefix_cache_event(
        self,
        request_id: str,
        *,
        hit: bool,
        exact: bool,
        prefix_len: int,
        saved_prefill_tokens: int,
        is_multimodal: bool = False,
    ) -> None:
        del is_multimodal
        logger.debug(
            "runtime prefix cache id=%s hit=%s exact=%s "
            "prefix_len=%s saved_prefill_tokens=%s",
            request_id,
            hit,
            exact,
            prefix_len,
            saved_prefill_tokens,
        )

    def on_preemption_event(
        self,
        *,
        preempted_prefill_requests: int,
        queued_backlog: int,
        multimodal_prefill_requests: int = 0,
    ) -> None:
        logger.debug(
            "runtime preemption preempted_prefill_requests=%s "
            "queued_backlog=%s multimodal_prefills=%s",
            preempted_prefill_requests,
            queued_backlog,
            multimodal_prefill_requests,
        )

    def on_multimodal_preemption_guard(
        self,
        *,
        protected_prefill_requests: int,
        prefix_cache_hit_rate: float,
    ) -> None:
        logger.debug(
            "runtime multimodal preemption guard "
            "protected_prefill_requests=%s prefix_cache_hit_rate=%.4f",
            protected_prefill_requests,
            prefix_cache_hit_rate,
        )

    def on_step_started(self, plan: Any) -> None:
        logger.debug("runtime step plan=%s", plan)

    def on_prefill_executed(self, plan: Any, num_outputs: int) -> None:
        logger.debug("runtime prefill executed plan=%s outputs=%s", plan, num_outputs)

    def on_decode_executed(self, plan: Any, num_outputs: int) -> None:
        logger.debug("runtime decode executed plan=%s outputs=%s", plan, num_outputs)

    def on_request_finished(self, request_id: str, reason: str) -> None:
        logger.info("runtime request finished id=%s reason=%s", request_id, reason)

    def on_first_token(self, request_id: str, ttft_s: float) -> None:
        logger.debug("runtime first token id=%s ttft_s=%.6f", request_id, ttft_s)

    def on_request_aborted(self, request_id: str) -> None:
        logger.info("runtime request aborted id=%s", request_id)

    def on_background_error(self, exc: BaseException, request_ids: list[str]) -> None:
        logger.exception(
            "runtime background error requests=%s exc=%s", request_ids, exc
        )

    def on_model_surface_resolved(
        self,
        *,
        event_name: str,
        model_name: str,
        model_type: str,
        status: str,
        reason: str,
    ) -> None:
        log = logger.warning if status == "experimental" else logger.info
        log(
            "runtime model surface event=%s model=%s model_type=%s status=%s reason=%s",
            event_name,
            model_name,
            model_type,
            status,
            reason,
        )

    def stats(self) -> dict[str, Any]:
        return {}

    def reset_stats(self) -> None:
        return None
