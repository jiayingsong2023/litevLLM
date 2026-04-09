# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class RuntimeObserver:
    def on_request_added(self, request_id: str, request: dict[str, Any]) -> None:
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
    ) -> None:
        pass

    def on_preemption_event(
        self,
        *,
        preempted_prefill_requests: int,
        queued_backlog: int,
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

    def stats(self) -> dict[str, Any]:
        return {}

    def reset_stats(self) -> None:
        return None


class NullRuntimeObserver(RuntimeObserver):
    pass


@dataclass
class InMemoryRuntimeObserver(RuntimeObserver):
    added: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    admitted: list[tuple[str, float, str]] = field(default_factory=list)
    prefix_cache_events: list[tuple[str, bool, bool, int, int]] = field(default_factory=list)
    prefix_cache_hits: int = 0
    prefix_cache_misses: int = 0
    prefix_cache_exact_hits: int = 0
    prefix_cache_partial_hits: int = 0
    prefix_cache_saved_prefill_tokens: int = 0
    preemption_events: list[tuple[int, int]] = field(default_factory=list)
    preempted_steps: int = 0
    preempted_prefill_requests: int = 0
    fairness_admitted_service_classes: dict[str, int] = field(default_factory=dict)
    fairness_prefill_service_classes: dict[str, int] = field(default_factory=dict)
    fairness_decode_service_classes: dict[str, int] = field(default_factory=dict)
    fairness_backlog_service_classes: dict[str, int] = field(default_factory=dict)
    starvation_protected_steps: int = 0
    fairness_guardrail_triggered_steps: int = 0
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
    step_count: int = 0

    def on_request_added(self, request_id: str, request: dict[str, Any]) -> None:
        del request
        self.added.append(request_id)

    def on_request_rejected(self, request_id: str, reason: str) -> None:
        self.rejected.append((request_id, reason))

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

    def on_preemption_event(
        self,
        *,
        preempted_prefill_requests: int,
        queued_backlog: int,
    ) -> None:
        self.preemption_events.append(
            (preempted_prefill_requests, queued_backlog)
        )
        self.preempted_steps += 1
        self.preempted_prefill_requests += max(0, int(preempted_prefill_requests))

    def on_step_started(self, plan: Any) -> None:
        self.step_count += 1
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
        self._merge_counts(
            self.fairness_backlog_service_classes,
            getattr(plan, "queued_service_classes", None) or {},
        )
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
        self.queue_wait_samples_s.append(float(getattr(plan, "queued_p95_wait_s", 0.0) or 0.0))
        self._merge_wait_samples(
            self.per_class_queue_wait_samples_s,
            getattr(plan, "queued_service_class_p95_wait_s", None) or {},
        )

    def on_request_finished(self, request_id: str, reason: str) -> None:
        self.finished.append((request_id, reason))

    def on_first_token(self, request_id: str, ttft_s: float) -> None:
        self.first_tokens.append((request_id, ttft_s))

    def on_request_aborted(self, request_id: str) -> None:
        self.aborted.append(request_id)

    def on_background_error(self, exc: BaseException, request_ids: list[str]) -> None:
        self.background_errors.append(f"{type(exc).__name__}:{exc}:{','.join(request_ids)}")

    def stats(self) -> dict[str, Any]:
        prefix_cache_hits = self.prefix_cache_hits
        prefix_cache_events = len(self.prefix_cache_events)
        return {
            "added_requests": len(self.added),
            "rejected_requests": len(self.rejected),
            "admitted_requests": len(self.admitted),
            "finished_requests": len(self.finished),
            "aborted_requests": len(self.aborted),
            "background_error_count": len(self.background_errors),
            "step_count": self.step_count,
            "first_token_count": len(self.first_tokens),
            "prefix_cache": {
                "events": prefix_cache_events,
                "hits": prefix_cache_hits,
                "misses": self.prefix_cache_misses,
                "exact_hits": self.prefix_cache_exact_hits,
                "partial_hits": self.prefix_cache_partial_hits,
                "saved_prefill_tokens": self.prefix_cache_saved_prefill_tokens,
                "hit_rate": (
                    prefix_cache_hits / prefix_cache_events if prefix_cache_events else 0.0
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
            },
            "fairness": {
                "aged_admissions": self.aged_admissions,
                "starvation_protected_steps": self.starvation_protected_steps,
                "fairness_guardrail_triggered_steps": self.fairness_guardrail_triggered_steps,
                "backlog_service_classes": dict(self.fairness_backlog_service_classes),
                "admitted_service_classes": dict(self.fairness_admitted_service_classes),
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
        }

    def reset_stats(self) -> None:
        self.added.clear()
        self.rejected.clear()
        self.admitted.clear()
        self.prefix_cache_events.clear()
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0
        self.prefix_cache_exact_hits = 0
        self.prefix_cache_partial_hits = 0
        self.prefix_cache_saved_prefill_tokens = 0
        self.preemption_events.clear()
        self.preempted_steps = 0
        self.preempted_prefill_requests = 0
        self.fairness_admitted_service_classes.clear()
        self.fairness_prefill_service_classes.clear()
        self.fairness_decode_service_classes.clear()
        self.fairness_backlog_service_classes.clear()
        self.starvation_protected_steps = 0
        self.fairness_guardrail_triggered_steps = 0
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
        self.step_count = 0

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

    @staticmethod
    def _merge_wait_samples(
        target: dict[str, list[float]],
        delta: dict[str, float],
    ) -> None:
        for key, value in delta.items():
            target.setdefault(key, []).append(float(value))

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
    def on_request_added(self, request_id: str, request: dict[str, Any]) -> None:
        logger.info(
            "runtime request added id=%s prompt_tokens=%s is_prefill=%s",
            request_id,
            len(request.get("input_ids", [])),
            request.get("is_prefill"),
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
    ) -> None:
        logger.debug(
            "runtime prefix cache id=%s hit=%s exact=%s prefix_len=%s saved_prefill_tokens=%s",
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
    ) -> None:
        logger.debug(
            "runtime preemption preempted_prefill_requests=%s queued_backlog=%s",
            preempted_prefill_requests,
            queued_backlog,
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

    def stats(self) -> dict[str, Any]:
        return {}

    def reset_stats(self) -> None:
        return None
