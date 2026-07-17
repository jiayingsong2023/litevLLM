# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from vllm.engine.request_state import RequestState


class RuntimeObserver:
    def on_request_added(self, request_id: str, request: RequestState) -> None:
        pass

    def on_request_admitted(self, request_id: str, queue_wait_s: float) -> None:
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
    """Bounded diagnostics for the supported runtime, not a scheduling feedback loop."""

    history_limit: int = 4096
    added: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    admitted: list[tuple[str, float]] = field(default_factory=list)
    prefix_cache_events: list[tuple[str, bool, bool, int, int]] = field(
        default_factory=list
    )
    finished: list[tuple[str, str]] = field(default_factory=list)
    first_tokens: list[tuple[str, float]] = field(default_factory=list)
    aborted: list[str] = field(default_factory=list)
    background_errors: list[str] = field(default_factory=list)
    model_surface_events: list[dict[str, str]] = field(default_factory=list)
    rejection_reason_counts: Counter[str] = field(default_factory=Counter)
    prefix_cache_hits: int = 0
    prefix_cache_misses: int = 0
    prefix_cache_exact_hits: int = 0
    prefix_cache_partial_hits: int = 0
    prefix_cache_saved_prefill_tokens: int = 0
    starvation_protected_steps: int = 0
    prefill_step_count: int = 0
    decode_step_count: int = 0
    deepseek_events: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        for name in (
            "added",
            "rejected",
            "admitted",
            "prefix_cache_events",
            "finished",
            "first_tokens",
            "aborted",
            "background_errors",
            "model_surface_events",
        ):
            setattr(self, name, list(getattr(self, name))[-self.history_limit :])

    def _append(self, name: str, value: Any) -> None:
        values = getattr(self, name)
        values.append(value)
        if len(values) > self.history_limit:
            del values[: len(values) - self.history_limit]

    def on_request_added(self, request_id: str, request: RequestState) -> None:
        del request
        self._append("added", request_id)

    def on_request_rejected(self, request_id: str, reason: str) -> None:
        self._append("rejected", (request_id, reason))
        self.rejection_reason_counts[
            "queue_timeout" if reason.startswith("queue timeout") else reason
        ] += 1

    def on_request_admitted(self, request_id: str, queue_wait_s: float) -> None:
        self._append("admitted", (request_id, queue_wait_s))

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
        self._append(
            "prefix_cache_events",
            (request_id, hit, exact, prefix_len, saved_prefill_tokens),
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

    def on_step_started(self, plan: Any) -> None:
        metrics = getattr(plan, "metrics", None) or plan
        self.starvation_protected_steps += int(
            bool(getattr(metrics, "prefill_starvation_protected", False))
        )

    def on_prefill_executed(self, plan: Any, num_outputs: int) -> None:
        del plan, num_outputs
        self.prefill_step_count += 1

    def on_decode_executed(self, plan: Any, num_outputs: int) -> None:
        del plan, num_outputs
        self.decode_step_count += 1

    def on_request_finished(self, request_id: str, reason: str) -> None:
        self._append("finished", (request_id, reason))

    def on_first_token(self, request_id: str, ttft_s: float) -> None:
        self._append("first_tokens", (request_id, ttft_s))

    def on_request_aborted(self, request_id: str) -> None:
        self._append("aborted", request_id)

    def on_background_error(self, exc: BaseException, request_ids: list[str]) -> None:
        self._append(
            "background_errors", f"{type(exc).__name__}: {exc}; requests={request_ids}"
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
        self._append(
            "model_surface_events",
            {
                "event_name": event_name,
                "model_name": model_name,
                "model_type": model_type,
                "status": status,
                "reason": reason,
            },
        )

    def on_deepseek_event(self, event: str, **payload: Any) -> None:
        del payload
        self.deepseek_events[event] += 1

    def stats(self) -> dict[str, Any]:
        events = self.prefix_cache_hits + self.prefix_cache_misses
        return {
            "requests": {
                "added": len(self.added),
                "admitted": len(self.admitted),
                "finished": len(self.finished),
                "aborted": len(self.aborted),
            },
            "rejections": {
                "reasons": dict(self.rejection_reason_counts),
                "queue_timeout": self.rejection_reason_counts["queue_timeout"],
            },
            "scheduler": {
                "prefill_steps": self.prefill_step_count,
                "decode_steps": self.decode_step_count,
                "starvation_protected_steps": self.starvation_protected_steps,
            },
            "prefix_cache": {
                "events": events,
                "hits": self.prefix_cache_hits,
                "misses": self.prefix_cache_misses,
                "exact_hits": self.prefix_cache_exact_hits,
                "partial_hits": self.prefix_cache_partial_hits,
                "saved_prefill_tokens": self.prefix_cache_saved_prefill_tokens,
                "hit_rate": self.prefix_cache_hits / events if events else 0.0,
            },
            "deepseek": {"events": dict(self.deepseek_events)},
        }

    def reset_stats(self) -> None:
        replacement = InMemoryRuntimeObserver(history_limit=self.history_limit)
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
