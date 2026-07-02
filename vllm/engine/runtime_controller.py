# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from typing import Any

import torch

from vllm.engine.errors import RequestRejectedError
from vllm.outputs import RequestOutput


class RuntimeController:
    """Owns Lite runtime control-plane orchestration for one engine instance."""

    def __init__(
        self,
        *,
        scheduler: Any,
        step_scheduler: Any,
        observer: Any,
        backend: Any,
        queue_timeout_s: float,
        lora_registry: Any | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.step_scheduler = step_scheduler
        self.observer = observer
        self.backend = backend
        self.queue_timeout_s = queue_timeout_s
        self.lora_registry = lora_registry
        self._stats_reset_at_unix_s = time.time()
        self._stats_reset_at_monotonic_s = time.perf_counter()

    @torch.inference_mode()
    def step(self) -> list[RequestOutput]:
        if self.scheduler.active_request_count == 0:
            return []

        now = time.perf_counter()
        self._reject_expired_queued_requests(now)

        if self.scheduler.active_request_count == 0:
            return []

        if hasattr(self.step_scheduler, "update_runtime_feedback") and hasattr(
            self.observer, "stats"
        ):
            self.step_scheduler.update_runtime_feedback(self.observer.stats())

        step_plan = self.step_scheduler.build_plan(self.scheduler)
        if not (
            self.scheduler.running_request_count == 1
            and self.scheduler.queued_request_count == 0
        ):
            step_plan = self.backend.maybe_preempt(step_plan, self.scheduler)
        self._admit_requests(step_plan, now)

        if self.scheduler.running_request_count == 0:
            return []

        self.backend.ensure_kv_blocks(step_plan)

        self.observer.on_step_started(step_plan)

        if (
            step_plan.decodes is not None
            and step_plan.decodes.use_fast_path
            and step_plan.prefills is None
        ):
            return self.backend.decode_step_sync(step_plan.decodes.request_ids)

        results: list[RequestOutput] = []
        self.backend.run_prefills(step_plan, results)
        self.backend.run_decodes(step_plan, results)
        return results

    def _reject_expired_queued_requests(self, now: float) -> None:
        expired = self.scheduler.reject_expired_queued_requests(
            now=now,
            max_queue_wait_s=self.queue_timeout_s,
        )
        for request_id, reason, request in expired:
            if self.lora_registry is not None:
                self.lora_registry.on_request_removed(request.lora_id)
            self.observer.on_request_rejected(request_id, reason)
            self.scheduler.publish_exception(
                request_id,
                RequestRejectedError(reason),
            )

    def _admit_requests(self, step_plan: Any, now: float) -> None:
        if step_plan.admissions is None:
            return
        admitted = self.scheduler.admit_specific_requests(
            step_plan.admissions.request_ids,
            admitted_at=now,
        )
        for request_id in admitted:
            req = self.scheduler.get_request(request_id)
            self.backend.maybe_apply_prefix_cache(req)
            queue_wait_s = max(0.0, now - float(req.queued_at or now))
            self.observer.on_request_admitted(
                request_id,
                queue_wait_s,
                str(req.service_class or "latency"),
            )

    def stats(self) -> dict[str, Any]:
        observer_stats: dict[str, Any] = {}
        if hasattr(self.observer, "stats"):
            observer_stats = dict(self.observer.stats())
        observed_at_unix_s = time.time()
        return {
            "observed_at_unix_s": observed_at_unix_s,
            "stats_window_started_at_unix_s": self._stats_reset_at_unix_s,
            "stats_window_elapsed_s": max(
                0.0, time.perf_counter() - self._stats_reset_at_monotonic_s
            ),
            "queue_timeout_s": self.queue_timeout_s,
            "profile": self._profile_stats(),
            "scheduler": {
                "active_request_count": int(
                    getattr(self.scheduler, "active_request_count", 0)
                ),
                "running_request_count": int(
                    getattr(self.scheduler, "running_request_count", 0)
                ),
                "queued_request_count": int(
                    getattr(self.scheduler, "queued_request_count", 0)
                ),
                "available_slots": int(getattr(self.scheduler, "available_slots", 0)),
            },
            "observer": observer_stats,
            "backend": dict(self.backend.stats()),
            "lora": dict(self.lora_registry.stats())
            if self.lora_registry is not None
            else {},
        }

    def _profile_stats(self) -> dict[str, object]:
        runtime_config = getattr(self.scheduler, "runtime_config", None)
        profile = (
            getattr(runtime_config, "profile", None)
            if runtime_config is not None
            else None
        )
        if profile is not None:
            out = dict(profile.stats())
            out["kv_cache_dtype"] = getattr(
                runtime_config, "kv_cache_dtype", out.get("kv_cache_dtype")
            )
            out["block_size"] = getattr(
                runtime_config, "block_size", out.get("block_size")
            )
            out["kv_max_model_len"] = getattr(
                runtime_config, "kv_max_model_len", out.get("kv_max_model_len")
            )
            out["kv_max_active_requests"] = getattr(
                runtime_config,
                "kv_max_active_requests",
                out.get("kv_max_active_requests"),
            )
            out["fusion_level"] = getattr(
                runtime_config, "fusion_level", out.get("fusion_level")
            )
            return out
        if hasattr(self.backend, "profile_stats"):
            return dict(self.backend.profile_stats())
        return {}

    def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
        if hasattr(self.observer, "reset_stats"):
            self.observer.reset_stats()
        self.backend.reset_stats(clear_prefix_cache=clear_prefix_cache)
        self._stats_reset_at_unix_s = time.time()
        self._stats_reset_at_monotonic_s = time.perf_counter()
