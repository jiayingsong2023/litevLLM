# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import torch

from vllm.engine.errors import ExecutionStepError
from vllm.engine.prefix_cache import PrefixCache, PrefixCacheEntry
from vllm.engine.request_state import RequestState
from vllm.engine.step_plan import StepPlan
from vllm.logger import init_logger
from vllm.outputs import RequestOutput

logger = init_logger(__name__)


class LiteSingleGpuBackend:
    """Single-GPU execution backend for the lite runtime."""

    def __init__(
        self,
        *,
        scheduler: Any,
        observer: Any,
        prefill_executor: Any,
        decode_executor: Any,
        sampling_driver: Any | None,
        output_coordinator: Any | None,
        kv_block_manager: Any,
        lora_registry: Any | None = None,
        max_prefix_cache_entries: int = 8,
        min_prefix_cache_partial_hit_tokens: int = 1,
        preemption_mode: str = "defer_prefill",
        preemption_min_backlog: int = 1,
        preemption_min_decodes: int = 1,
        preemption_max_queue_wait_s: float = 0.0,
        preemptible_service_classes: set[str] | None = None,
        preempt_multimodal_prefills: bool = False,
        preempt_multimodal_max_queue_wait_s: float = 0.0,
        multimodal_prefix_cache_protect_threshold: float = 0.0,
        gpu_greedy_sampling: bool = False,
        gpu_greedy_max_tokens_only: bool = False,
        gpu_greedy_bypass_cpu_policies: bool = False,
        gpu_greedy_ignore_eos: bool = False,
    ) -> None:
        self.scheduler = scheduler
        self.observer = observer
        self.prefill_executor = prefill_executor
        self.decode_executor = decode_executor
        self.sampling_driver = sampling_driver
        self.output_coordinator = output_coordinator
        self.kv_block_manager = kv_block_manager
        self.lora_registry = lora_registry
        self.block_size = self.kv_block_manager.block_size
        self.num_blocks_per_seq = self.kv_block_manager.num_blocks_per_seq
        self.prefix_cache = PrefixCache(max_entries=max_prefix_cache_entries)
        self.min_prefix_cache_partial_hit_tokens = max(
            1, int(min_prefix_cache_partial_hit_tokens)
        )
        self.prefix_cache_materialized_hits = 0
        self.prefix_cache_materialized_exact_hits = 0
        self.prefix_cache_materialized_partial_hits = 0
        self.prefix_cache_materialized_saved_prefill_tokens = 0
        self.preemption_mode = str(preemption_mode or "defer_prefill")
        self.preemption_min_backlog = max(1, int(preemption_min_backlog))
        self.preemption_min_decodes = max(1, int(preemption_min_decodes))
        self.preemption_max_queue_wait_s = max(0.0, float(preemption_max_queue_wait_s))
        self.preemptible_service_classes = {
            str(service_class).strip()
            for service_class in (preemptible_service_classes or set())
            if str(service_class).strip()
        }
        self.preempt_multimodal_prefills = bool(preempt_multimodal_prefills)
        self.preempt_multimodal_max_queue_wait_s = max(
            0.0, float(preempt_multimodal_max_queue_wait_s)
        )
        self.multimodal_prefix_cache_protect_threshold = max(
            0.0, float(multimodal_prefix_cache_protect_threshold)
        )
        self.gpu_greedy_sampling = bool(gpu_greedy_sampling)
        self.gpu_greedy_max_tokens_only = bool(gpu_greedy_max_tokens_only)
        self.gpu_greedy_bypass_cpu_policies = bool(gpu_greedy_bypass_cpu_policies)
        self.gpu_greedy_ignore_eos = bool(gpu_greedy_ignore_eos)

    def maybe_apply_prefix_cache(self, request_state: RequestState) -> None:
        cache_key = self._prefix_cache_key(request_state)
        if not cache_key:
            return None

        entry = request_state._prefix_cache_entry
        prefix_len = int(request_state._prefix_cache_hit_len or 0)
        if entry is None:
            entry, cache_prefix_len = self.prefix_cache.get_longest_prefix(cache_key)
            prefix_len = self._request_prefix_len_from_cache_prefix(cache_prefix_len)
            exact_hit = prefix_len == len(request_state.input_ids)
            if (
                entry is None
                or prefix_len <= 0
                or (
                    not exact_hit
                    and prefix_len < self.min_prefix_cache_partial_hit_tokens
                )
            ):
                self._record_prefix_cache_event(
                    request_state,
                    hit=False,
                    exact=False,
                    prefix_len=0,
                    saved_prefill_tokens=0,
                )
                return None
            request_state._prefix_cache_entry = entry
            request_state._prefix_cache_hit_len = prefix_len
            self._record_prefix_cache_event(
                request_state,
                hit=True,
                exact=exact_hit,
                prefix_len=prefix_len,
                saved_prefill_tokens=prefix_len,
            )

        if request_state.slot_idx is None or request_state._prefix_cache_applied:
            return None

        self._ensure_runtime_ready()
        self._materialize_prefix_cache_entry(request_state, entry, prefix_len)
        request_state._prefix_cache_applied = True
        return None

    def maybe_preempt(self, step_plan, scheduler):
        # Soft preemption: when decode work exists and queued backlog remains,
        # defer selected prefills for this step instead of partially executing them.
        if not hasattr(step_plan, "prefills") or not hasattr(step_plan, "decodes"):
            return step_plan
        if self.preemption_mode == "off":
            return step_plan
        if self.preemption_mode != "defer_prefill":
            return step_plan
        if (
            step_plan.prefills is not None
            and step_plan.decodes is not None
            and not getattr(step_plan.metrics, "prefill_starvation_protected", False)
            and getattr(scheduler, "queued_request_count", 0)
            >= self.preemption_min_backlog
            and len(step_plan.decodes.request_ids) >= self.preemption_min_decodes
            and not self._queue_wait_blocks_preemption(step_plan)
            and self._multimodal_prefills_are_preemptible(step_plan, scheduler)
            and self._prefills_are_preemptible(step_plan, scheduler)
        ):
            preempted_multimodal = 0
            if hasattr(scheduler, "get_request"):
                preempted_multimodal = sum(
                    1
                    for rid in step_plan.prefills.request_ids
                    if self._is_multimodal_request(scheduler.get_request(rid))
                )
            self.observer.on_preemption_event(
                preempted_prefill_requests=len(step_plan.prefills.request_ids),
                queued_backlog=int(getattr(scheduler, "queued_request_count", 0)),
                multimodal_prefill_requests=preempted_multimodal,
            )
            return StepPlan(
                admissions=step_plan.admissions,
                prefills=None,
                decodes=step_plan.decodes,
                step_token_budget=step_plan.step_token_budget,
                metrics=step_plan.metrics,
            )
        return step_plan

    def decode_step_sync(self, request_ids: list[str]) -> list[RequestOutput]:
        self._ensure_runtime_ready()
        logits, _ = self.decode_executor.execute_sync_fast(request_ids, self.scheduler)
        if self._can_use_gpu_greedy_decode(request_ids):
            return self._decode_step_gpu_greedy(request_ids, logits)
        results: list[RequestOutput] = []
        requests = [self.scheduler.get_request(rid) for rid in request_ids]
        next_tokens = self.sampling_driver.sample_batch_tokens(
            logits[:, -1, :], requests
        )
        for rid, req, token in zip(request_ids, requests, next_tokens):
            req.generated_ids.append(token)
            req.seq_len += 1
            self._process_completion(rid, token, results)
        return results

    def run_prefills(self, step_plan, results: list[RequestOutput]) -> None:
        self._ensure_runtime_ready()
        if step_plan.prefills is None:
            return
        try:
            logits, _req_dicts_prefill, is_last_chunk_flags = (
                self.prefill_executor.execute(
                    step_plan.prefills.request_ids,
                    self.scheduler,
                    step_plan.prefills.chunk_len,
                )
            )

            finished_indices: list[int] = []
            finished_rids: list[str] = []
            finished_reqs: list[RequestState] = []
            finished_logits_list: list[torch.Tensor] = []

            for i, rid in enumerate(step_plan.prefills.request_ids):
                req = self.scheduler.get_request(rid)
                req.seq_len += step_plan.prefills.chunk_len
                if not is_last_chunk_flags[i]:
                    continue
                self._store_prefix_cache_entry(rid, req, logits[i, -1, :])
                finished_indices.append(i)
                finished_rids.append(rid)
                finished_reqs.append(req)
                finished_logits_list.append(logits[i, -1, :])

            if finished_reqs:
                stacked_logits = torch.stack(finished_logits_list)
                next_tokens = self.sampling_driver.sample_batch_tokens(
                    stacked_logits, finished_reqs
                )
                for rid, req, next_token in zip(
                    finished_rids, finished_reqs, next_tokens
                ):
                    req.generated_ids.append(next_token)
                    req.is_prefill = False
                    self._process_completion(rid, next_token, results)

            self.observer.on_prefill_executed(step_plan.prefills, len(results))
        except Exception as e:
            error = ExecutionStepError(f"prefill failed: {e}")
            logger.exception("LiteEngine prefill execution error: %s", error)
            for rid in step_plan.prefills.request_ids:
                self._free_request(rid)
            raise error from e

    def run_decodes(self, step_plan, results: list[RequestOutput]) -> None:
        self._ensure_runtime_ready()
        if step_plan.decodes is None:
            return
        try:
            logits, _req_dicts = self.decode_executor.execute_batch(
                step_plan.decodes.request_ids, self.scheduler
            )
            requests = [
                self.scheduler.get_request(rid) for rid in step_plan.decodes.request_ids
            ]
            next_tokens = self.sampling_driver.sample_batch_tokens(
                logits[:, -1, :], requests
            )
            for rid, req_i, token in zip(
                step_plan.decodes.request_ids, requests, next_tokens
            ):
                req_i.generated_ids.append(token)
                req_i.seq_len += 1
                self._process_completion(rid, token, results)
            self.observer.on_decode_executed(step_plan.decodes, len(results))
        except Exception as e:
            error = ExecutionStepError(f"decode failed: {e}")
            logger.exception("LiteEngine decode execution error: %s", error)
            for rid in step_plan.decodes.request_ids:
                self._free_request(rid)
            raise error from e

    def _can_use_gpu_greedy_decode(self, request_ids: list[str]) -> bool:
        if not self.gpu_greedy_sampling:
            return False
        if not request_ids:
            return False
        # This fast path intentionally handles the latency benchmark shape first:
        # greedy, max-token-bounded decode with no per-token CPU policy work.
        if not self.gpu_greedy_max_tokens_only:
            return False
        bypass_cpu_policies = self.gpu_greedy_bypass_cpu_policies
        ignore_eos = self.gpu_greedy_ignore_eos
        for rid in request_ids:
            req = self.scheduler.get_request(rid)
            sp = req.sampling_params
            if float(getattr(sp, "temperature", 0.0) or 0.0) > 1e-6:
                return False
            if abs(float(getattr(sp, "repetition_penalty", 1.0) or 1.0) - 1.0) > 1e-12:
                return False
            if abs(float(getattr(sp, "frequency_penalty", 0.0) or 0.0)) > 1e-12:
                return False
            if abs(float(getattr(sp, "presence_penalty", 0.0) or 0.0)) > 1e-12:
                return False
            if not ignore_eos:
                stop_token_ids = getattr(sp, "stop_token_ids", None)
                if stop_token_ids:
                    return False
                if self.sampling_driver is not None:
                    try:
                        if self.sampling_driver.completion_eos_ids(req):
                            return False
                    except Exception:
                        return False
            if req.structured_output_constraint is not None:
                return False
            if not bypass_cpu_policies:
                if req.anti_template_token_ids:
                    return False
                if (
                    req.capital_question_bias_token_ids
                    or req.is_chinese_capital_question
                ):
                    return False
        return True

    def _decode_step_gpu_greedy(
        self,
        request_ids: list[str],
        logits: torch.Tensor,
    ) -> list[RequestOutput]:
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
        results: list[RequestOutput] = []
        for i, rid in enumerate(request_ids):
            req = self.scheduler.get_request(rid)
            token_t = next_tokens[i].detach()
            req._last_token_tensor = token_t
            pending = req._pending_token_tensors
            pending.append(token_t)
            req.seq_len += 1

            now = time.perf_counter()
            if req.first_token_at is None:
                req.first_token_at = now
                admitted_at = float(req.admitted_at or now)
                self.observer.on_first_token(rid, max(0.0, now - admitted_at))

            generated_len = len(req.generated_ids) + len(pending)
            max_tok = int(req.sampling_params.max_tokens or 16)
            if generated_len < max_tok:
                continue

            # One synchronization at request completion replaces one sync per
            # token. The generated list is restored before finalize_step so
            # output text and token_ids keep the normal API contract.
            pending_tokens = torch.stack(pending).to(device="cpu").tolist()
            req.generated_ids.extend(int(t) for t in pending_tokens)
            req._pending_token_tensors = []
            req._last_token_tensor = None
            last_token = (
                int(pending_tokens[-1])
                if pending_tokens
                else int(req.generated_ids[-1])
            )
            self._process_completion(rid, last_token, results)
        return results

    def stats(self) -> dict[str, object]:
        materialized_hits = self.prefix_cache_materialized_hits
        multimodal_processor = getattr(
            self.prefill_executor, "multimodal_processor", None
        )
        return {
            "backend_type": "lite_single_gpu",
            "block_size": self.block_size,
            "num_blocks_per_seq": self.num_blocks_per_seq,
            "num_kv_layers": self.kv_block_manager.num_layers,
            "min_prefix_cache_partial_hit_tokens": (
                self.min_prefix_cache_partial_hit_tokens
            ),
            "preemption_mode": self.preemption_mode,
            "preemption_min_backlog": self.preemption_min_backlog,
            "preemption_min_decodes": self.preemption_min_decodes,
            "preemption_max_queue_wait_s": self.preemption_max_queue_wait_s,
            "preemptible_service_classes": sorted(self.preemptible_service_classes),
            "preempt_multimodal_prefills": self.preempt_multimodal_prefills,
            "preempt_multimodal_max_queue_wait_s": (
                self.preempt_multimodal_max_queue_wait_s
            ),
            "multimodal_prefix_cache_protect_threshold": (
                self.multimodal_prefix_cache_protect_threshold
            ),
            "gpu_greedy_sampling": self.gpu_greedy_sampling,
            "gpu_greedy_max_tokens_only": self.gpu_greedy_max_tokens_only,
            "gpu_greedy_bypass_cpu_policies": self.gpu_greedy_bypass_cpu_policies,
            "gpu_greedy_ignore_eos": self.gpu_greedy_ignore_eos,
            "prefix_cache_materialized_hits": materialized_hits,
            "prefix_cache_materialized_exact_hits": (
                self.prefix_cache_materialized_exact_hits
            ),
            "prefix_cache_materialized_partial_hits": (
                self.prefix_cache_materialized_partial_hits
            ),
            "prefix_cache_materialized_saved_prefill_tokens": (
                self.prefix_cache_materialized_saved_prefill_tokens
            ),
            "prefix_cache_avg_saved_prefill_tokens_per_materialized_hit": (
                self.prefix_cache_materialized_saved_prefill_tokens / materialized_hits
                if materialized_hits
                else 0.0
            ),
            "prefix_cache": self.prefix_cache.stats(),
            "multimodal": (
                dict(multimodal_processor.stats())
                if multimodal_processor is not None
                and hasattr(multimodal_processor, "stats")
                else {}
            ),
        }

    def reset_stats(self, *, clear_prefix_cache: bool = False) -> None:
        self.prefix_cache_materialized_hits = 0
        self.prefix_cache_materialized_exact_hits = 0
        self.prefix_cache_materialized_partial_hits = 0
        self.prefix_cache_materialized_saved_prefill_tokens = 0
        multimodal_processor = getattr(
            self.prefill_executor, "multimodal_processor", None
        )
        if multimodal_processor is not None and hasattr(
            multimodal_processor, "reset_stats"
        ):
            multimodal_processor.reset_stats()
        if clear_prefix_cache:
            self.prefix_cache.clear()

    def _process_completion(
        self,
        request_id: str,
        next_token: int,
        results: list[RequestOutput],
    ) -> None:
        req = self.scheduler.get_request(request_id)
        now = time.perf_counter()
        if req.first_token_at is None:
            req.first_token_at = now
            admitted_at = float(req.admitted_at or now)
            self.observer.on_first_token(request_id, max(0.0, now - admitted_at))
        out = self.output_coordinator.finalize_step(request_id, req, next_token)
        self.scheduler.publish_output(request_id, out)
        results.append(out)

        if req.finished:
            finish_reason = "completed"
            if next_token in self.sampling_driver.completion_eos_ids(req):
                finish_reason = "eos"
            elif len(req.generated_ids) >= int(req.sampling_params.max_tokens or 16):
                finish_reason = "max_tokens"
            self.observer.on_request_finished(request_id, finish_reason)
            self._free_request(request_id)

    def _free_request(self, request_id: str) -> None:
        request = self.scheduler.free_request(request_id)
        if request is not None and self.lora_registry is not None:
            self.lora_registry.on_request_removed(request.lora_id)

    def _ensure_runtime_ready(self) -> None:
        if self.sampling_driver is None or self.output_coordinator is None:
            raise RuntimeError(
                "Execution backend is not ready: sampling/output is unset"
            )

    def _record_prefix_cache_event(
        self,
        request_state: RequestState,
        *,
        hit: bool,
        exact: bool,
        prefix_len: int,
        saved_prefill_tokens: int,
    ) -> None:
        if request_state._prefix_cache_observed:
            return
        request_state._prefix_cache_observed = True
        request_id = request_state.request_id or "<unknown>"
        self.observer.on_prefix_cache_event(
            request_id,
            hit=hit,
            exact=exact,
            prefix_len=prefix_len,
            saved_prefill_tokens=saved_prefill_tokens,
            is_multimodal=self._is_multimodal_request(request_state),
        )

    def _queue_wait_blocks_preemption(self, step_plan: StepPlan) -> bool:
        threshold = self.preemption_max_queue_wait_s
        if threshold <= 0:
            return False
        return (
            float(getattr(step_plan.metrics, "queued_p95_wait_s", 0.0) or 0.0)
            >= threshold
        )

    def _prefills_are_preemptible(self, step_plan: StepPlan, scheduler: Any) -> bool:
        if not self.preemptible_service_classes or step_plan.prefills is None:
            return True
        return all(
            str(scheduler.get_request(rid).service_class or "latency")
            in self.preemptible_service_classes
            for rid in step_plan.prefills.request_ids
        )

    def _multimodal_prefills_are_preemptible(
        self,
        step_plan: StepPlan,
        scheduler: Any,
    ) -> bool:
        if step_plan.prefills is None:
            return True
        if not hasattr(scheduler, "get_request"):
            return True
        has_multimodal_prefill = any(
            self._is_multimodal_request(scheduler.get_request(rid))
            for rid in step_plan.prefills.request_ids
        )
        if not has_multimodal_prefill:
            return True
        if self._protect_multimodal_prefix_prefills(step_plan):
            self.observer.on_multimodal_preemption_guard(
                protected_prefill_requests=len(step_plan.prefills.request_ids),
                prefix_cache_hit_rate=float(
                    getattr(step_plan.metrics, "multimodal_prefix_cache_hit_rate", 0.0)
                    or 0.0
                ),
            )
            return False
        if self.preempt_multimodal_prefills:
            if self.preempt_multimodal_max_queue_wait_s <= 0:
                return True
            return (
                float(
                    getattr(step_plan.metrics, "queued_multimodal_p95_wait_s", 0.0)
                    or 0.0
                )
                >= self.preempt_multimodal_max_queue_wait_s
            )
        return False

    def _protect_multimodal_prefix_prefills(self, step_plan: StepPlan) -> bool:
        threshold = self.multimodal_prefix_cache_protect_threshold
        if threshold <= 0:
            return False
        return (
            float(
                getattr(step_plan.metrics, "multimodal_prefix_cache_hit_rate", 0.0)
                or 0.0
            )
            >= threshold
        )

    @staticmethod
    def _is_multimodal_request(request: RequestState) -> bool:
        return bool(
            request.is_multimodal or (request.multi_modal_data or {}).get("image")
        )

    def _store_prefix_cache_entry(
        self,
        request_id: str,
        request_state: RequestState,
        last_prompt_logits: torch.Tensor,
    ) -> None:
        del request_id
        prompt_len = int(request_state.seq_len or 0)
        slot_idx = request_state.slot_idx
        if prompt_len <= 0 or slot_idx is None:
            return

        key = self._prefix_cache_key(request_state)
        if not key:
            return

        entry = self.kv_block_manager.capture_prefix_entry(
            key=key,
            slot_idx=int(slot_idx),
            prompt_len=prompt_len,
            last_prompt_logits=last_prompt_logits,
        )
        self.prefix_cache.put(entry)

    def _materialize_prefix_cache_entry(
        self,
        request_state: RequestState,
        entry: PrefixCacheEntry,
        prefix_len: int,
    ) -> None:
        prefix_len = max(0, min(int(prefix_len), int(entry.prompt_len)))
        if prefix_len <= 0:
            return
        slot_idx = int(request_state.slot_idx)
        self.kv_block_manager.materialize_prefix_entry(
            slot_idx=slot_idx,
            entry=entry,
            prefix_len=prefix_len,
        )

        request_state.seq_len = prefix_len
        request_state.is_prefill = prefix_len < len(request_state.input_ids)
        self.prefix_cache_materialized_hits += 1
        self.prefix_cache_materialized_saved_prefill_tokens += prefix_len
        if prefix_len != len(request_state.input_ids):
            self.prefix_cache_materialized_partial_hits += 1
            return
        self.prefix_cache_materialized_exact_hits += 1

        request_id = request_state.request_id
        next_token = self.sampling_driver.sample_next_token(
            entry.last_prompt_logits,
            request_state,
        )
        request_state.generated_ids.append(next_token)
        results: list[RequestOutput] = []
        self._process_completion(request_id, next_token, results)

    def _prefix_cache_key(self, request_state: RequestState) -> tuple[int, ...]:
        input_ids = tuple(int(tok) for tok in request_state.input_ids)
        if not input_ids:
            return ()
        namespace = self._prefix_cache_namespace(request_state)
        return (namespace, *input_ids)

    def _prefix_cache_namespace(self, request_state: RequestState) -> int:
        mm_data = request_state.multi_modal_data or {}
        images = mm_data.get("image") or []
        if not images:
            return -1
        image_urls = [
            str(item.get("image") or "")
            for item in images
            if isinstance(item, dict) and str(item.get("image") or "")
        ]
        if not image_urls:
            return -1
        digest = hashlib.sha1(
            json.dumps(image_urls, ensure_ascii=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).digest()
        return -int.from_bytes(digest[:8], byteorder="big", signed=False) - 2

    @staticmethod
    def _request_prefix_len_from_cache_prefix(cache_prefix_len: int) -> int:
        return max(0, int(cache_prefix_len) - 1)
