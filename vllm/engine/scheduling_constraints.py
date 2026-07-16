# SPDX-License-Identifier: Apache-2.0
from typing import Any

from vllm.engine.scheduling_helpers import (
    is_multimodal,
    is_multimodal_lora,
    lora_adapter_key,
    max_abs_share_gap,
    normalize_quotas,
    normalized_share_map,
    rotate_candidates,
    service_class_priority,
    share_gap_map,
)


class ServiceClassSelector:
    """Encapsulates priority and weighted service class selection and cursor updates."""

    def select_weighted_requests(
        self,
        *,
        scheduler: Any,
        step_scheduler: Any,
        request_ids: list[str],
        limit: int,
        cursor_attr: str,
        prefer_short_prompts: bool,
        intra_class_rotate: bool = False,
        quotas: dict[str, int] | None = None,
    ) -> list[str]:
        if limit <= 0 or not request_ids:
            return []
        groups: dict[str, list[str]] = {}
        request_meta: dict[str, tuple[str, int, float]] = {}
        for rid in request_ids:
            request = scheduler.get_request(rid)
            service_class = str(request.service_class or "latency")
            request_meta[rid] = (
                service_class,
                len(request.input_ids),
                float(request.queued_at or 0.0),
            )
            groups.setdefault(service_class, []).append(rid)
        for service_class, grouped_ids in groups.items():
            groups[service_class] = sorted(
                grouped_ids,
                key=lambda rid: (
                    request_meta[rid][1] if prefer_short_prompts else 0,
                    request_meta[rid][2],
                    rid,
                ),
            )
        service_classes = sorted(
            groups,
            key=lambda service_class: (
                service_class_priority(service_class),
                service_class,
            ),
        )
        if not service_classes:
            return []
        selected: list[str] = []
        cursor = getattr(step_scheduler, cursor_attr) % len(service_classes)
        rotated_classes = rotate_candidates(service_classes, cursor)
        requested_quotas = normalize_quotas(quotas or {})
        for service_class in rotated_classes:
            if len(selected) >= limit:
                break
            candidates = groups.get(service_class, [])
            if not candidates:
                continue
            quota = requested_quotas.get(service_class, 0)
            if quota <= 0:
                continue
            picked = self._take_candidates(
                step_scheduler=step_scheduler,
                candidates=candidates,
                take=min(quota, limit - len(selected), len(candidates)),
                intra_class_rotate=intra_class_rotate,
            )
            if not picked:
                continue
            selected.extend(picked)
            picked_set = set(picked)
            groups[service_class] = [
                rid for rid in groups[service_class] if rid not in picked_set
            ]
        while len(selected) < limit:
            progressed = False
            for service_class in rotated_classes:
                candidates = groups.get(service_class, [])
                if not candidates:
                    continue
                quota = step_scheduler.service_class_weights.get(service_class, 1)
                if quota <= 0:
                    continue
                picked = self._take_candidates(
                    step_scheduler=step_scheduler,
                    candidates=candidates,
                    take=min(quota, limit - len(selected), len(candidates)),
                    intra_class_rotate=intra_class_rotate,
                )
                if not picked:
                    continue
                selected.extend(picked)
                picked_set = set(picked)
                groups[service_class] = [
                    rid for rid in groups[service_class] if rid not in picked_set
                ]
                progressed = True
                if len(selected) >= limit:
                    break
            if not progressed:
                break
            cursor = (cursor + 1) % len(service_classes)
            rotated_classes = rotate_candidates(service_classes, cursor)
        setattr(
            step_scheduler,
            cursor_attr,
            (getattr(step_scheduler, cursor_attr) + 1) % max(1, len(service_classes)),
        )
        return selected

    def _take_candidates(
        self,
        *,
        step_scheduler: Any,
        candidates: list[str],
        take: int,
        intra_class_rotate: bool,
    ) -> list[str]:
        if take <= 0 or not candidates:
            return []
        selected_candidates = candidates
        if intra_class_rotate and len(candidates) > 1:
            selected_candidates = rotate_candidates(
                candidates,
                step_scheduler._decode_rr_cursor,
            )
        picked = selected_candidates[:take]
        if intra_class_rotate and picked:
            step_scheduler._decode_rr_cursor = (
                step_scheduler._decode_rr_cursor + len(picked)
            ) % max(1, len(candidates))
        return picked


class LoRAConstraint:
    """Applies LoRA batch limits, rotation, and fairness accounting."""

    def shape_lora_batch(
        self,
        *,
        scheduler: Any,
        step_scheduler: Any,
        request_ids: list[str],
        baseline_counts: dict[str, int],
        max_lora_adapters: int,
        cursor_attr: str,
    ) -> tuple[list[str], int, dict[str, float], bool, bool]:
        if not request_ids:
            return [], max_lora_adapters, {}, False, False
        baseline_share = normalized_share_map(baseline_counts)
        pre_limit_gap = share_gap_map(
            normalized_share_map(self.count_lora_adapters(scheduler, request_ids)),
            baseline_share,
        )
        effective_limit = max_lora_adapters
        relaxed = False
        tightened = False
        if effective_limit > 0:
            adapter_count = len(self.lora_adapters_for_ids(scheduler, request_ids))
            max_gap = max_abs_share_gap(pre_limit_gap)
            if (
                step_scheduler.lora_fairness_relax_threshold > 0
                and max_gap >= step_scheduler.lora_fairness_relax_threshold
            ):
                effective_limit = min(
                    adapter_count,
                    effective_limit + step_scheduler.lora_limit_relax_delta,
                )
                relaxed = effective_limit > max_lora_adapters
            elif (
                step_scheduler.lora_locality_tighten_threshold > 0
                and max_gap <= step_scheduler.lora_locality_tighten_threshold
                and effective_limit > 1
            ):
                effective_limit = max(
                    1,
                    effective_limit - step_scheduler.lora_limit_tighten_delta,
                )
                tightened = effective_limit < max_lora_adapters
            request_ids = self.apply_lora_adapter_batch_limit(
                scheduler=scheduler,
                step_scheduler=step_scheduler,
                request_ids=request_ids,
                max_lora_adapters=effective_limit,
                cursor_attr=cursor_attr,
            )
        fairness_gap = share_gap_map(
            normalized_share_map(self.count_lora_adapters(scheduler, request_ids)),
            baseline_share,
        )
        return request_ids, effective_limit, fairness_gap, relaxed, tightened

    def apply_lora_adapter_batch_limit(
        self,
        *,
        scheduler: Any,
        step_scheduler: Any,
        request_ids: list[str],
        max_lora_adapters: int,
        cursor_attr: str,
    ) -> list[str]:
        if max_lora_adapters <= 0 or len(request_ids) <= 1:
            return request_ids
        adapter_order = self.lora_adapters_for_ids(scheduler, request_ids)
        if len(adapter_order) <= max_lora_adapters:
            return request_ids
        cursor = getattr(step_scheduler, cursor_attr) % max(1, len(adapter_order))
        allowed_adapters = set(
            rotate_candidates(adapter_order, cursor)[:max_lora_adapters]
        )
        setattr(
            step_scheduler,
            cursor_attr,
            (getattr(step_scheduler, cursor_attr) + 1) % max(1, len(adapter_order)),
        )
        shaped = [
            rid
            for rid in request_ids
            if lora_adapter_key(scheduler.get_request(rid)) in allowed_adapters
        ]
        return shaped or request_ids[:1]

    def lora_adapters_for_ids(
        self, scheduler: Any, request_ids: list[str]
    ) -> list[str]:
        adapters = {lora_adapter_key(scheduler.get_request(rid)) for rid in request_ids}
        return sorted(adapters)

    def count_lora_adapters(
        self, scheduler: Any, request_ids: list[str]
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rid in request_ids:
            key = lora_adapter_key(scheduler.get_request(rid))
            counts[key] = counts.get(key, 0) + 1
        return counts


class MultiModalConstraint:
    """Applies multimodal limits, batch shaping, and prefix-cache feedback."""

    def shape_multimodal_prefill_batch(
        self,
        *,
        scheduler: Any,
        step_scheduler: Any,
        request_ids: list[str],
        candidate_prefills: list[str],
    ) -> tuple[list[str], int, bool, bool, bool]:
        effective_limit = step_scheduler.max_prefill_multimodal_requests_per_batch
        relaxed = False
        tightened = False
        if effective_limit > 0:
            candidate_mm_count = self.count_multimodal_requests(
                scheduler, candidate_prefills
            )
            hit_rate = step_scheduler._multimodal_prefix_cache_hit_rate_feedback
            if (
                step_scheduler.multimodal_prefix_cache_relax_threshold > 0
                and hit_rate <= step_scheduler.multimodal_prefix_cache_relax_threshold
            ):
                effective_limit = min(
                    candidate_mm_count,
                    effective_limit
                    + step_scheduler.multimodal_prefill_limit_relax_delta,
                )
                relaxed = (
                    effective_limit
                    > step_scheduler.max_prefill_multimodal_requests_per_batch
                )
            elif (
                step_scheduler.multimodal_prefix_cache_tighten_threshold > 0
                and hit_rate >= step_scheduler.multimodal_prefix_cache_tighten_threshold
                and effective_limit > 1
            ):
                effective_limit = max(
                    1,
                    effective_limit
                    - step_scheduler.multimodal_prefill_limit_tighten_delta,
                )
                tightened = (
                    effective_limit
                    < step_scheduler.max_prefill_multimodal_requests_per_batch
                )
        shaped = self.apply_multimodal_request_limit(
            scheduler=scheduler,
            step_scheduler=step_scheduler,
            request_ids=request_ids,
            max_multimodal_requests=effective_limit,
            cursor_attr="_prefill_multimodal_cursor",
        )
        triggered = shaped != request_ids
        return shaped, effective_limit, triggered, relaxed, tightened

    def apply_multimodal_request_limit(
        self,
        *,
        scheduler: Any,
        step_scheduler: Any,
        request_ids: list[str],
        max_multimodal_requests: int,
        cursor_attr: str,
    ) -> list[str]:
        if max_multimodal_requests <= 0 or len(request_ids) <= 1:
            return request_ids
        multimodal_ids = [
            rid for rid in request_ids if is_multimodal(scheduler.get_request(rid))
        ]
        if len(multimodal_ids) <= max_multimodal_requests:
            return request_ids
        cursor = getattr(step_scheduler, cursor_attr) % max(1, len(multimodal_ids))
        allowed_multimodal = set(
            rotate_candidates(multimodal_ids, cursor)[:max_multimodal_requests]
        )
        setattr(
            step_scheduler,
            cursor_attr,
            (getattr(step_scheduler, cursor_attr) + 1) % max(1, len(multimodal_ids)),
        )
        shaped = [
            rid
            for rid in request_ids
            if (not is_multimodal(scheduler.get_request(rid)))
            or rid in allowed_multimodal
        ]
        return shaped or request_ids[:1]

    def count_multimodal_requests(self, scheduler: Any, request_ids: list[str]) -> int:
        return sum(
            1 for rid in request_ids if is_multimodal(scheduler.get_request(rid))
        )


class MultiModalLoRAConstraint:
    """Applies multimodal LoRA limits and adaptive state updates."""

    def apply_multimodal_lora_request_limit(
        self,
        *,
        scheduler: Any,
        step_scheduler: Any,
        request_ids: list[str],
        max_multimodal_lora_requests: int,
        cursor_attr: str,
        candidate_request_ids: list[str] | None = None,
    ) -> tuple[list[str], int, bool, bool, bool, bool, bool, float]:
        if cursor_attr == "_prefill_multimodal_lora_cursor":
            base_limit = step_scheduler.max_prefill_multimodal_lora_requests_per_batch
        elif cursor_attr == "_decode_multimodal_lora_cursor":
            base_limit = step_scheduler.max_decode_multimodal_lora_requests_per_batch
        else:
            base_limit = max_multimodal_lora_requests
        relaxed = (
            max_multimodal_lora_requests > 0
            and max_multimodal_lora_requests > base_limit
        )
        tightened = (
            max_multimodal_lora_requests > 0
            and base_limit > 0
            and max_multimodal_lora_requests < base_limit
        )
        (
            effective_limit,
            relaxed_by_fairness,
            tightened_by_locality,
            max_fairness_gap,
        ) = (
            self._effective_prefill_multimodal_lora_limit(
                scheduler=scheduler,
                step_scheduler=step_scheduler,
                candidate_request_ids=candidate_request_ids or request_ids,
                current_request_ids=request_ids,
            )
            if cursor_attr == "_prefill_multimodal_lora_cursor"
            else self._effective_decode_multimodal_lora_limit(
                scheduler=scheduler,
                step_scheduler=step_scheduler,
                candidate_request_ids=candidate_request_ids or request_ids,
                current_request_ids=request_ids,
            )
            if cursor_attr == "_decode_multimodal_lora_cursor"
            else (max_multimodal_lora_requests, False, False, 0.0)
        )
        if cursor_attr in {
            "_prefill_multimodal_lora_cursor",
            "_decode_multimodal_lora_cursor",
        }:
            max_multimodal_lora_requests = effective_limit
            relaxed = max_multimodal_lora_requests > base_limit
            tightened = (
                max_multimodal_lora_requests > 0
                and base_limit > 0
                and max_multimodal_lora_requests < base_limit
            )
        if max_multimodal_lora_requests <= 0 or len(request_ids) <= 1:
            return (
                request_ids,
                max(0, int(max_multimodal_lora_requests)),
                False,
                relaxed,
                tightened,
                relaxed_by_fairness,
                tightened_by_locality,
                max_fairness_gap,
            )
        multimodal_lora_ids = [
            rid for rid in request_ids if is_multimodal_lora(scheduler.get_request(rid))
        ]
        if len(multimodal_lora_ids) <= max_multimodal_lora_requests:
            return (
                request_ids,
                max_multimodal_lora_requests,
                False,
                relaxed,
                tightened,
                relaxed_by_fairness,
                tightened_by_locality,
                max_fairness_gap,
            )
        cursor = getattr(step_scheduler, cursor_attr) % max(1, len(multimodal_lora_ids))
        allowed_multimodal_lora = set(
            rotate_candidates(multimodal_lora_ids, cursor)[
                :max_multimodal_lora_requests
            ]
        )
        setattr(
            step_scheduler,
            cursor_attr,
            (getattr(step_scheduler, cursor_attr) + 1)
            % max(1, len(multimodal_lora_ids)),
        )
        shaped = [
            rid
            for rid in request_ids
            if (not is_multimodal_lora(scheduler.get_request(rid)))
            or rid in allowed_multimodal_lora
        ]
        return (
            shaped or request_ids[:1],
            max_multimodal_lora_requests,
            True,
            relaxed,
            tightened,
            relaxed_by_fairness,
            tightened_by_locality,
            max_fairness_gap,
        )

    def _effective_prefill_multimodal_lora_limit(
        self,
        *,
        scheduler: Any,
        step_scheduler: Any,
        candidate_request_ids: list[str],
        current_request_ids: list[str],
    ) -> tuple[int, bool, bool, float]:
        base_limit = step_scheduler.max_prefill_multimodal_lora_requests_per_batch
        if base_limit <= 0:
            return 0, False, False, 0.0
        hit_rate = step_scheduler._multimodal_prefix_cache_hit_rate_feedback
        candidate_count = self.count_multimodal_lora_requests(
            scheduler, candidate_request_ids
        )
        effective_limit = base_limit
        baseline_counts = self.count_multimodal_lora_adapters(
            scheduler, candidate_request_ids
        )
        current_counts = self.count_multimodal_lora_adapters(
            scheduler, current_request_ids
        )
        fairness_gap = share_gap_map(
            normalized_share_map(current_counts),
            normalized_share_map(baseline_counts),
        )
        max_gap = max_abs_share_gap(fairness_gap)
        relaxed_by_fairness = False
        tightened_by_locality = False
        if (
            step_scheduler.multimodal_lora_fairness_relax_threshold > 0
            and max_gap >= step_scheduler.multimodal_lora_fairness_relax_threshold
        ):
            effective_limit = min(
                candidate_count,
                effective_limit
                + step_scheduler.multimodal_lora_prefill_limit_relax_delta,
            )
            relaxed_by_fairness = effective_limit > base_limit
        elif (
            step_scheduler.multimodal_prefix_cache_relax_threshold > 0
            and hit_rate <= step_scheduler.multimodal_prefix_cache_relax_threshold
        ):
            effective_limit = min(
                candidate_count,
                effective_limit
                + step_scheduler.multimodal_lora_prefill_limit_relax_delta,
            )
        elif (
            step_scheduler.multimodal_prefix_cache_tighten_threshold > 0
            and hit_rate >= step_scheduler.multimodal_prefix_cache_tighten_threshold
            and (
                step_scheduler.multimodal_lora_locality_tighten_threshold <= 0
                or max_gap <= step_scheduler.multimodal_lora_locality_tighten_threshold
            )
            and effective_limit > 1
        ):
            effective_limit = max(
                1,
                effective_limit
                - step_scheduler.multimodal_lora_prefill_limit_tighten_delta,
            )
            tightened_by_locality = (
                step_scheduler.multimodal_lora_locality_tighten_threshold > 0
                and max_gap <= step_scheduler.multimodal_lora_locality_tighten_threshold
                and effective_limit < base_limit
            )
        return effective_limit, relaxed_by_fairness, tightened_by_locality, max_gap

    def _effective_decode_multimodal_lora_limit(
        self,
        *,
        scheduler: Any,
        step_scheduler: Any,
        candidate_request_ids: list[str],
        current_request_ids: list[str],
    ) -> tuple[int, bool, bool, float]:
        base_limit = step_scheduler.max_decode_multimodal_lora_requests_per_batch
        if base_limit <= 0:
            return 0, False, False, 0.0
        hit_rate = step_scheduler._multimodal_prefix_cache_hit_rate_feedback
        candidate_count = self.count_multimodal_lora_requests(
            scheduler, candidate_request_ids
        )
        effective_limit = base_limit
        baseline_counts = self.count_multimodal_lora_adapters(
            scheduler, candidate_request_ids
        )
        current_counts = self.count_multimodal_lora_adapters(
            scheduler, current_request_ids
        )
        fairness_gap = share_gap_map(
            normalized_share_map(current_counts),
            normalized_share_map(baseline_counts),
        )
        max_gap = max_abs_share_gap(fairness_gap)
        relaxed_by_fairness = False
        tightened_by_locality = False
        if (
            step_scheduler.multimodal_lora_fairness_relax_threshold > 0
            and max_gap >= step_scheduler.multimodal_lora_fairness_relax_threshold
        ):
            effective_limit = min(
                candidate_count,
                effective_limit
                + step_scheduler.multimodal_lora_prefill_limit_relax_delta,
            )
            relaxed_by_fairness = effective_limit > base_limit
        elif (
            step_scheduler.multimodal_prefix_cache_relax_threshold > 0
            and hit_rate <= step_scheduler.multimodal_prefix_cache_relax_threshold
        ):
            effective_limit = min(
                candidate_count,
                effective_limit
                + step_scheduler.multimodal_lora_prefill_limit_relax_delta,
            )
        elif (
            step_scheduler.multimodal_prefix_cache_tighten_threshold > 0
            and hit_rate >= step_scheduler.multimodal_prefix_cache_tighten_threshold
            and (
                step_scheduler.multimodal_lora_locality_tighten_threshold <= 0
                or max_gap <= step_scheduler.multimodal_lora_locality_tighten_threshold
            )
            and effective_limit > 1
        ):
            effective_limit = max(
                1,
                effective_limit
                - step_scheduler.multimodal_lora_prefill_limit_tighten_delta,
            )
            tightened_by_locality = (
                step_scheduler.multimodal_lora_locality_tighten_threshold > 0
                and max_gap <= step_scheduler.multimodal_lora_locality_tighten_threshold
                and effective_limit < base_limit
            )
        return effective_limit, relaxed_by_fairness, tightened_by_locality, max_gap

    def count_multimodal_lora_requests(
        self, scheduler: Any, request_ids: list[str]
    ) -> int:
        return sum(
            1 for rid in request_ids if is_multimodal_lora(scheduler.get_request(rid))
        )

    def count_multimodal_lora_adapters(
        self, scheduler: Any, request_ids: list[str]
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rid in request_ids:
            req = scheduler.get_request(rid)
            if is_multimodal_lora(req):
                key = lora_adapter_key(req)
                counts[key] = counts.get(key, 0) + 1
        return counts
