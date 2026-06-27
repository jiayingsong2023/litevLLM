# SPDX-License-Identifier: Apache-2.0
"""Stateless scheduling helpers extracted from StepScheduler.

These are pure functions with no dependency on StepScheduler instance state.
They are shared across the admission, prefill, and decode planners.
"""

BASE_LORA_ADAPTER = "__base__"


def rotate_candidates(request_ids: list[str], cursor: int) -> list[str]:
    """Rotate a list of request IDs by ``cursor`` positions."""
    if not request_ids:
        return []
    offset = cursor % len(request_ids)
    return request_ids[offset:] + request_ids[:offset]


def normalize_quotas(quotas: dict[str, int]) -> dict[str, int]:
    return {key: max(0, int(value)) for key, value in quotas.items()}


def percentile(values: list[float], q: float) -> float:
    """Compute the q-th percentile (0.0-1.0) of a list of values."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(v) for v in values)
    idx = int(round((len(sorted_values) - 1) * max(0.0, min(1.0, q))))
    return sorted_values[idx]


def is_multimodal(request) -> bool:
    return bool(request.is_multimodal or (request.multi_modal_data or {}).get("image"))


def is_multimodal_lora(request) -> bool:
    return is_multimodal(request) and bool(request.lora_id)


def lora_adapter_key(request, *, base_lora_adapter: str = BASE_LORA_ADAPTER) -> str:
    lora_id = request.lora_id
    if not lora_id:
        return base_lora_adapter
    return str(lora_id)


def service_class_priority(service_class: object) -> int:
    priorities = {
        "latency": 0,
        "interactive": 0,
        "balanced": 1,
        "throughput": 2,
        "background": 3,
    }
    key = str(service_class or "latency")
    return priorities.get(key, 4)


def normalized_share_map(counts: dict[str, int]) -> dict[str, float]:
    if not counts:
        return {}
    normalized = {key: float(value) for key, value in counts.items()}
    total = sum(normalized.values())
    if total <= 0:
        return {key: 0.0 for key in normalized}
    return {key: value / total for key, value in normalized.items()}


def share_gap_map(
    target_share: dict[str, float],
    baseline_share: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(target_share) | set(baseline_share))
    return {
        key: float(target_share.get(key, 0.0) or 0.0)
        - float(baseline_share.get(key, 0.0) or 0.0)
        for key in keys
    }


def max_abs_share_gap(gaps: dict[str, float]) -> float:
    return max((abs(float(value or 0.0)) for value in gaps.values()), default=0.0)
