# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from typing import Any

import torch

from vllm.model_executor.models.lite_config import LiteConfig


def _get_eps(config: LiteConfig) -> float:
    return float(
        getattr(config, "rms_norm_eps", getattr(config, "layer_norm_epsilon", 1e-6))
    )


def _meta_get(meta: Any, key: str, default: Any = None) -> Any:
    if isinstance(meta, dict):
        return meta.get(key, default)
    return getattr(meta, key, default)


def _meta_cpu_seq_lens(meta: Any) -> list[int] | None:
    """
    Return the host-side per-sequence length list if the engine-side builder
    has already surfaced it; otherwise return None.

    Keeps the Gemma4 60-layer decode loop free of ``.item()`` D->H syncs which
    profiling showed dominate end-to-end latency (~107ms per sync).
    """
    raw = _meta_get(meta, "seq_lens_cpu", None)
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        return [int(v) for v in raw]
    return None


def _meta_cpu_max_seq_len(meta: Any) -> int | None:
    raw = _meta_get(meta, "max_seq_len_cpu", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _resolve_max_position_plus_one_cpu(
    attn_metadata: Any, positions: torch.Tensor
) -> int | None:
    """
    Best-effort CPU-side upper bound for RoPE cache extension.

    The engine-side builder populates ``max_seq_len_cpu`` per request on both
    prefill and decode paths. For any call site where this is present we can
    avoid ``positions.max().item()`` which forces a full device sync on the
    hot path (one per decoder layer x 60 layers).
    """
    cpu_max = _meta_cpu_max_seq_len(attn_metadata)
    if cpu_max is not None:
        return int(cpu_max)
    pos_cpu = _meta_get(attn_metadata, "positions_cpu", None)
    if isinstance(pos_cpu, (list, tuple)) and len(pos_cpu) > 0:
        return int(max(int(p) for p in pos_cpu)) + 1
    return None


def _meta_set(meta: Any, key: str, value: Any) -> bool:
    if isinstance(meta, dict):
        meta[key] = value
        return True
    try:
        setattr(meta, key, value)
        return True
    except Exception:
        return False


def _gemma4_policy_value(
    inf_config: Any,
    policy_name: str,
    default: object = None,
    *,
    policy_attr: str = "model_policy",
) -> object:
    policy = _meta_get(inf_config, policy_attr, None)
    if isinstance(policy, dict) and policy_name in policy:
        return policy[policy_name]
    return default


def _gemma4_model_policy_truthy(
    inf_config: Any,
    policy_name: str,
    *,
    default: bool = False,
) -> bool:
    raw = _gemma4_policy_value(inf_config, policy_name, default)
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().lower() in ("1", "true", "yes", "on")


def _gemma4_kernel_policy_truthy(
    inf_config: Any,
    policy_name: str,
    *,
    default: bool = False,
) -> bool:
    raw = _gemma4_policy_value(
        inf_config,
        policy_name,
        default,
        policy_attr="kernel_policy",
    )
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().lower() in ("1", "true", "yes", "on")


def _gemma4_fp32_residual_guard_policy(runtime_config: Any) -> tuple[bool, int, int]:
    model_policy = _meta_get(runtime_config, "model_policy", None)
    if isinstance(model_policy, dict):
        enabled = bool(model_policy.get("fp32_residual_guard_enabled", False))
        start = int(model_policy.get("fp32_residual_guard_start", 8))
        span = max(1, int(model_policy.get("fp32_residual_guard_span", 3)))
        return enabled, start, span
    enabled = bool(
        getattr(runtime_config, "gemma4_26b_fp32_residual_guard_enabled", False)
    )
    start = int(getattr(runtime_config, "gemma4_26b_fp32_residual_guard_start", 8))
    span = max(
        1, int(getattr(runtime_config, "gemma4_26b_fp32_residual_guard_span", 3))
    )
    return enabled, start, span


def _resolve_gemma4_rope_cache_max_pos(
    config: LiteConfig,
    runtime_config: Any = None,
) -> int:
    max_pos = int(getattr(config, "max_position_embeddings", 2048))
    kv_max_raw = getattr(runtime_config, "kv_max_model_len", None)
    if kv_max_raw is not None:
        with contextlib.suppress(ValueError):
            max_pos = min(max_pos, max(64, int(kv_max_raw)))
    rope_cap_raw = _gemma4_policy_value(runtime_config, "rope_cache_max_pos", None)
    if rope_cap_raw is None:
        rope_cap_raw = getattr(runtime_config, "gemma4_rope_cache_max_pos", None)
    if rope_cap_raw is not None:
        with contextlib.suppress(ValueError):
            max_pos = min(max_pos, max(64, int(rope_cap_raw)))
    return max(64, int(max_pos))


def _resolve_gemma4_rope_cache_pool_limit(runtime_config: Any = None) -> int:
    raw = _gemma4_policy_value(runtime_config, "rope_cache_pool_max", None)
    if raw is None:
        raw = getattr(runtime_config, "gemma4_rope_cache_pool_max", 8)
    try:
        return max(1, min(128, int(raw)))
    except ValueError:
        return 8


def _reshape_hidden_to_2d(
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    bsz, seqlen, hidden_dim = hidden_states.shape
    return hidden_states.reshape(bsz * seqlen, hidden_dim), (bsz, seqlen, hidden_dim)


def _restore_hidden_from_2d(
    hidden_states_2d: torch.Tensor,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    bsz, seqlen, hidden_dim = shape
    return hidden_states_2d.reshape(bsz, seqlen, hidden_dim)
