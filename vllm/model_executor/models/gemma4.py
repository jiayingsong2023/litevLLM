# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import atexit
import json
import os
import time
from collections import OrderedDict
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.lite_linear import LiteLinear
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from vllm.model_executor.layers.quantization.tensor import (
    dequantize_awq_pytorch,
    dequantize_symmetric_packed_int4_pytorch,
)
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from .lite_config import LiteConfig

_GEMMA4_TUNING: dict[str, str] = {
    key: value for key, value in os.environ.items() if key.startswith("FASTINFERENCE_GEMMA4_") or key.startswith("FASTINFERENCE_KV_MAX_")
}
_GEMMA4_TUNING_LOCKED = False


def set_gemma4_tuning_config(
    values: dict[str, object] | None, *, locked: bool = False
) -> None:
    """Install Gemma4 tuning flags before model construction."""
    global _GEMMA4_TUNING
    global _GEMMA4_TUNING_LOCKED
    global _GEMMA4_PROFILE_ENABLED
    global _GEMMA4_ROCTX_PROFILE_ENABLED
    _GEMMA4_TUNING = {
        str(key): str(value)
        for key, value in (values or {}).items()
        if (
            str(key).startswith("FASTINFERENCE_GEMMA4_")
            or str(key).startswith("FASTINFERENCE_KV_MAX_")
        )
        and value is not None
    }
    _GEMMA4_TUNING_LOCKED = bool(locked)
    _GEMMA4_PROFILE_ENABLED = _truthy_string(
        _GEMMA4_TUNING.get("FASTINFERENCE_GEMMA4_LAYER_PROFILE")
    )
    _GEMMA4_ROCTX_PROFILE_ENABLED = _truthy_string(
        _GEMMA4_TUNING.get("FASTINFERENCE_GEMMA4_ROCTX_PROFILE")
    )


def _env_get(name: str, default: str = "") -> str:
    if _GEMMA4_TUNING_LOCKED:
        return _GEMMA4_TUNING.get(name, default)
    return os.environ.get(name, _GEMMA4_TUNING.get(name, default))


def _truthy_string(raw: object) -> bool:
    return str(raw or "").strip().lower() in ("1", "true", "yes", "on")


_GEMMA4_PROFILE_ENABLED = False
_GEMMA4_ROCTX_PROFILE_ENABLED = False
_GEMMA4_PROFILE_STATS: dict[str, dict[str, float]] = {}
_GEMMA4_PROFILE_PRINTED = False
_GEMMA4_ROPE_CACHE_POOL: OrderedDict[
    tuple[int, int, float, str, float, str, int, str],
    tuple[torch.Tensor, torch.Tensor],
] = OrderedDict()

try:
    from torch.cuda import nvtx as _gemma4_roctx

    _gemma4_range_push = _gemma4_roctx.range_push
    _gemma4_range_pop = _gemma4_roctx.range_pop
except Exception:  # pragma: no cover - best-effort profiling hook

    def _gemma4_range_push(name: str) -> None:
        return None

    def _gemma4_range_pop() -> None:
        return None


def _resolve_gemma4_rope_cache_max_pos(
    config: LiteConfig,
    runtime_config: Any = None,
) -> int:
    max_pos = int(getattr(config, "max_position_embeddings", 2048))
    kv_max_raw = getattr(runtime_config, "kv_max_model_len", None)
    if kv_max_raw is not None:
        try:
            max_pos = min(max_pos, max(64, int(kv_max_raw)))
        except ValueError:
            pass
    rope_cap_raw = getattr(runtime_config, "gemma4_rope_cache_max_pos", None)
    if rope_cap_raw is not None:
        try:
            max_pos = min(max_pos, max(64, int(rope_cap_raw)))
        except ValueError:
            pass
    return max(64, int(max_pos))


def _resolve_gemma4_rope_cache_pool_limit(runtime_config: Any = None) -> int:
    raw = getattr(runtime_config, "gemma4_rope_cache_pool_max", 8)
    try:
        return max(1, min(128, int(raw)))
    except ValueError:
        return 8


def _gemma4_profile_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gemma4_profile_record(scope: str, elapsed_s: float) -> None:
    bucket = _GEMMA4_PROFILE_STATS.setdefault(scope, {"time_s": 0.0, "count": 0.0})
    bucket["time_s"] += float(elapsed_s)
    bucket["count"] += 1.0


class _Gemma4ProfileSpan:
    def __init__(self, scope: str):
        self.scope = scope
        self._start = 0.0

    def __enter__(self) -> "_Gemma4ProfileSpan":
        if _GEMMA4_ROCTX_PROFILE_ENABLED:
            _gemma4_range_push(self.scope)
        if _GEMMA4_PROFILE_ENABLED:
            _gemma4_profile_sync()
            self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if _GEMMA4_PROFILE_ENABLED:
            _gemma4_profile_sync()
            _gemma4_profile_record(self.scope, time.perf_counter() - self._start)
        if _GEMMA4_ROCTX_PROFILE_ENABLED:
            _gemma4_range_pop()


def _gemma4_profile_span(scope: str) -> _Gemma4ProfileSpan:
    return _Gemma4ProfileSpan(scope)


def _dump_gemma4_profile() -> None:
    global _GEMMA4_PROFILE_PRINTED
    if (
        (not _GEMMA4_PROFILE_ENABLED)
        or _GEMMA4_PROFILE_PRINTED
        or (not _GEMMA4_PROFILE_STATS)
    ):
        return
    _GEMMA4_PROFILE_PRINTED = True
    total_s = sum(v["time_s"] for v in _GEMMA4_PROFILE_STATS.values())
    rows = []
    for scope, data in sorted(
        _GEMMA4_PROFILE_STATS.items(),
        key=lambda item: item[1]["time_s"],
        reverse=True,
    ):
        time_s = float(data["time_s"])
        count = int(data["count"])
        rows.append(
            {
                "scope": scope,
                "time_s": time_s,
                "time_ms": time_s * 1000.0,
                "share_pct": (time_s / total_s * 100.0) if total_s > 0 else 0.0,
                "count": count,
                "avg_ms": (time_s * 1000.0 / count) if count > 0 else 0.0,
            }
        )
    print(
        "[Gemma4LayerProfile] "
        + json.dumps({"total_s": total_s, "rows": rows}, ensure_ascii=True)
    )


atexit.register(_dump_gemma4_profile)


def _get_eps(config: LiteConfig) -> float:
    return float(
        getattr(config, "rms_norm_eps", getattr(config, "layer_norm_epsilon", 1e-6))
    )


def _meta_get(meta: Any, key: str, default: Any = None) -> Any:
    if isinstance(meta, dict):
        return meta.get(key, default)
    return getattr(meta, key, default)


def _meta_cpu_seq_lens(meta: Any) -> Optional[list[int]]:
    """
    Return the host-side per-sequence length list if the engine-side builder
    has already surfaced it; otherwise return None.

    Keeps the Gemma4 60-layer decode loop free of `.item()` D->H syncs which
    profiling showed dominate end-to-end latency (~107ms per sync).
    """
    raw = _meta_get(meta, "seq_lens_cpu", None)
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        return [int(v) for v in raw]
    return None


def _meta_cpu_max_seq_len(meta: Any) -> Optional[int]:
    raw = _meta_get(meta, "max_seq_len_cpu", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _resolve_max_position_plus_one_cpu(
    attn_metadata: Any, positions: torch.Tensor
) -> Optional[int]:
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


def _get_sig_for_layer(
    attn_metadata: Any,
    layer_idx: int,
) -> "torch.Tensor | None":
    """Return the signature cache tensor for a given layer, or None."""
    _sig_list = _meta_get(attn_metadata, "sig_cache", None)
    if _sig_list is None:
        return None
    if not isinstance(_sig_list, list) or len(_sig_list) <= layer_idx:
        return None
    _sig = _sig_list[layer_idx]
    if _sig.numel() == 0:
        return None
    return _sig


def _env_truthy(name: str) -> bool:
    return _env_get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_truthy_default_on(name: str) -> bool:
    raw = _env_get(name)
    if raw is None:
        return True
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = _env_get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _gemma4_config_truthy_default_on(inf_config: Any, name: str) -> bool:
    tuning_env = _meta_get(inf_config, "tuning_env", None)
    if not isinstance(tuning_env, dict):
        return True
    raw = tuning_env.get(name)
    if raw is None:
        return True
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _gemma4_config_truthy_default_off(inf_config: Any, name: str) -> bool:
    tuning_env = _meta_get(inf_config, "tuning_env", None)
    if not isinstance(tuning_env, dict):
        return False
    raw = tuning_env.get(name)
    if raw is None:
        return False
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _gemma4_fp32_residual_guard_policy(runtime_config: Any) -> tuple[bool, int, int]:
    enabled = bool(
        getattr(runtime_config, "gemma4_26b_fp32_residual_guard_enabled", False)
    )
    start = int(getattr(runtime_config, "gemma4_26b_fp32_residual_guard_start", 8))
    span = max(
        1, int(getattr(runtime_config, "gemma4_26b_fp32_residual_guard_span", 3))
    )
    return enabled, start, span


def _env_int_alias(primary_name: str, legacy_name: str, default: int) -> int:
    raw = _env_get(primary_name, "").strip()
    if not raw:
        raw = _env_get(legacy_name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _repeat_kv_for_gqa(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, n_kv, seqlen, hd = x.shape
    x = x[:, :, None, :, :].expand(bsz, n_kv, n_rep, seqlen, hd)
    return x.reshape(bsz, n_kv * n_rep, seqlen, hd)


def _causal_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    local_window: Optional[int] = None,
    softcap: Optional[float] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    q_positions: Optional[torch.Tensor] = None,
    k_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    n_rep = q.shape[1] // k.shape[1]
    k_full = _repeat_kv_for_gqa(k, n_rep)
    v_full = _repeat_kv_for_gqa(v, n_rep)
    scores = torch.matmul(q, k_full.transpose(2, 3)) * scale
    bsz = int(q.shape[0])
    q_len = int(q.shape[2])
    kv_len = int(k_full.shape[2])
    # Support both square (q_len == kv_len) and chunked prefill (q_len < kv_len).
    # For chunked prefill, q is assumed to be the tail segment of the kv timeline.
    if q_positions is None:
        q_pos = (
            torch.arange(q_len, device=q.device, dtype=torch.long) + (kv_len - q_len)
        )[None, :].expand(bsz, q_len)
    else:
        q_pos = q_positions.to(device=q.device, dtype=torch.long)
    if k_positions is None:
        k_pos = torch.arange(kv_len, device=q.device, dtype=torch.long)[None, :].expand(
            bsz, kv_len
        )
    else:
        k_pos = k_positions.to(device=q.device, dtype=torch.long)
    causal = k_pos[:, None, :] > q_pos[:, :, None]
    scores = scores.masked_fill(causal[:, None, :, :], float("-inf"))
    if key_padding_mask is not None:
        pad_mask = ~key_padding_mask.to(device=q.device, dtype=torch.bool)
        scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
    if local_window is not None and local_window > 0:
        dist = q_pos[:, :, None] - k_pos[:, None, :]
        local_mask = dist >= int(local_window)
        scores = scores.masked_fill(local_mask[:, None, :, :], float("-inf"))
    if softcap is not None and softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(probs, v_full)
    return out.transpose(1, 2).contiguous()


def _decode_int4_row(
    cache: torch.Tensor,
    scale_cache: Optional[torch.Tensor],
    block_idx: int,
    block_offset: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    packed = cache[block_idx, block_offset, :num_kv_heads, : head_dim // 2].to(
        torch.int32
    )
    low = ((packed << 28) >> 28).to(torch.float32)
    high = ((packed << 24) >> 28).to(torch.float32)
    out = torch.empty(
        (num_kv_heads, head_dim), device=cache.device, dtype=torch.float32
    )
    out[:, : head_dim // 2] = low
    out[:, head_dim // 2 :] = high
    if scale_cache is not None:
        scale = scale_cache[block_idx, block_offset, :num_kv_heads, 0].to(torch.float32)
        out = out * scale[:, None]
    return out


def _decode_int4_rows(
    packed_rows: torch.Tensor,
    scales: Optional[torch.Tensor],
    head_dim: int,
) -> torch.Tensor:
    # packed_rows: [T, num_kv_heads, head_dim/2] uint8/int32
    packed_i32 = packed_rows.to(torch.int32)
    low = ((packed_i32 << 28) >> 28).to(torch.float32)
    high = ((packed_i32 << 24) >> 28).to(torch.float32)
    out = torch.empty(
        (packed_rows.shape[0], packed_rows.shape[1], head_dim),
        device=packed_rows.device,
        dtype=torch.float32,
    )
    half = head_dim // 2
    out[:, :, :half] = low
    out[:, :, half:] = high
    if scales is not None:
        out = out * scales.to(torch.float32).unsqueeze(-1)
    return out


def _gather_recent_kv(
    kv_cache: Any,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_idx: int,
    num_kv_heads: int,
    head_dim: int,
    local_window: Optional[int],
    kv_cache_dtype: str,
    kv_scale_cache: Optional[tuple[Any, Any]] = None,
    seq_len_cpu: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_cache, v_cache = kv_cache
    block_size = int(k_cache.shape[1])
    if seq_len_cpu is not None:
        seq_len = int(seq_len_cpu)
    else:
        seq_len = int(seq_lens[batch_idx].item())
    if local_window is None or int(local_window) <= 0:
        start = 0
    else:
        start = max(0, seq_len - int(local_window))
    k_scale_cache = kv_scale_cache[0] if kv_scale_cache is not None else None
    v_scale_cache = kv_scale_cache[1] if kv_scale_cache is not None else None
    is_int4 = "int4" in str(kv_cache_dtype).lower()
    token_positions = torch.arange(
        start, seq_len, device=block_tables.device, dtype=torch.long
    )
    bt_row = block_tables[batch_idx]
    block_indices = bt_row[token_positions // block_size]
    block_offsets = torch.remainder(token_positions, block_size)

    if is_int4:
        k_packed = k_cache[block_indices, block_offsets, :num_kv_heads, : head_dim // 2]
        v_packed = v_cache[block_indices, block_offsets, :num_kv_heads, : head_dim // 2]
        k_scales = (
            k_scale_cache[block_indices, block_offsets, :num_kv_heads, 0]
            if k_scale_cache is not None
            else None
        )
        v_scales = (
            v_scale_cache[block_indices, block_offsets, :num_kv_heads, 0]
            if v_scale_cache is not None
            else None
        )
        k = _decode_int4_rows(k_packed, k_scales, head_dim).unsqueeze(0)
        v = _decode_int4_rows(v_packed, v_scales, head_dim).unsqueeze(0)
    else:
        k = (
            k_cache[block_indices, block_offsets, :num_kv_heads, :head_dim]
            .to(torch.float32)
            .unsqueeze(0)
        )
        v = (
            v_cache[block_indices, block_offsets, :num_kv_heads, :head_dim]
            .to(torch.float32)
            .unsqueeze(0)
        )
    return k, v


def _gather_recent_kv_batched(
    kv_cache: Any,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    local_window: Optional[int],
    kv_cache_dtype: str,
    inf_config: Any = None,
    kv_scale_cache: Optional[tuple[Any, Any]] = None,
    seq_lens_cpu: Optional[list[int]] = None,
    max_seq_len_cpu: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k_cache, v_cache = kv_cache
    block_size = int(k_cache.shape[1])
    seq_lens_i64 = seq_lens.to(dtype=torch.long)
    bsz = int(seq_lens_i64.shape[0])
    if local_window is None or int(local_window) <= 0:
        starts = torch.zeros_like(seq_lens_i64)
    else:
        starts = torch.clamp(seq_lens_i64 - int(local_window), min=0)
    ctx_lens = seq_lens_i64 - starts
    # Prefer CPU-side scalar upper bound (injected by the engine-side builder)
    # to avoid a full device sync on the 60-layer decode loop.
    if seq_lens_cpu is not None and len(seq_lens_cpu) >= bsz and bsz > 0:
        if local_window is None or int(local_window) <= 0:
            max_ctx = int(max(int(v) for v in seq_lens_cpu[:bsz]))
        else:
            lw = int(local_window)
            max_ctx = int(max(min(int(v), lw) for v in seq_lens_cpu[:bsz]))
    elif max_seq_len_cpu is not None and bsz > 0:
        if local_window is None or int(local_window) <= 0:
            max_ctx = int(max_seq_len_cpu)
        else:
            max_ctx = min(int(max_seq_len_cpu), int(local_window))
    elif _gemma4_config_truthy_default_off(
        inf_config, "FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH"
    ):
        max_ctx = int(torch.max(ctx_lens).item()) if bsz > 0 else 0
    else:
        # Conservative upper bound: fall back to cache width instead of a
        # device sync. Trades a bit of extra gather work for zero sync cost.
        max_ctx = int(block_tables.shape[1]) * block_size if bsz > 0 else 0
        if local_window is not None and int(local_window) > 0:
            max_ctx = min(max_ctx, int(local_window))
    if max_ctx <= 0:
        empty = torch.empty(
            (bsz, 0, num_kv_heads, head_dim),
            device=block_tables.device,
            dtype=torch.float32,
        )
        return (
            empty,
            empty,
            torch.empty((bsz, 0), device=block_tables.device, dtype=torch.long),
            torch.empty((bsz, 0), device=block_tables.device, dtype=torch.bool),
        )

    rel = torch.arange(max_ctx, device=block_tables.device, dtype=torch.long)[None, :]
    valid = rel < ctx_lens[:, None]
    token_positions = starts[:, None] + rel
    safe_token_positions = torch.where(
        valid, token_positions, torch.zeros_like(token_positions)
    )
    bt_idx = torch.div(safe_token_positions, block_size, rounding_mode="floor")
    block_indices = block_tables.gather(1, bt_idx)
    block_offsets = torch.remainder(safe_token_positions, block_size)

    k_scale_cache = kv_scale_cache[0] if kv_scale_cache is not None else None
    v_scale_cache = kv_scale_cache[1] if kv_scale_cache is not None else None
    is_int4 = "int4" in str(kv_cache_dtype).lower()
    if is_int4:
        half = head_dim // 2
        k_packed = k_cache[block_indices, block_offsets, :num_kv_heads, :half]
        v_packed = v_cache[block_indices, block_offsets, :num_kv_heads, :half]
        k_scales = (
            k_scale_cache[block_indices, block_offsets, :num_kv_heads, 0]
            if k_scale_cache is not None
            else None
        )
        v_scales = (
            v_scale_cache[block_indices, block_offsets, :num_kv_heads, 0]
            if v_scale_cache is not None
            else None
        )
        k = _decode_int4_rows(
            k_packed.reshape(-1, num_kv_heads, half),
            None if k_scales is None else k_scales.reshape(-1, num_kv_heads),
            head_dim,
        ).reshape(bsz, max_ctx, num_kv_heads, head_dim)
        v = _decode_int4_rows(
            v_packed.reshape(-1, num_kv_heads, half),
            None if v_scales is None else v_scales.reshape(-1, num_kv_heads),
            head_dim,
        ).reshape(bsz, max_ctx, num_kv_heads, head_dim)
    else:
        k = k_cache[block_indices, block_offsets, :num_kv_heads, :head_dim].to(
            torch.float32
        )
        v = v_cache[block_indices, block_offsets, :num_kv_heads, :head_dim].to(
            torch.float32
        )

    valid4d = valid[:, :, None, None]
    k = torch.where(valid4d, k, torch.zeros_like(k))
    v = torch.where(valid4d, v, torch.zeros_like(v))
    k_positions = torch.where(
        valid, token_positions, torch.full_like(token_positions, -1)
    )
    return k, v, k_positions, valid


def _build_local_decode_aligned_metadata(
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    local_window: int,
    block_size: int,
    seq_lens_cpu: Optional[list[int]] = None,
    inf_config: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build local-window metadata for paged_attention_v1.

    The window start is aligned to block boundary so decode can run through
    paged-attention directly from KV cache. This may include up to
    (block_size - 1) extra left-context tokens.

    When ``seq_lens_cpu`` is provided the implementation avoids all
    ``.item()`` calls; the slicing is expressed via ``torch.gather`` so the
    whole batch of ``block_tables`` rows is assembled in a single kernel.
    """
    seq_lens_i64 = seq_lens.to(dtype=torch.long)
    bsz = int(seq_lens_i64.shape[0])
    if bsz <= 0:
        return seq_lens_i64, block_tables.new_zeros((0, 0))

    if int(local_window) > 0:
        starts = torch.clamp(seq_lens_i64 - int(local_window), min=0)
    else:
        starts = torch.zeros_like(seq_lens_i64)
    start_aligned = torch.div(starts, block_size, rounding_mode="floor") * block_size
    seq_lens_aligned = seq_lens_i64 - start_aligned
    num_blocks = torch.div(
        seq_lens_aligned + block_size - 1,
        block_size,
        rounding_mode="floor",
    )

    # Host-side scalar max_blocks: prefer CPU path (no D->H sync).
    # When seq_lens_cpu is absent we fall back to `.item()` to preserve the
    # tight-shape contract of the legacy API. Callers on the Gemma4 decode
    # hot path always pass seq_lens_cpu via the engine-side builder, so this
    # sync never triggers in production; direct callers in tests and ad-hoc
    # scripts keep their previous shape guarantees.
    if seq_lens_cpu is not None and len(seq_lens_cpu) >= bsz:
        lw = int(local_window) if int(local_window) > 0 else 0
        max_blocks = 0
        for i in range(bsz):
            sl = int(seq_lens_cpu[i])
            if lw > 0:
                st = max(0, sl - lw)
            else:
                st = 0
            sa = (st // block_size) * block_size
            nb = (sl - sa + block_size - 1) // block_size
            if nb > max_blocks:
                max_blocks = nb
    else:
        max_blocks = int(torch.max(num_blocks).item())

    if max_blocks <= 0:
        return seq_lens_aligned, block_tables.new_zeros((bsz, 0))

    # Vectorized gather replaces the per-sequence Python loop + `.item()`.
    # sblk[b] = start_aligned[b] // block_size
    sblk = torch.div(start_aligned, block_size, rounding_mode="floor")
    # rel[max_blocks] = [0, 1, ..., max_blocks - 1]
    rel = torch.arange(max_blocks, device=block_tables.device, dtype=sblk.dtype)
    # col[b, j] = sblk[b] + j, clamped into the source table's column range
    col = sblk[:, None] + rel[None, :]
    max_col = int(block_tables.shape[1]) - 1
    col_clamped = col.clamp(min=0, max=max(0, max_col))
    block_tables_aligned = torch.gather(
        block_tables.to(dtype=torch.long), 1, col_clamped
    ).to(dtype=block_tables.dtype)
    # Zero out columns that are past this sequence's real block count.
    valid = rel[None, :] < num_blocks[:, None]
    block_tables_aligned = torch.where(
        valid, block_tables_aligned, torch.zeros_like(block_tables_aligned)
    )
    return seq_lens_aligned, block_tables_aligned


def _get_or_build_local_decode_aligned_metadata(
    *,
    attn_metadata: Any,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    local_window: int,
    block_size: int,
    inf_config: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cache local decode aligned metadata in attn_metadata so all layers in the same
    decode step can reuse the same tensors.
    """
    cache_key = (
        int(local_window),
        int(block_size),
        int(block_tables.data_ptr()),
        int(seq_lens.data_ptr()),
        tuple(int(x) for x in block_tables.shape),
        tuple(int(x) for x in seq_lens.shape),
        str(block_tables.device),
        str(seq_lens.device),
    )
    cache = _meta_get(attn_metadata, "_gemma4_local_decode_aligned_cache", None)
    if isinstance(cache, dict) and cache.get("key") == cache_key:
        seq_cached = cache.get("seq_lens_local")
        bt_cached = cache.get("block_tables_local")
        if isinstance(seq_cached, torch.Tensor) and isinstance(bt_cached, torch.Tensor):
            return seq_cached, bt_cached

    seq_lens_cpu = _meta_cpu_seq_lens(attn_metadata)
    seq_lens_local, block_tables_local = _build_local_decode_aligned_metadata(
        block_tables=block_tables,
        seq_lens=seq_lens,
        local_window=local_window,
        block_size=block_size,
        seq_lens_cpu=seq_lens_cpu,
        inf_config=inf_config,
    )
    _meta_set(
        attn_metadata,
        "_gemma4_local_decode_aligned_cache",
        {
            "key": cache_key,
            "seq_lens_local": seq_lens_local,
            "block_tables_local": block_tables_local,
        },
    )
    return seq_lens_local, block_tables_local


def _local_prefill_attention_sdpa(
    q: torch.Tensor,
    k_ctx: torch.Tensor,
    v_ctx: torch.Tensor,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    k_valid: torch.Tensor,
    local_window: Optional[int],
    scale: float,
) -> torch.Tensor:
    qh = q.transpose(1, 2).contiguous()
    kh = k_ctx.transpose(1, 2).contiguous()
    vh = v_ctx.transpose(1, 2).contiguous()
    n_rep = qh.shape[1] // max(1, kh.shape[1])
    if n_rep > 1:
        kh = _repeat_kv_for_gqa(kh, n_rep)
        vh = _repeat_kv_for_gqa(vh, n_rep)
    if kh.dtype != qh.dtype:
        kh = kh.to(qh.dtype)
    if vh.dtype != qh.dtype:
        vh = vh.to(qh.dtype)

    causal = k_positions[:, None, :] > q_positions[:, :, None]
    invalid = causal | (~k_valid)[:, None, :]
    if local_window is not None and int(local_window) > 0:
        dist = q_positions[:, :, None] - k_positions[:, None, :]
        invalid = invalid | (dist >= int(local_window))
    attn_bias = torch.zeros(
        (qh.shape[0], 1, qh.shape[2], kh.shape[2]),
        device=qh.device,
        dtype=qh.dtype,
    )
    attn_bias = attn_bias.masked_fill(invalid[:, None, :, :], float("-inf"))
    out = F.scaled_dot_product_attention(
        qh,
        kh,
        vh,
        attn_mask=attn_bias,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return out.transpose(1, 2).contiguous()


def _should_use_full_decode_reference(inf_config: Any, kv_cache_dtype: str) -> bool:
    # Step 3 rewire: the Triton paged-attention kernel now supports Gemma-style
    # logit soft-capping natively, so full-precision (fp16/bf16) KV no longer
    # needs to fall back to the eager pytorch reference path on the decode hot
    # loop. We still honour two escape hatches:
    #   FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN=1
    #     Emergency rollback: force ref-path for every dtype (numerical debugging).
    #   FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN=1
    #     Preserve the pre-Step-3 behaviour where fp16/bf16 KV forces ref-path
    #     while int4/fp8 still takes the Triton kernel. Useful during rollout.
    if _gemma4_config_truthy_default_off(
        inf_config, "FASTINFERENCE_GEMMA4_FORCE_FULL_REF_ATTN"
    ):
        return True
    if _gemma4_config_truthy_default_off(
        inf_config, "FASTINFERENCE_GEMMA4_LEGACY_FP16_REF_ATTN"
    ):
        kvt = str(kv_cache_dtype).lower()
        return ("int4" not in kvt) and ("fp8" not in kvt)
    return False


def _is_packed_or_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    kvt = str(kv_cache_dtype).lower()
    return ("int4" in kvt) or ("fp8" in kvt)


def _use_legacy_full_precision_kv_write(inf_config: Any) -> bool:
    return _gemma4_config_truthy_default_off(
        inf_config, "FASTINFERENCE_GEMMA4_LEGACY_FULLPREC_KV_WRITE"
    )


def _write_full_precision_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    flat_k = k.reshape(-1, num_kv_heads, head_dim).contiguous()
    flat_v = v.reshape(-1, num_kv_heads, head_dim).contiguous()
    valid = slot_mapping >= 0
    if not bool(valid.any()):
        return
    slots = slot_mapping[valid].to(torch.long)
    block_size = int(k_cache.shape[1])
    block_idx = torch.div(slots, block_size, rounding_mode="floor")
    block_off = torch.remainder(slots, block_size)
    k_cache[block_idx, block_off, :num_kv_heads, :head_dim] = flat_k[valid].to(
        dtype=k_cache.dtype
    )
    v_cache[block_idx, block_off, :num_kv_heads, :head_dim] = flat_v[valid].to(
        dtype=v_cache.dtype
    )


def _layer_type_for_idx(config: LiteConfig, layer_idx: int) -> str:
    layer_types = getattr(config, "layer_types", None)
    if isinstance(layer_types, list) and layer_idx < len(layer_types):
        return str(layer_types[layer_idx]).lower()
    return "global"


def _is_local_layer(layer_type: str) -> bool:
    return any(x in layer_type for x in ("local", "sliding"))


class Gemma4LayerRotaryEmbedding(nn.Module):
    def __init__(
        self,
        config: LiteConfig,
        head_size: int,
        layer_type: str,
        runtime_config: Any = None,
    ):
        super().__init__()
        self.head_size = int(head_size)
        self.layer_type = layer_type
        self.max_position_embeddings_limit = int(config.max_position_embeddings)
        self.max_position_embeddings = _resolve_gemma4_rope_cache_max_pos(
            config, runtime_config
        )
        self._rope_cache_pool_limit = _resolve_gemma4_rope_cache_pool_limit(
            runtime_config
        )
        self.apply_rotary_emb = ApplyRotaryEmb(is_neox_style=True)

        rope_params = {}
        cfg_rope = getattr(config, "rope_parameters", None)
        if isinstance(cfg_rope, dict):
            layer_rope = cfg_rope.get(layer_type)
            if isinstance(layer_rope, dict):
                rope_params = layer_rope
        self.base = float(
            rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        )
        self.rope_type = str(rope_params.get("rope_type", "default"))
        self.partial_rotary_factor = float(
            rope_params.get("partial_rotary_factor", 1.0)
        )
        self._inv_freq_cpu = self._build_inv_freq().cpu()
        self._last_cache_key: Optional[
            tuple[int, int, float, str, float, str, int, str]
        ] = None
        self._last_cache_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def _build_inv_freq(self) -> torch.Tensor:
        if self.rope_type == "proportional":
            rope_angles = int(self.partial_rotary_factor * self.head_size // 2)
            rope_angles = max(0, min(self.head_size // 2, rope_angles))
            inv_rot = 1.0 / (
                self.base
                ** (
                    torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32)
                    / float(self.head_size)
                )
            )
            no_rot = self.head_size // 2 - rope_angles
            if no_rot > 0:
                inv_freq = torch.cat(
                    (inv_rot, torch.zeros(no_rot, dtype=torch.float32)), dim=0
                )
            else:
                inv_freq = inv_rot
            return inv_freq
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.head_size, 2, dtype=torch.float32)
                / float(self.head_size)
            )
        )

    def _ensure_cache_len(self, required_len: int) -> None:
        if required_len <= self.max_position_embeddings:
            return
        new_len = min(int(required_len), int(self.max_position_embeddings_limit))
        if new_len <= self.max_position_embeddings:
            return
        self.max_position_embeddings = int(new_len)
        self._last_cache_key = None
        self._last_cache_value = None

    def _cache_pool_key(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[int, int, float, str, float, str, int, str]:
        return (
            int(self.max_position_embeddings),
            int(self.head_size),
            float(self.base),
            str(self.rope_type),
            float(self.partial_rotary_factor),
            str(device.type),
            int(device.index) if device.index is not None else -1,
            str(dtype),
        )

    def _get_or_build_cache(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._cache_pool_key(device=device, dtype=dtype)
        if self._last_cache_key == key and self._last_cache_value is not None:
            return self._last_cache_value
        cached = _GEMMA4_ROPE_CACHE_POOL.get(key)
        if cached is not None:
            _GEMMA4_ROPE_CACHE_POOL.move_to_end(key)
            self._last_cache_key = key
            self._last_cache_value = cached
            return cached
        inv_freq = self._inv_freq_cpu.to(device=device, dtype=torch.float32)
        t = torch.arange(
            self.max_position_embeddings, device=device, dtype=torch.float32
        )
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        _GEMMA4_ROPE_CACHE_POOL[key] = (cos, sin)
        self._last_cache_key = key
        self._last_cache_value = (cos, sin)
        pool_limit = int(self._rope_cache_pool_limit)
        while len(_GEMMA4_ROPE_CACHE_POOL) > pool_limit:
            _GEMMA4_ROPE_CACHE_POOL.popitem(last=False)
        return cos, sin

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        max_position_plus_one_cpu: Optional[int] = None,
        inf_config: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Prefer a caller-provided CPU-side bound to avoid a GPU->CPU
        # sync (`positions.max().item()`) on every decoder layer.
        if max_position_plus_one_cpu is not None and int(max_position_plus_one_cpu) > 0:
            self._ensure_cache_len(int(max_position_plus_one_cpu))
        elif positions.numel() > 0:
            if _gemma4_config_truthy_default_off(
                inf_config, "FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH"
            ):
                required_len = int(positions.max().item()) + 1
                self._ensure_cache_len(required_len)
            else:
                # Ensure the cache covers the configured model limit once; then
                # growth happens lazily only when a hint explicitly exceeds it.
                self._ensure_cache_len(int(self.max_position_embeddings))
        if positions.device != query.device:
            positions = positions.to(query.device)
        cos_cached, sin_cached = self._get_or_build_cache(
            device=query.device,
            dtype=query.dtype,
        )
        cos = cos_cached[positions]
        sin = sin_cached[positions]
        return (
            self.apply_rotary_emb.forward_native(query, cos, sin),
            self.apply_rotary_emb.forward_native(key, cos, sin),
        )


def _get_rope(config: LiteConfig, head_size: int, layer_type: str):
    return _get_rope_with_runtime(config, head_size, layer_type, None)


def _get_rope_with_runtime(
    config: LiteConfig,
    head_size: int,
    layer_type: str,
    runtime_config: Any = None,
):
    rope_params = {}
    cfg_rope = getattr(config, "rope_parameters", None)
    if isinstance(cfg_rope, dict):
        layer_rope = cfg_rope.get(layer_type)
        if isinstance(layer_rope, dict):
            rope_params = layer_rope
    return Gemma4LayerRotaryEmbedding(
        config, head_size, layer_type, runtime_config=runtime_config
    )


class Gemma4Attention(nn.Module):
    def __init__(
        self,
        config: LiteConfig,
        quant_config: Any,
        prefix: str,
        layer_idx: int,
        runtime_config: Any = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = _layer_type_for_idx(config, layer_idx)
        self.is_sliding = _is_local_layer(self.layer_type)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(
            config.head_dim
            if self.is_sliding
            else getattr(config, "global_head_dim", config.head_dim)
        )
        self.use_alternative_attention = bool(
            getattr(config, "attention_k_eq_v", False) and not self.is_sliding
        )
        self.num_kv_heads = int(
            config.num_key_value_heads
            if self.is_sliding
            else getattr(
                config, "num_global_key_value_heads", config.num_key_value_heads
            )
        )
        # HF Gemma4TextAttention uses q/k RMSNorm and sets attention scaling to 1.0.
        # Keeping it aligned avoids over-damping logits (especially in MoE variants).
        self.scale = 1.0
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = LiteLinear(
            config.hidden_size,
            self.q_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.q_proj",
        )
        self.k_proj = LiteLinear(
            config.hidden_size,
            self.kv_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.k_proj",
        )
        self.v_proj = None
        if not self.use_alternative_attention:
            self.v_proj = LiteLinear(
                config.hidden_size,
                self.kv_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn.v_proj",
            )
        self.o_proj = LiteLinear(
            self.q_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=_get_eps(config))
        self.k_norm = RMSNorm(self.head_dim, eps=_get_eps(config))
        self.v_norm_eps = _get_eps(config)
        self.rotary_emb = _get_rope_with_runtime(
            config, self.head_dim, self.layer_type, runtime_config=runtime_config
        )

    @staticmethod
    def _apply_head_norm(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = norm(x.reshape(-1, shape[-1]))
        return y.view(*shape)

    @staticmethod
    def _apply_head_norm_noscale(x: torch.Tensor, eps: float) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        out = x_fp32 * torch.rsqrt(variance + eps)
        return out.to(input_dtype)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Any,
        attn_metadata: Any,
        lora_mapping: Any = None,
    ) -> torch.Tensor:
        with _gemma4_profile_span("attn_q_proj"):
            q = self.q_proj(x, lora_mapping)
        with _gemma4_profile_span("attn_k_proj"):
            k = self.k_proj(x, lora_mapping)
        if self.v_proj is not None:
            with _gemma4_profile_span("attn_v_proj"):
                v = self.v_proj(x, lora_mapping)
        else:
            v = k
        bsz, seqlen = x.shape[:2]
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = self._apply_head_norm(self.q_norm, q)
        k = self._apply_head_norm(self.k_norm, k)
        v = self._apply_head_norm_noscale(v, self.v_norm_eps)
        # Surface the max position upper bound from the engine-side builder so
        # rotary_emb can extend its cache without a per-layer D->H sync.
        max_pos_plus_one_cpu = _resolve_max_position_plus_one_cpu(
            attn_metadata, positions
        )
        inf_config = _meta_get(attn_metadata, "config", None)
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            max_position_plus_one_cpu=max_pos_plus_one_cpu,
            inf_config=inf_config,
        )
        is_local = self.is_sliding
        local_window = (
            int(getattr(self.config, "sliding_window", 0) or 0) if is_local else None
        )
        softcap = getattr(self.config, "attn_logit_softcapping", None)
        slot_mapping = _meta_get(attn_metadata, "slot_mapping", None)
        if slot_mapping is not None and kv_cache is not None:
            from vllm.kernels.triton.reshape_and_cache import reshape_and_cache

            kv_cache_dtype = (
                inf_config.kv_type
                if inf_config is not None
                else _meta_get(attn_metadata, "kv_cache_dtype", "auto")
            )
            k_scale = (
                inf_config.k_scale
                if inf_config is not None
                else _meta_get(attn_metadata, "k_scale", 1.0)
            )
            v_scale = (
                inf_config.v_scale
                if inf_config is not None
                else _meta_get(attn_metadata, "v_scale", 1.0)
            )
            k_cache, v_cache = kv_cache
            kv_scale_cache = _meta_get(attn_metadata, "kv_scale_cache", None)
            if kv_scale_cache is not None:
                k_scale_cache, v_scale_cache = kv_scale_cache[self.layer_idx]
            else:
                k_scale_cache, v_scale_cache = (None, None)

            if (
                _use_legacy_full_precision_kv_write(inf_config)
                and _should_use_full_decode_reference(inf_config, str(kv_cache_dtype))
                and (not _is_packed_or_quantized_kv_cache(str(kv_cache_dtype)))
            ):
                with _gemma4_profile_span("kv_write_full_precision"):
                    _write_full_precision_kv_cache(
                        k,
                        v,
                        k_cache,
                        v_cache,
                        slot_mapping,
                        self.num_kv_heads,
                        self.head_dim,
                    )
            else:
                with _gemma4_profile_span("kv_write_reshape_and_cache"):
                    reshape_and_cache(
                        k.reshape(-1, self.num_kv_heads, self.head_dim).contiguous(),
                        v.reshape(-1, self.num_kv_heads, self.head_dim).contiguous(),
                        k_cache,
                        v_cache,
                        slot_mapping,
                        kv_cache_dtype,
                        k_scale,
                        v_scale,
                        k_scale_cache=k_scale_cache,
                        v_scale_cache=v_scale_cache,
                    )

            # Finalize block signatures for any blocks that just filled.
            _sig_cache_list = _meta_get(attn_metadata, "sig_cache", None)
            _sig_temp_list = _meta_get(attn_metadata, "sig_temp", None)
            if (
                _sig_cache_list is not None
                and _sig_temp_list is not None
                and len(_sig_cache_list) > self.layer_idx
                and _sig_cache_list[self.layer_idx].numel() > 0
            ):
                _block_sz = int(k_cache.shape[1])
                _slot = slot_mapping.reshape(-1)
                _block_ids = _slot // _block_sz
                _offsets = _slot % _block_sz
                _filled_mask = _offsets == (_block_sz - 1)
                if _filled_mask.any():
                    _filled_blocks = torch.unique(_block_ids[_filled_mask])
                    from vllm.kernels.triton.kv_sig import kv_sig_finalize
                    kv_sig_finalize(
                        _sig_temp_list[self.layer_idx],
                        _sig_cache_list[self.layer_idx],
                        _filled_blocks,
                    )

            is_prefill = bool(_meta_get(attn_metadata, "is_prefill", False))
            if is_local and is_prefill and seqlen > 1:
                with _gemma4_profile_span("attn_local_prefill"):
                    block_tables = _meta_get(attn_metadata, "block_tables", None)
                    seq_lens = _meta_get(attn_metadata, "seq_lens", None)
                    kv_start_t = _meta_get(attn_metadata, "kv_start_indices", None)
                    if kv_start_t is None:
                        q_starts = (
                            seq_lens.to(device=q.device, dtype=torch.long) - seqlen
                        )
                    else:
                        q_starts = kv_start_t.to(
                            device=q.device, dtype=torch.long
                        ).reshape(-1)
                    q_positions = (
                        q_starts[:, None]
                        + torch.arange(seqlen, device=q.device, dtype=torch.long)[
                            None, :
                        ]
                    )
                    with _gemma4_profile_span("kv_read_local_prefill"):
                        k_ctx, v_ctx, k_positions, k_valid = _gather_recent_kv_batched(
                            kv_cache=kv_cache,
                            block_tables=block_tables,
                            seq_lens=seq_lens,
                            num_kv_heads=self.num_kv_heads,
                            head_dim=self.head_dim,
                            local_window=int(local_window or 0),
                            kv_cache_dtype=str(kv_cache_dtype),
                            inf_config=inf_config,
                            kv_scale_cache=(k_scale_cache, v_scale_cache),
                            seq_lens_cpu=_meta_cpu_seq_lens(attn_metadata),
                            max_seq_len_cpu=_meta_cpu_max_seq_len(attn_metadata),
                        )
                    if softcap is not None and float(softcap) > 0:
                        out = (
                            _causal_attention_ref(
                                q.transpose(1, 2).float(),
                                k_ctx.transpose(1, 2).float(),
                                v_ctx.transpose(1, 2).float(),
                                self.scale,
                                local_window=None,
                                softcap=softcap,
                                key_padding_mask=k_valid,
                                q_positions=q_positions,
                                k_positions=k_positions,
                            )
                            .to(q.dtype)
                            .view(bsz, seqlen, -1)
                        )
                    else:
                        out = (
                            _local_prefill_attention_sdpa(
                                q,
                                k_ctx,
                                v_ctx,
                                q_positions,
                                k_positions,
                                k_valid,
                                local_window=None,
                                scale=self.scale,
                            )
                            .to(q.dtype)
                            .view(bsz, seqlen, -1)
                        )
            elif is_local and not is_prefill:
                with _gemma4_profile_span("attn_local_decode"):
                    block_tables = _meta_get(attn_metadata, "block_tables", None)
                    seq_lens = _meta_get(attn_metadata, "seq_lens", None)
                    kv_dtype_name = (
                        inf_config.kv_type
                        if inf_config is not None
                        else _meta_get(attn_metadata, "kv_cache_dtype", "auto")
                    )
                    ctx_window = int(local_window or 0)
                    # Step 3 rewire: the Triton paged-attention kernel now
                    # applies Gemma-style softcap internally (scale -> softcap
                    # -> -inf mask -> online softmax), so we no longer need to
                    # route softcap>0 through the eager pytorch ref path here.
                    use_triton_local_decode = (
                        _gemma4_config_truthy_default_on(
                            inf_config,
                            "FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON"
                        )
                        and block_tables is not None
                        and seq_lens is not None
                        and seqlen == 1
                    )
                    if use_triton_local_decode:
                        from vllm.kernels.triton.paged_attention import (
                            paged_attention_v1,
                        )

                        attn_out = torch.empty(
                            (bsz * seqlen, self.num_heads, self.head_dim),
                            device=q.device,
                            dtype=q.dtype,
                        )
                        with _gemma4_profile_span("kv_read_local_decode"):
                            seq_lens_local, block_tables_local = (
                                _get_or_build_local_decode_aligned_metadata(
                                    attn_metadata=attn_metadata,
                                    block_tables=block_tables,
                                    seq_lens=seq_lens,
                                    local_window=ctx_window,
                                    block_size=int(k_cache.shape[1]),
                                    inf_config=inf_config,
                                )
                            )
                        # Upper bound for paged_attention_v1: prefer a cheap
                        # CPU-side scalar so the 60-layer decode loop never
                        # triggers a D->H sync here. The bound only needs to
                        # upper-cover seq_lens_local per batch row.
                        _slc_cpu = _meta_cpu_seq_lens(attn_metadata)
                        _max_cpu = _meta_cpu_max_seq_len(attn_metadata)
                        _block_sz = int(k_cache.shape[1])
                        if _slc_cpu is not None and len(_slc_cpu) > 0:
                            if ctx_window > 0:
                                max_ctx_local = max(
                                    (min(int(s), ctx_window) + _block_sz - 1)
                                    for s in _slc_cpu
                                )
                            else:
                                max_ctx_local = max(int(s) for s in _slc_cpu)
                        elif _max_cpu is not None:
                            if ctx_window > 0:
                                max_ctx_local = (
                                    min(int(_max_cpu), ctx_window) + _block_sz - 1
                                )
                            else:
                                max_ctx_local = int(_max_cpu)
                        elif _gemma4_config_truthy_default_off(
                            inf_config,
                            "FASTINFERENCE_GEMMA4_LEGACY_ITEM_PATH",
                        ):
                            max_ctx_local = (
                                int(torch.max(seq_lens_local).item())
                                if int(seq_lens_local.numel()) > 0
                                else 0
                            )
                        else:
                            # Conservative upper bound derived from tensor
                            # shapes (free) rather than a device-side reduce.
                            max_ctx_local = (
                                int(block_tables_local.shape[1]) * _block_sz
                                if int(seq_lens_local.numel()) > 0
                                else 0
                            )
                        with _gemma4_profile_span("attn_local_decode_kernel"):
                            _sig_local = _get_sig_for_layer(
                                attn_metadata, self.layer_idx
                            )
                            paged_attention_v1(
                                attn_out,
                                q.reshape(
                                    bsz * seqlen, self.num_heads, self.head_dim
                                ).contiguous(),
                                k_cache,
                                v_cache,
                                self.num_heads,
                                self.scale,
                                block_tables_local,
                                seq_lens_local.to(
                                    device=q.device, dtype=seq_lens.dtype
                                ),
                                k_cache.shape[1],
                                max_ctx_local,
                                None,
                                kv_dtype_name,
                                k_scale,
                                v_scale,
                                k_scale_ptrs=k_scale_cache,
                                v_scale_ptrs=v_scale_cache,
                                num_kv_heads=self.num_kv_heads,
                                attn_scope="local",
                                layer_type=self.layer_type,
                                config=inf_config,
                                softcap=(
                                    float(softcap)
                                    if softcap is not None and float(softcap) > 0.0
                                    else None
                                ),
                                sig_cache=_sig_local,
                                kv_select_ratio=0.0,
                                kv_select_min_blocks=0,
                            )
                        out = attn_out.view(bsz, seqlen, -1)
                    else:
                        with _gemma4_profile_span("kv_read_local_decode"):
                            k_ctx, v_ctx, k_positions, k_valid = (
                                _gather_recent_kv_batched(
                                    kv_cache=kv_cache,
                                    block_tables=block_tables,
                                    seq_lens=seq_lens,
                                num_kv_heads=self.num_kv_heads,
                                head_dim=self.head_dim,
                                local_window=ctx_window,
                                kv_cache_dtype=str(kv_dtype_name),
                                inf_config=inf_config,
                                kv_scale_cache=(k_scale_cache, v_scale_cache),
                                seq_lens_cpu=_meta_cpu_seq_lens(attn_metadata),
                                max_seq_len_cpu=_meta_cpu_max_seq_len(
                                    attn_metadata
                                ),
                                )
                            )
                        q_positions = (
                            seq_lens.to(device=q.device, dtype=torch.long)[:, None]
                            - seqlen
                        ) + torch.arange(seqlen, device=q.device, dtype=torch.long)[
                            None, :
                        ]
                        out = (
                            _causal_attention_ref(
                                q.transpose(1, 2).float(),
                                k_ctx.transpose(1, 2).float(),
                                v_ctx.transpose(1, 2).float(),
                                self.scale,
                                local_window=None,
                                softcap=softcap,
                                key_padding_mask=k_valid,
                                q_positions=q_positions,
                                k_positions=k_positions,
                            )
                            .to(q.dtype)
                            .view(bsz, seqlen, -1)
                        )
            else:
                with _gemma4_profile_span("attn_global"):
                    from vllm.engine.lite_engine import (
                        expand_metadata_for_paged_attention,
                    )
                    from vllm.kernels.triton.paged_attention import paged_attention_v1

                    attn_out = torch.empty(
                        (bsz * seqlen, self.num_heads, self.head_dim),
                        device=q.device,
                        dtype=q.dtype,
                    )
                    block_tables = _meta_get(attn_metadata, "block_tables", None)
                    seq_lens = _meta_get(attn_metadata, "seq_lens", None)
                    use_full_ref = _should_use_full_decode_reference(
                        inf_config, str(kv_cache_dtype)
                    )
                    if (
                        use_full_ref
                        and block_tables is not None
                        and seq_lens is not None
                    ):
                        outs = []
                        _global_seq_lens_cpu = _meta_cpu_seq_lens(attn_metadata)
                        for bi in range(bsz):
                            with _gemma4_profile_span("kv_read_global_ref"):
                                _slc_hint = None
                                if _global_seq_lens_cpu is not None and bi < len(
                                    _global_seq_lens_cpu
                                ):
                                    _slc_hint = int(_global_seq_lens_cpu[bi])
                                k_ctx, v_ctx = _gather_recent_kv(
                                    kv_cache=kv_cache,
                                    block_tables=block_tables,
                                    seq_lens=seq_lens,
                                    batch_idx=bi,
                                    num_kv_heads=self.num_kv_heads,
                                    head_dim=self.head_dim,
                                    local_window=None,
                                    kv_cache_dtype=str(kv_cache_dtype),
                                    kv_scale_cache=(k_scale_cache, v_scale_cache),
                                    seq_len_cpu=_slc_hint,
                                )
                            q_i = q[bi : bi + 1].transpose(1, 2).float()
                            out_i = (
                                _causal_attention_ref(
                                    q_i,
                                    k_ctx.transpose(1, 2).float(),
                                    v_ctx.transpose(1, 2).float(),
                                    self.scale,
                                    local_window=None,
                                    softcap=softcap,
                                )
                                .to(q.dtype)
                                .view(1, seqlen, -1)
                            )
                            outs.append(out_i)
                        out = torch.cat(outs, dim=0)
                        return self.o_proj(out, lora_mapping)
                    seq_lens_ext, block_tables_ext = (
                        expand_metadata_for_paged_attention(
                            bsz,
                            seqlen,
                            is_prefill,
                            seq_lens,
                            block_tables,
                            q.device,
                            seq_lens_cpu=_meta_cpu_seq_lens(attn_metadata),
                        )
                    )
                    max_ctx = int(
                        max(
                            self.num_heads * self.head_dim,
                            getattr(self.config, "max_position_embeddings", 4096),
                        )
                    )
                    with _gemma4_profile_span("attn_global_kernel"):
                        _sig_global = _get_sig_for_layer(
                            attn_metadata, self.layer_idx
                        )
                        _kv_sel_ratio = float(
                            getattr(inf_config, "kv_select_ratio", 0.0)
                            if inf_config is not None
                            else 0.0
                        )
                        _kv_sel_min = int(
                            getattr(inf_config, "kv_select_min_blocks", 4)
                            if inf_config is not None
                            else 4
                        )
                        paged_attention_v1(
                            attn_out,
                            q.reshape(
                                bsz * seqlen, self.num_heads, self.head_dim
                            ).contiguous(),
                            k_cache,
                            v_cache,
                            self.num_heads,
                            self.scale,
                            block_tables_ext,
                            seq_lens_ext,
                            k_cache.shape[1],
                            max_ctx,
                            None,
                            kv_cache_dtype,
                            k_scale,
                            v_scale,
                            k_scale_ptrs=k_scale_cache,
                            v_scale_ptrs=v_scale_cache,
                            num_kv_heads=self.num_kv_heads,
                            attn_scope="global",
                            layer_type=self.layer_type,
                            config=inf_config,
                            softcap=(
                                float(softcap)
                                if softcap is not None and float(softcap) > 0.0
                                else None
                            ),
                            sig_cache=_sig_global,
                            kv_select_ratio=_kv_sel_ratio,
                            kv_select_min_blocks=_kv_sel_min,
                        )
                    out = attn_out.view(bsz, seqlen, -1)
        else:
            with _gemma4_profile_span("attn_nocache"):
                out = _causal_attention_ref(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    self.scale,
                    local_window=local_window,
                    softcap=softcap,
                ).view(bsz, seqlen, -1)
        with _gemma4_profile_span("attn_o_proj"):
            return self.o_proj(out, lora_mapping)


class Gemma4MLP(nn.Module):
    def __init__(self, config: LiteConfig, quant_config: Any, prefix: str):
        super().__init__()
        self.hidden_act = str(
            getattr(config, "hidden_activation", getattr(config, "hidden_act", "silu"))
        ).lower()
        self.intermediate_size = int(config.intermediate_size)
        self.gate_proj = LiteLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.gate_proj",
        )
        self.up_proj = LiteLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.up_proj",
        )
        self.down_proj = LiteLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.down_proj",
        )

    def _apply_activation(self, gate: torch.Tensor) -> torch.Tensor:
        # Keep activation dispatch isolated so the fused and unfused branches
        # share one implementation.
        if self.hidden_act in ("gelu", "gelu_pytorch_tanh"):
            return F.gelu(gate, approximate="tanh")
        return F.silu(gate)

    def forward(
        self,
        x: torch.Tensor,
        lora_mapping: Any = None,
        inf_config: Any = None,
    ) -> torch.Tensor:
        # Step 4 pair fusion: concat gate_proj and up_proj into a single
        # quantized GEMM, halving the per-layer kernel launches for AWQ/int4
        # weights.
        #
        # LoRA-activity detection is delegated to the helper because the
        # input_batch_builder always hands us a list like ``[None, None]``
        # for non-LoRA requests (see vllm/engine/input_batch_builder.py).
        # The outer env gate only toggles the optimization wholesale.
        #
        # The helper returns None when structural guards trip (mismatched
        # shapes, high-fidelity flag, active LoRA, etc.) and the caller
        # falls back to the two-matmul path below.
        if _gemma4_config_truthy_default_on(
            inf_config, "FASTINFERENCE_AWQ_FUSED_GATE_UP"
        ):
            from vllm.model_executor.models._fused_awq_pair import (
                try_fused_awq_gate_up_activation,
            )

            h = try_fused_awq_gate_up_activation(
                x,
                self.gate_proj,
                self.up_proj,
                activation=self.hidden_act,
                lora_mapping=lora_mapping,
            )
            if h is not None:
                return self.down_proj(h, lora_mapping)

        if _gemma4_config_truthy_default_on(
            inf_config, "FASTINFERENCE_GEMMA4_MLP_PAIR_FUSION"
        ):
            from vllm.model_executor.models._fused_awq_pair import (
                try_fused_awq_pair_matmul,
            )

            gu = try_fused_awq_pair_matmul(
                x,
                self.gate_proj,
                self.up_proj,
                self,
                "mlp_gate_up",
                lora_mapping=lora_mapping,
            )
            if gu is not None:
                gate, up = torch.split(gu, self.intermediate_size, dim=-1)
                act = self._apply_activation(gate)
                return self.down_proj(act * up, lora_mapping)

        gate = self.gate_proj(x, lora_mapping)
        up = self.up_proj(x, lora_mapping)
        act = self._apply_activation(gate)
        return self.down_proj(act * up, lora_mapping)


def _is_gemma4_moe_enabled(config: LiteConfig) -> bool:
    return bool(
        int(getattr(config, "num_experts", 0) or 0) > 0
        and int(getattr(config, "num_experts_per_tok", 0) or 0) > 0
        and int(getattr(config, "moe_intermediate_size", 0) or 0) > 0
    )


def _is_gemma4_moe_layer(config: LiteConfig, layer_idx: int) -> bool:
    if not _is_gemma4_moe_enabled(config):
        return False
    if hasattr(config, "is_moe_layer"):
        try:
            return bool(config.is_moe_layer(layer_idx))
        except Exception:
            pass
    return True


def _is_gemma4_26b_a4b_like(config: LiteConfig) -> bool:
    return bool(
        int(getattr(config, "hidden_size", 0) or 0) == 2816
        and int(getattr(config, "num_hidden_layers", 0) or 0) == 30
        and int(getattr(config, "num_experts", 0) or 0) >= 64
        and int(getattr(config, "moe_intermediate_size", 0) or 0) > 0
    )


def _residual_add_fp32(residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    return (residual.float() + update.float()).to(residual.dtype)


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


class Gemma4TopKRouterLite(nn.Module):
    def __init__(self, config: LiteConfig, quant_config: Any, prefix: str):
        super().__init__()
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.hidden_size = int(config.hidden_size)
        self.eps = float(_get_eps(config))
        self.scalar_root_size = float(max(1, self.hidden_size)) ** -0.5
        self.proj = LiteLinear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.router.proj",
        )
        # Optional router scaling tensors used by Gemma4-26B A4B checkpoints.
        self.scale = nn.Parameter(torch.empty(0), requires_grad=False)
        self.per_expert_scale = nn.Parameter(torch.empty(0), requires_grad=False)

    def forward(
        self,
        hidden_states_2d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Match HF Gemma4 router math:
        # x = RMSNorm(with_scale=False)(x) * scale * (hidden_size ** -0.5)
        x_fp32 = hidden_states_2d.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x = (x_fp32 * torch.rsqrt(variance + self.eps)).to(hidden_states_2d.dtype)
        if self.scale.numel() > 1:
            x = x * self.scale.to(device=x.device, dtype=x.dtype)
        x = x * self.scalar_root_size
        router_logits = self.proj(x)
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        routing_weights, selected_experts = torch.topk(
            routing_weights,
            k=self.top_k,
            dim=-1,
        )
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
        if self.per_expert_scale.numel() > 1:
            per_exp = self.per_expert_scale.to(
                device=routing_weights.device,
                dtype=routing_weights.dtype,
            )
            routing_weights = routing_weights * per_exp[selected_experts]
        return (
            router_logits,
            routing_weights.to(hidden_states_2d.dtype),
            selected_experts,
        )


def _materialize_litelinear_dense_weight_awqaware(
    layer: LiteLinear,
    *,
    out_features: int,
    in_features: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    dense_weight = getattr(layer, "weight", None)
    if isinstance(dense_weight, torch.Tensor) and dense_weight.numel() > 1:
        dense_weight = dense_weight[:out_features, :in_features].contiguous()
        return dense_weight.to(device=device, dtype=dtype)

    qweight = getattr(layer, "qweight", None)
    scales = getattr(layer, "scales", None)
    qzeros = getattr(layer, "qzeros", None)
    group_size = int(getattr(layer, "group_size", 128))
    if qweight is None or not isinstance(qweight, torch.Tensor) or qweight.numel() <= 1:
        raise RuntimeError(
            f"Layer '{getattr(layer, 'prefix', '<unknown>')}' has neither dense nor packed weights."
        )
    if scales is None or not isinstance(scales, torch.Tensor) or scales.numel() <= 1:
        raise RuntimeError(
            f"Layer '{getattr(layer, 'prefix', '<unknown>')}' is missing AWQ scales."
        )

    if isinstance(qzeros, torch.Tensor) and qzeros.numel() > 1:
        dense_weight = dequantize_awq_pytorch(
            qweight.to(device=device, dtype=torch.int32),
            scales.to(device=device),
            qzeros.to(device=device, dtype=torch.int32),
            group_size=group_size,
        )
    else:
        dense_weight = dequantize_symmetric_packed_int4_pytorch(
            qweight.to(device=device, dtype=torch.int32),
            scales.to(device=device),
            group_size=group_size,
        )

    if dense_weight.shape[0] < out_features or dense_weight.shape[1] < in_features:
        raise RuntimeError(
            f"Layer '{getattr(layer, 'prefix', '<unknown>')}' dequantized weight too small: "
            f"got {tuple(dense_weight.shape)}, need ({out_features}, {in_features})"
        )
    return (
        dense_weight[:out_features, :in_features]
        .contiguous()
        .to(device=device, dtype=dtype)
    )


class Gemma4MoeExpertsLite(nn.Module):
    def __init__(
        self,
        config: LiteConfig,
        quant_config: Any,
        prefix: str,
        runtime_config: Any = None,
    ):
        super().__init__()
        self.hidden_dim = int(config.hidden_size)
        self.intermediate_dim = int(config.moe_intermediate_size)
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.hidden_act = str(
            getattr(config, "hidden_activation", getattr(config, "hidden_act", "silu"))
        ).lower()
        self.gate_up_proj = LiteLinear(
            self.hidden_dim,
            self.num_experts * (2 * self.intermediate_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts.gate_up_proj",
        )
        self.down_proj = LiteLinear(
            self.intermediate_dim,
            self.num_experts * self.hidden_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts.down_proj",
        )
        self._cached_device: Optional[torch.device] = None
        self._cached_dtype: Optional[torch.dtype] = None
        self._cached_w1: Optional[torch.Tensor] = None
        self._cached_w2: Optional[torch.Tensor] = None
        self._expert_weight_cache: "OrderedDict[int, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._expert_cache_device: Optional[torch.device] = None
        self._expert_cache_dtype: Optional[torch.dtype] = None
        self._max_expert_cache = max(
            0,
            int(getattr(runtime_config, "gemma4_moe_expert_cache_size", 8)),
        )

    def _apply_gate_activation(self, gate: torch.Tensor) -> torch.Tensor:
        if self.hidden_act in ("gelu", "gelu_pytorch_tanh"):
            return F.gelu(gate, approximate="tanh")
        return F.silu(gate)

    def _has_awq_packed_expert_major(self) -> bool:
        qweight_gu = getattr(self.gate_up_proj, "qweight", None)
        scales_gu = getattr(self.gate_up_proj, "scales", None)
        qweight_d = getattr(self.down_proj, "qweight", None)
        scales_d = getattr(self.down_proj, "scales", None)
        return (
            isinstance(qweight_gu, torch.Tensor)
            and isinstance(scales_gu, torch.Tensor)
            and qweight_gu.ndim == 3
            and scales_gu.ndim == 3
            and isinstance(qweight_d, torch.Tensor)
            and isinstance(scales_d, torch.Tensor)
            and qweight_d.ndim == 3
            and scales_d.ndim == 3
        )

    def _materialize_one_expert_awq(
        self,
        expert_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _gemma4_profile_span("moe_materialize_one_expert_awq"):
            return self._materialize_one_expert_awq_impl(expert_id, device, dtype)

    def _materialize_one_expert_awq_impl(
        self,
        expert_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._max_expert_cache > 0
            and self._expert_cache_device == device
            and self._expert_cache_dtype == dtype
        ):
            cached = self._expert_weight_cache.get(expert_id)
            if cached is not None:
                self._expert_weight_cache.move_to_end(expert_id)
                return cached
        if self._expert_cache_device != device or self._expert_cache_dtype != dtype:
            self._expert_weight_cache.clear()
            self._expert_cache_device = device
            self._expert_cache_dtype = dtype

        qweight_gu = self.gate_up_proj.qweight
        scales_gu = self.gate_up_proj.scales
        qweight_d = self.down_proj.qweight
        scales_d = self.down_proj.scales

        gsz_gu = max(1, int((qweight_gu.shape[2] * 8) // max(1, scales_gu.shape[2])))
        gsz_d = max(1, int((qweight_d.shape[2] * 8) // max(1, scales_d.shape[2])))
        w1e = dequantize_symmetric_packed_int4_pytorch(
            qweight_gu[expert_id].to(device=device, dtype=torch.int32),
            scales_gu[expert_id].to(device=device),
            group_size=gsz_gu,
        )
        w2e = dequantize_symmetric_packed_int4_pytorch(
            qweight_d[expert_id].to(device=device, dtype=torch.int32),
            scales_d[expert_id].to(device=device),
            group_size=gsz_d,
        )
        w1 = (
            w1e[: 2 * self.intermediate_dim, : self.hidden_dim]
            .contiguous()
            .to(device=device, dtype=dtype)
        )
        w2 = (
            w2e[: self.hidden_dim, : self.intermediate_dim]
            .contiguous()
            .to(device=device, dtype=dtype)
        )
        if self._max_expert_cache > 0:
            self._expert_weight_cache[expert_id] = (w1, w2)
            self._expert_weight_cache.move_to_end(expert_id)
            while len(self._expert_weight_cache) > self._max_expert_cache:
                self._expert_weight_cache.popitem(last=False)
        return w1, w2

    def _forward_awq_streaming(
        self,
        hidden_states_2d: torch.Tensor,
        router_logits: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if topk_weights is None or topk_ids is None:
            if router_logits is None:
                raise RuntimeError(
                    "router_logits or top-k routing inputs are required."
                )
            topk_weights, topk_ids = torch.topk(
                router_logits,
                k=self.top_k,
                dim=-1,
            )
            topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).to(
                hidden_states_2d.dtype
            )
        compute_dtype = torch.float32
        out = torch.zeros_like(hidden_states_2d, dtype=compute_dtype)
        n_tokens = int(hidden_states_2d.shape[0])
        flat_topk_ids = topk_ids.reshape(-1).to(torch.long)
        flat_topk_weights = topk_weights.reshape(-1)
        flat_token_idx = torch.arange(
            n_tokens, device=hidden_states_2d.device, dtype=torch.long
        ).repeat_interleave(self.top_k)

        sorted_expert_ids, sort_idx = torch.sort(flat_topk_ids)
        sorted_token_idx = flat_token_idx.index_select(0, sort_idx)
        sorted_weights = flat_topk_weights.index_select(0, sort_idx)
        unique_experts, counts = torch.unique_consecutive(
            sorted_expert_ids, return_counts=True
        )

        start = 0
        for expert_id_t, count_t in zip(unique_experts, counts):
            expert_id = int(expert_id_t.item())
            count = int(count_t.item())
            if count <= 0:
                continue
            end = start + count
            token_idx = sorted_token_idx[start:end]
            coeff = sorted_weights[start:end].unsqueeze(-1).to(compute_dtype)
            start = end
            x_sel = hidden_states_2d.index_select(0, token_idx).to(compute_dtype)
            w1e, w2e = self._materialize_one_expert_awq(
                expert_id,
                hidden_states_2d.device,
                compute_dtype,
            )
            with _gemma4_profile_span("moe_sparse_expert_linear"):
                gu = F.linear(x_sel, w1e)
                g, u = torch.chunk(gu, 2, dim=-1)
                h = self._apply_gate_activation(g) * u
                y = F.linear(h, w2e) * coeff
            out.index_add_(0, token_idx, y)
        return out.to(hidden_states_2d.dtype)

    def _materialize_expert_weights(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cached_w1 is not None
            and self._cached_w2 is not None
            and self._cached_device == device
            and self._cached_dtype == dtype
        ):
            return self._cached_w1, self._cached_w2

        qweight_gu = getattr(self.gate_up_proj, "qweight", None)
        scales_gu = getattr(self.gate_up_proj, "scales", None)
        qweight_d = getattr(self.down_proj, "qweight", None)
        scales_d = getattr(self.down_proj, "scales", None)

        # Gemma4-26B-A4B checkpoint path: expert-major packed tensors
        # gate_up_proj_packed: [E, 2I, H/8], gate_up_proj_scale: [E, 2I, H/group]
        # down_proj_packed: [E, H, I/8], down_proj_scale: [E, H, I/group]
        if (
            isinstance(qweight_gu, torch.Tensor)
            and isinstance(scales_gu, torch.Tensor)
            and qweight_gu.ndim == 3
            and scales_gu.ndim == 3
            and isinstance(qweight_d, torch.Tensor)
            and isinstance(scales_d, torch.Tensor)
            and qweight_d.ndim == 3
            and scales_d.ndim == 3
        ):
            w1_parts = []
            w2_parts = []
            gsz_gu = max(
                1, int((qweight_gu.shape[2] * 8) // max(1, scales_gu.shape[2]))
            )
            gsz_d = max(1, int((qweight_d.shape[2] * 8) // max(1, scales_d.shape[2])))
            for e in range(self.num_experts):
                w1e = dequantize_symmetric_packed_int4_pytorch(
                    qweight_gu[e].to(device=device, dtype=torch.int32),
                    scales_gu[e].to(device=device),
                    group_size=gsz_gu,
                )
                w2e = dequantize_symmetric_packed_int4_pytorch(
                    qweight_d[e].to(device=device, dtype=torch.int32),
                    scales_d[e].to(device=device),
                    group_size=gsz_d,
                )
                w1_parts.append(
                    w1e[: 2 * self.intermediate_dim, : self.hidden_dim].to(
                        device=device, dtype=dtype
                    )
                )
                w2_parts.append(
                    w2e[: self.hidden_dim, : self.intermediate_dim].to(
                        device=device, dtype=dtype
                    )
                )
            w1 = torch.stack(w1_parts, dim=0).contiguous()
            w2 = torch.stack(w2_parts, dim=0).contiguous()
        else:
            gate_up_dense = _materialize_litelinear_dense_weight_awqaware(
                self.gate_up_proj,
                out_features=self.num_experts * (2 * self.intermediate_dim),
                in_features=self.hidden_dim,
                device=device,
                dtype=dtype,
            )
            down_dense = _materialize_litelinear_dense_weight_awqaware(
                self.down_proj,
                out_features=self.num_experts * self.hidden_dim,
                in_features=self.intermediate_dim,
                device=device,
                dtype=dtype,
            )
            w1 = gate_up_dense.view(
                self.num_experts,
                2 * self.intermediate_dim,
                self.hidden_dim,
            ).contiguous()
            w2 = down_dense.view(
                self.num_experts,
                self.hidden_dim,
                self.intermediate_dim,
            ).contiguous()

        self._cached_device = device
        self._cached_dtype = dtype
        self._cached_w1 = w1
        self._cached_w2 = w2
        return w1, w2

    def forward(
        self,
        hidden_states_2d: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._has_awq_packed_expert_major():
            return self._forward_awq_streaming(
                hidden_states_2d,
                router_logits,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )
        if topk_weights is not None and topk_ids is not None:
            router_logits = None
            # Build sparse logits proxy for fused_moe path: fill selected experts
            # with log-weights and keep others very negative.
            n_tok = int(hidden_states_2d.shape[0])
            if router_logits is None:
                router_logits = torch.full(
                    (n_tok, self.num_experts),
                    -1e9,
                    device=hidden_states_2d.device,
                    dtype=hidden_states_2d.dtype,
                )
            router_logits.scatter_(1, topk_ids, topk_weights.clamp_min(1e-20).log())
        if router_logits is None:
            raise RuntimeError(
                "router_logits is required when top-k routing is not provided."
            )
        w1, w2 = self._materialize_expert_weights(
            hidden_states_2d.device,
            hidden_states_2d.dtype,
        )
        return fused_moe(
            hidden_states_2d,
            w1,
            w2,
            router_logits,
            topk=self.top_k,
            renormalize=True,
        )


class Gemma4SparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: LiteConfig,
        quant_config: Any,
        prefix: str,
        runtime_config: Any = None,
    ):
        super().__init__()
        self.router = Gemma4TopKRouterLite(config, quant_config, prefix)
        self.experts = Gemma4MoeExpertsLite(
            config, quant_config, prefix, runtime_config=runtime_config
        )
        self.shared_mlp = Gemma4MLP(config, quant_config, prefix)

    def forward_branches(
        self,
        hidden_states_dense: torch.Tensor,
        hidden_states_sparse: torch.Tensor,
        hidden_states_router: Optional[torch.Tensor] = None,
        lora_mapping: Any = None,
        inf_config: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_2d, shape = _reshape_hidden_to_2d(hidden_states_sparse)
        router_src = (
            hidden_states_router
            if hidden_states_router is not None
            else hidden_states_sparse
        )
        router_2d, _ = _reshape_hidden_to_2d(router_src)
        router_logits, routing_weights, selected_experts = self.router(router_2d)
        sparse_out_2d = self.experts(
            hidden_states_2d,
            router_logits,
            topk_weights=routing_weights,
            topk_ids=selected_experts,
        )
        sparse_out = _restore_hidden_from_2d(sparse_out_2d, shape)
        dense_out = self.shared_mlp(
            hidden_states_dense, lora_mapping, inf_config=inf_config
        )
        return dense_out, sparse_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        lora_mapping: Any = None,
        inf_config: Any = None,
    ) -> torch.Tensor:
        dense_out, sparse_out = self.forward_branches(
            hidden_states,
            hidden_states,
            hidden_states_router=hidden_states,
            lora_mapping=lora_mapping,
            inf_config=inf_config,
        )
        return dense_out + sparse_out


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: LiteConfig,
        quant_config: Any,
        prefix: str,
        layer_idx: int,
        fp32_residual_guard_enabled: bool = False,
        fp32_residual_guard_start: int = 8,
        fp32_residual_guard_span: int = 3,
        runtime_config: Any = None,
    ):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=_get_eps(config))
        self.self_attn = Gemma4Attention(
            config,
            quant_config,
            prefix,
            layer_idx,
            runtime_config=runtime_config,
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=_get_eps(config)
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=_get_eps(config)
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=_get_eps(config)
        )
        self.pre_feedforward_layernorm_2: Optional[RMSNorm] = None
        self.post_feedforward_layernorm_1: Optional[RMSNorm] = None
        self.post_feedforward_layernorm_2: Optional[RMSNorm] = None
        self.layer_scalar = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self._fp32_residual_guard_enabled = bool(fp32_residual_guard_enabled)
        self._fp32_residual_guard_start = int(fp32_residual_guard_start)
        self._fp32_residual_guard_span = max(1, int(fp32_residual_guard_span))
        self.use_moe = _is_gemma4_moe_layer(config, layer_idx)
        if self.use_moe:
            # Gemma4-26B-A4B checkpoints expose dual pre-FFN norms and dual
            # branch post-FFN norms at layer root.
            self.pre_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=_get_eps(config)
            )
            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, eps=_get_eps(config)
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=_get_eps(config)
            )
            self.mlp = Gemma4SparseMoeBlock(
                config,
                quant_config,
                prefix,
                runtime_config=runtime_config,
            )
        else:
            self.mlp = Gemma4MLP(config, quant_config, prefix)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Any,
        attn_metadata: Any,
        lora_mapping: Any = None,
    ) -> torch.Tensor:
        inf_config = _meta_get(attn_metadata, "config", None)
        residual = x
        h = self.input_layernorm(x)
        with _gemma4_profile_span("layer_self_attn"):
            h = self.self_attn(h, positions, kv_cache, attn_metadata, lora_mapping)
        h = self.post_attention_layernorm(h)
        guard_hit = (
            self._fp32_residual_guard_enabled
            and self._fp32_residual_guard_start
            <= self.layer_idx
            < (self._fp32_residual_guard_start + self._fp32_residual_guard_span)
        )
        if guard_hit:
            x = _residual_add_fp32(residual, h)
        else:
            x = residual + h

        residual = x
        h_dense = self.pre_feedforward_layernorm(x)
        if self.use_moe and isinstance(self.mlp, Gemma4SparseMoeBlock):
            # Match HF Gemma4 MoE flow:
            # - dense MLP branch consumes pre_feedforward_layernorm(residual)
            # - router consumes raw residual (before pre-FF norms)
            # - sparse experts consume pre_feedforward_layernorm_2(residual)
            with _gemma4_profile_span("layer_dense_mlp"):
                dense_out = self.mlp.shared_mlp(
                    h_dense, lora_mapping=lora_mapping, inf_config=inf_config
                )
            if self.post_feedforward_layernorm_1 is not None:
                dense_out = self.post_feedforward_layernorm_1(dense_out)

            router_in_2d, router_shape = _reshape_hidden_to_2d(residual)
            with _gemma4_profile_span("layer_moe_router"):
                router_logits, routing_weights, selected_experts = self.mlp.router(
                    router_in_2d
                )
            if self.pre_feedforward_layernorm_2 is not None:
                sparse_in = self.pre_feedforward_layernorm_2(residual)
            else:
                sparse_in = residual
            sparse_in_2d, _ = _reshape_hidden_to_2d(sparse_in)
            with _gemma4_profile_span("layer_moe_sparse_experts"):
                sparse_out_2d = self.mlp.experts(
                    sparse_in_2d,
                    router_logits,
                    topk_weights=routing_weights,
                    topk_ids=selected_experts,
                )
            sparse_out = _restore_hidden_from_2d(sparse_out_2d, router_shape)
            if self.post_feedforward_layernorm_2 is not None:
                sparse_out = self.post_feedforward_layernorm_2(sparse_out)
            h = dense_out + sparse_out
        else:
            with _gemma4_profile_span("layer_dense_mlp"):
                h = self.mlp(h_dense, lora_mapping, inf_config=inf_config)
        h = self.post_feedforward_layernorm(h)
        if guard_hit:
            x = _residual_add_fp32(residual, h)
        else:
            x = residual + h
        return x * self.layer_scalar


class Gemma4TextModel(nn.Module):
    def __init__(
        self,
        hf_config: Any,
        quant_config: Any,
        prefix: str = "model",
        runtime_config: Any = None,
    ):
        super().__init__()
        self.config = LiteConfig(hf_config)
        fp32_residual_guard_enabled, fp32_residual_guard_start, fp32_residual_guard_span = (
            _gemma4_fp32_residual_guard_policy(runtime_config)
        )
        padding_idx = int(getattr(hf_config, "pad_token_id", 0) or 0)
        self.embed_scale = float(self.config.hidden_size) ** 0.5
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            padding_idx=padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    self.config,
                    quant_config,
                    prefix=f"layers.{i}",
                    layer_idx=i,
                    fp32_residual_guard_enabled=fp32_residual_guard_enabled,
                    fp32_residual_guard_start=fp32_residual_guard_start,
                    fp32_residual_guard_span=fp32_residual_guard_span,
                    runtime_config=runtime_config,
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=_get_eps(self.config))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Any,
        attn_metadata: Any,
        lora_mapping: Any = None,
    ) -> torch.Tensor:
        if input_ids.dtype == torch.long:
            x = self.embed_tokens(input_ids) * self.embed_scale
        else:
            x = input_ids
        for i, layer in enumerate(self.layers):
            x = layer(x, positions, kv_caches[i], attn_metadata, lora_mapping)
        return self.norm(x)


def _assert_text_only_kwargs(kwargs: dict[str, Any]) -> None:
    banned = (
        "pixel_values",
        "image_embeds",
        "audio_values",
        "input_features",
        "multimodal_embeddings",
    )
    for k in banned:
        if k in kwargs and kwargs[k] is not None:
            raise NotImplementedError(
                f"Gemma4 text-only path does not support multimodal input: {k}"
            )


class Gemma4ForConditionalGeneration(nn.Module):
    def __init__(self, vllm_config: Any, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        self.model = Gemma4TextModel(
            hf_config,
            vllm_config.quant_config,
            prefix="model",
            runtime_config=getattr(vllm_config, "runtime_config", None),
        )
        self.lm_head = LiteLinear(
            self.model.config.hidden_size,
            self.model.config.vocab_size,
            bias=False,
            prefix="lm_head",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Any,
        attn_metadata: Any,
        lora_mapping: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        _assert_text_only_kwargs(kwargs)
        hidden = self.model(
            input_ids, positions, kv_caches, attn_metadata, lora_mapping
        )
        if getattr(self.model.config, "tie_word_embeddings", False):
            logits = torch.nn.functional.linear(
                hidden[:, -1:, :], self.model.embed_tokens.weight
            )
        else:
            logits = self.lm_head(hidden[:, -1:, :], lora_mapping)
        final_softcap = getattr(self.model.config, "final_logit_softcapping", None)
        if final_softcap is not None and float(final_softcap) > 0:
            logits = torch.tanh(logits / float(final_softcap)) * float(final_softcap)
        return logits


class Gemma4ForCausalLM(Gemma4ForConditionalGeneration):
    pass
