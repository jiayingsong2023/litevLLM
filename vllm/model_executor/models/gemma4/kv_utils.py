# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .policy_utils import (
    _gemma4_model_policy_truthy,
    _meta_cpu_seq_lens,
    _meta_get,
    _meta_set,
)


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
    local_window: int | None = None,
    softcap: float | None = None,
    key_padding_mask: torch.Tensor | None = None,
    q_positions: torch.Tensor | None = None,
    k_positions: torch.Tensor | None = None,
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
    scale_cache: torch.Tensor | None,
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
    scales: torch.Tensor | None,
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
    local_window: int | None,
    kv_cache_dtype: str,
    kv_scale_cache: tuple[Any, Any] | None = None,
    seq_len_cpu: int | None = None,
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
    local_window: int | None,
    kv_cache_dtype: str,
    inf_config: Any = None,
    kv_scale_cache: tuple[Any, Any] | None = None,
    seq_lens_cpu: list[int] | None = None,
    max_seq_len_cpu: int | None = None,
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
    elif _gemma4_model_policy_truthy(inf_config, "legacy_item_path", default=False):
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
    seq_lens_cpu: list[int] | None = None,
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
    # When seq_lens_cpu is absent we fall back to ``.item()`` to preserve the
    # tight-shape contract of the legacy API. Callers on the Gemma4 decode
    # hot path always pass seq_lens_cpu via the engine-side builder, so this
    # sync never triggers in production; direct callers in tests and ad-hoc
    # scripts keep their previous shape guarantees.
    if seq_lens_cpu is not None and len(seq_lens_cpu) >= bsz:
        lw = int(local_window) if int(local_window) > 0 else 0
        max_blocks = 0
        for i in range(bsz):
            sl = int(seq_lens_cpu[i])
            st = max(0, sl - lw) if lw > 0 else 0
            sa = (st // block_size) * block_size
            nb = (sl - sa + block_size - 1) // block_size
            if nb > max_blocks:
                max_blocks = nb
    else:
        max_blocks = int(torch.max(num_blocks).item())

    if max_blocks <= 0:
        return seq_lens_aligned, block_tables.new_zeros((bsz, 0))

    # Vectorized gather replaces the per-sequence Python loop + ``.item()``.
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
    local_window: int | None,
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
    #   force_full_ref_attn:
    #     Emergency rollback: force ref-path for every dtype (numerical debugging).
    #   legacy_fp16_ref_attn:
    #     Preserve the pre-Step-3 behaviour where fp16/bf16 KV forces ref-path
    #     while int4/fp8 still takes the Triton kernel. Useful during rollout.
    if _gemma4_model_policy_truthy(inf_config, "force_full_ref_attn", default=False):
        return True
    if _gemma4_model_policy_truthy(inf_config, "legacy_fp16_ref_attn", default=False):
        kvt = str(kv_cache_dtype).lower()
        return ("int4" not in kvt) and ("fp8" not in kvt)
    return False


def _is_packed_or_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    kvt = str(kv_cache_dtype).lower()
    return ("int4" in kvt) or ("fp8" in kvt)


def _use_legacy_full_precision_kv_write(inf_config: Any) -> bool:
    return _gemma4_model_policy_truthy(
        inf_config, "legacy_fullprec_kv_write", default=False
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
