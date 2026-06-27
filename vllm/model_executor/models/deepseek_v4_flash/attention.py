# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import torch

from .config import DEEPSEEK_V4_FLASH_SHAPE, layer_compress_ratio
from .ops import factorized_linear_reference, rms_norm_reference


def factorized_attention_projection_reference(
    hidden: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply GGUF-oriented factor weights to a hidden vector.

    DeepSeek V4 Flash GGUF attention factors are stored as (input, output).
    The shared linear reference consumes row-major (output, input), so transpose
    both factors before delegating.
    """
    return factorized_linear_reference(
        hidden,
        a_weight.transpose(0, 1),
        b_weight.transpose(0, 1),
    )


def q_lora_attention_projection_reference(
    hidden: torch.Tensor,
    q_a_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_b_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute the DeepSeek V4 Flash Q-LoRA query projection.

    Official reference order: ``q = q_norm(wq_a(x)); q = wq_b(q)``.
    GGUF factor tensors are stored as ``(input, output)``.
    """
    if q_a_weight.ndim != 2 or q_b_weight.ndim != 2:
        raise ValueError("Q-LoRA weights must be 2-D")
    q_rank = q_a_weight.shape[1]
    if q_norm_weight.shape != (q_rank,):
        raise ValueError(
            f"q_norm_weight shape must be ({q_rank},); "
            f"got {tuple(q_norm_weight.shape)}"
        )
    q_latent = q_a_weight.transpose(0, 1).to(torch.float32).matmul(
        hidden.to(torch.float32)
    )
    q_latent = rms_norm_reference(q_latent, q_norm_weight)
    return q_b_weight.transpose(0, 1).to(torch.float32).matmul(q_latent)


def latent_kv_projection_reference(
    hidden: torch.Tensor,
    kv_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute the DeepSeek V4 Flash single-latent K/V projection.

    The 512-wide ``attn_kv.weight`` is not split into separate key and value
    vectors. It projects hidden state into one head-dim latent that serves as
    both K and V in the reference attention path.
    """
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D; got {hidden.ndim}-D")
    if kv_weight.ndim != 2:
        raise ValueError(f"kv_weight must be 2-D; got {kv_weight.ndim}-D")
    if kv_weight.shape[0] != hidden.numel():
        raise ValueError(
            "kv_weight input dimension must match hidden size; "
            f"got {kv_weight.shape[0]} and {hidden.numel()}"
        )
    kv_width = kv_weight.shape[1]
    if kv_norm_weight.shape != (kv_width,):
        raise ValueError(
            f"kv_norm_weight shape must be ({kv_width},); "
            f"got {tuple(kv_norm_weight.shape)}"
        )
    kv = kv_weight.transpose(0, 1).to(torch.float32).matmul(hidden.to(torch.float32))
    return rms_norm_reference(kv, kv_norm_weight)


def split_combined_kv_reference(
    kv: torch.Tensor,
    *,
    key_width: int,
    value_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an already-projected synthetic K/V vector for legacy tests.

    Do not use this for DeepSeek V4 Flash real attention. The target GGUF uses
    a 512-wide shared K=V latent row from ``attn_kv.weight``.
    """
    if kv.ndim != 1:
        raise ValueError(f"kv must be 1-D; got {kv.ndim}-D")
    if key_width <= 0 or value_width <= 0:
        raise ValueError("widths must be positive")
    expected_width = key_width + value_width
    if kv.numel() != expected_width:
        raise ValueError(
            "kv length must equal key_width + value_width; "
            f"got {kv.numel()} and {expected_width}"
        )
    return kv[:key_width], kv[key_width:]


def raw_swa_attention_reference(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    if query.ndim != 1:
        raise ValueError(f"query must be 1-D; got {query.ndim}-D")
    if keys.ndim != 2:
        raise ValueError(f"keys must be 2-D; got {keys.ndim}-D")
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D; got {values.ndim}-D")
    if query.numel() == 0:
        raise ValueError("query must contain at least one element")
    if keys.shape[0] != values.shape[0]:
        raise ValueError(
            "keys and values must have the same row count; "
            f"got {keys.shape[0]} and {values.shape[0]}"
        )
    if keys.shape[0] == 0:
        raise ValueError("attention reference requires at least one key/value row")
    if keys.shape[1] != query.numel():
        raise ValueError(
            "key columns must match query size; "
            f"got {keys.shape[1]} and {query.numel()}"
        )

    scores = keys.to(torch.float32).matmul(query.to(torch.float32))
    scores = scores / math.sqrt(float(query.numel()))
    probs = torch.softmax(scores, dim=0)
    return probs.matmul(values.to(torch.float32))


def per_head_rms_norm_reference(
    query: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if query.ndim != 2:
        raise ValueError(f"query must be 2-D; got {query.ndim}-D")
    variance = query.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return query.to(torch.float32) * torch.rsqrt(variance + eps)


def apply_rope_to_tail_reference(
    vectors: torch.Tensor,
    *,
    token_idx: int,
    rotary_dim: int = 64,
    theta: float = 10000.0,
    inverse: bool = False,
) -> torch.Tensor:
    if vectors.ndim < 1:
        raise ValueError("vectors must have at least one dimension")
    if token_idx < 0:
        raise ValueError(f"token_idx must be non-negative; got {token_idx}")
    if rotary_dim <= 0:
        raise ValueError(f"rotary_dim must be positive; got {rotary_dim}")
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even; got {rotary_dim}")
    if vectors.shape[-1] < rotary_dim:
        raise ValueError(
            "vector width must be >= rotary_dim; "
            f"got {vectors.shape[-1]} and {rotary_dim}"
        )

    out = vectors.to(torch.float32).clone()
    tail = out[..., -rotary_dim:]
    pairs = tail.reshape(*tail.shape[:-1], rotary_dim // 2, 2)
    positions = torch.arange(
        0,
        rotary_dim,
        2,
        dtype=torch.float32,
        device=out.device,
    )
    inv_freq = torch.pow(
        torch.tensor(theta, dtype=torch.float32, device=out.device),
        -positions / float(rotary_dim),
    )
    angles = float(token_idx) * inv_freq
    if inverse:
        angles = -angles
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x0 = pairs[..., 0]
    x1 = pairs[..., 1]
    rotated = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    out[..., -rotary_dim:] = rotated.reshape_as(tail)
    return out


def _rope_yarn_ramp(low: float, high: float, i0: torch.Tensor) -> torch.Tensor:
    y = ((i0 // 2).to(torch.float32) - low) / max(0.001, high - low)
    return 1.0 - torch.minimum(
        torch.tensor(1.0, dtype=torch.float32, device=i0.device),
        torch.maximum(torch.tensor(0.0, dtype=torch.float32, device=i0.device), y),
    )


def _rope_yarn_corr_dim(
    *,
    n_dims: int,
    n_ctx_orig: int,
    n_rot: float,
    base: float,
) -> float:
    return float(n_dims) * math.log(
        float(n_ctx_orig) / (n_rot * 2.0 * math.pi)
    ) / (2.0 * math.log(base))


def _rope_yarn_corr_dims(
    *,
    n_dims: int,
    n_ctx_orig: int,
    freq_base: float,
    beta_fast: float,
    beta_slow: float,
) -> tuple[float, float]:
    start = math.floor(
        _rope_yarn_corr_dim(
            n_dims=n_dims,
            n_ctx_orig=n_ctx_orig,
            n_rot=beta_fast,
            base=freq_base,
        )
    )
    end = math.ceil(
        _rope_yarn_corr_dim(
            n_dims=n_dims,
            n_ctx_orig=n_ctx_orig,
            n_rot=beta_slow,
            base=freq_base,
        )
    )
    return max(0.0, float(start)), min(float(n_dims - 1), float(end))


def apply_deepseek_layer_rope_to_tail_reference(
    vectors: torch.Tensor,
    *,
    token_idx: int,
    layer_idx: int,
    rotary_dim: int = DEEPSEEK_V4_FLASH_SHAPE.rotary_dim,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply DS4 layer-aware RoPE to the tail of each attention vector.

    Dense layers use the default 10k base. Compressed layers use the DS4
    long-context base and interpolation schedule from the reference runtime.
    """
    ratio = layer_compress_ratio(layer_idx)
    if ratio == 0:
        return apply_rope_to_tail_reference(
            vectors,
            token_idx=token_idx,
            rotary_dim=rotary_dim,
            theta=DEEPSEEK_V4_FLASH_SHAPE.rope_freq_base,
            inverse=inverse,
        )
    if vectors.ndim < 1:
        raise ValueError("vectors must have at least one dimension")
    if token_idx < 0:
        raise ValueError(f"token_idx must be non-negative; got {token_idx}")
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError(
            "rotary_dim must be a positive even integer; "
            f"got {rotary_dim}"
        )
    if vectors.shape[-1] < rotary_dim:
        raise ValueError(
            "vector width must be >= rotary_dim; "
            f"got {vectors.shape[-1]} and {rotary_dim}"
        )

    shape = DEEPSEEK_V4_FLASH_SHAPE
    out = vectors.to(torch.float32).clone()
    tail = out[..., -rotary_dim:]
    pairs = tail.reshape(*tail.shape[:-1], rotary_dim // 2, 2)
    i0 = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=out.device)
    theta_scale = shape.compressed_rope_freq_base ** (-2.0 / float(rotary_dim))
    theta_extrap = float(token_idx) * torch.pow(theta_scale, i0 / 2.0)
    theta_interp = (1.0 / shape.rope_scale_factor) * theta_extrap
    corr_low, corr_high = _rope_yarn_corr_dims(
        n_dims=rotary_dim,
        n_ctx_orig=shape.rope_original_context,
        freq_base=shape.compressed_rope_freq_base,
        beta_fast=shape.rope_yarn_beta_fast,
        beta_slow=shape.rope_yarn_beta_slow,
    )
    ramp_mix = _rope_yarn_ramp(corr_low, corr_high, i0)
    angles = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix
    if inverse:
        angles = -angles
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x0 = pairs[..., 0]
    x1 = pairs[..., 1]
    rotated = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    out[..., -rotary_dim:] = rotated.reshape_as(tail)
    return out


def shared_kv_swa_attention_reference(
    queries: torch.Tensor,
    kv_rows: torch.Tensor,
    attn_sinks: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Sliding-window attention for DeepSeek V4 shared K=V latent rows.

    ``queries`` is ``(heads, head_dim)`` and ``kv_rows`` is
    ``(window, head_dim)``. The same KV rows are used as keys and values for
    every query head. ``attn_sinks`` contributes one extra softmax logit per
    head but has no value row, so it only changes the denominator.
    """
    if queries.ndim != 2:
        raise ValueError(f"queries must be 2-D; got {queries.ndim}-D")
    if kv_rows.ndim != 2:
        raise ValueError(f"kv_rows must be 2-D; got {kv_rows.ndim}-D")
    if kv_rows.shape[0] == 0:
        raise ValueError("kv_rows must contain at least one row")
    if queries.shape[1] != kv_rows.shape[1]:
        raise ValueError(
            "query and KV widths must match; "
            f"got {queries.shape[1]} and {kv_rows.shape[1]}"
        )
    if attn_sinks.shape != (queries.shape[0],):
        raise ValueError(
            f"attn_sinks shape must be ({queries.shape[0]},); "
            f"got {tuple(attn_sinks.shape)}"
        )

    scale = (
        float(softmax_scale)
        if softmax_scale is not None
        else 1.0 / math.sqrt(float(queries.shape[1]))
    )
    scores = queries.to(torch.float32).matmul(kv_rows.to(torch.float32).T) * scale
    sink_logits = attn_sinks.to(torch.float32).reshape(-1, 1)
    logits = torch.cat([scores, sink_logits], dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return probs[:, :-1].matmul(kv_rows.to(torch.float32))


def grouped_output_projection_reference(
    attention_output: torch.Tensor,
    output_a_weight: torch.Tensor,
    output_b_weight: torch.Tensor,
    *,
    output_groups: int,
) -> torch.Tensor:
    if attention_output.ndim != 2:
        raise ValueError(
            f"attention_output must be 2-D; got {attention_output.ndim}-D"
        )
    if output_a_weight.ndim != 2 or output_b_weight.ndim != 2:
        raise ValueError("output projection weights must be 2-D")
    if output_groups <= 0:
        raise ValueError(f"output_groups must be positive; got {output_groups}")
    num_heads, head_dim = attention_output.shape
    if num_heads % output_groups != 0:
        raise ValueError(
            "number of heads must be divisible by output_groups; "
            f"got {num_heads} and {output_groups}"
        )
    heads_per_group = num_heads // output_groups
    group_input = heads_per_group * head_dim
    if output_a_weight.shape[0] != group_input:
        raise ValueError(
            "output_a input dimension must match heads_per_group * head_dim; "
            f"got {output_a_weight.shape[0]} and {group_input}"
        )
    if output_a_weight.shape[1] % output_groups != 0:
        raise ValueError(
            "output_a output dimension must be divisible by output_groups; "
            f"got {output_a_weight.shape[1]} and {output_groups}"
        )
    rank_per_group = output_a_weight.shape[1] // output_groups
    if output_b_weight.shape[0] != output_groups * rank_per_group:
        raise ValueError(
            "output_b input dimension must match grouped output rank; "
            f"got {output_b_weight.shape[0]} and "
            f"{output_groups * rank_per_group}"
        )

    grouped = attention_output.to(torch.float32).reshape(
        output_groups,
        group_input,
    )
    a_by_group = output_a_weight.to(torch.float32).reshape(
        group_input,
        output_groups,
        rank_per_group,
    ).permute(1, 2, 0)
    low_rank = torch.einsum("gi,gri->gr", grouped, a_by_group)
    return output_b_weight.transpose(0, 1).to(torch.float32).matmul(
        low_rank.reshape(-1)
    )


def compressor_pair_projection_reference(
    hidden: torch.Tensor,
    kv_weight: torch.Tensor,
    gate_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D; got {hidden.ndim}-D")
    if kv_weight.ndim != 2 or gate_weight.ndim != 2:
        raise ValueError("compressor weights must be 2-D")
    if kv_weight.shape != gate_weight.shape:
        raise ValueError(
            "compressor KV and gate weights must have matching shapes; "
            f"got {tuple(kv_weight.shape)} and {tuple(gate_weight.shape)}"
        )
    hidden_f32 = hidden.to(torch.float32)
    if kv_weight.shape[1] == hidden.numel():
        return (
            kv_weight.to(torch.float32).matmul(hidden_f32),
            gate_weight.to(torch.float32).matmul(hidden_f32),
        )
    if kv_weight.shape[0] == hidden.numel():
        return (
            kv_weight.transpose(0, 1).to(torch.float32).matmul(hidden_f32),
            gate_weight.transpose(0, 1).to(torch.float32).matmul(hidden_f32),
        )
    raise ValueError(
        "compressor input dimension must match hidden size; "
        f"got {tuple(kv_weight.shape)} and {hidden.numel()}"
    )


def compressor_should_emit_reference(*, token_idx: int, ratio: int) -> bool:
    if token_idx < 0:
        raise ValueError(f"token_idx must be non-negative; got {token_idx}")
    if ratio <= 0:
        raise ValueError(f"ratio must be positive; got {ratio}")
    return (token_idx + 1) % ratio == 0


def indexer_query_projection_reference(
    qr_norm: torch.Tensor,
    indexer_q_b_weight: torch.Tensor,
    *,
    indexer_heads: int = 64,
    indexer_head_dim: int = 128,
) -> torch.Tensor:
    if qr_norm.ndim != 1:
        raise ValueError(f"qr_norm must be 1-D; got {qr_norm.ndim}-D")
    if indexer_q_b_weight.ndim != 2:
        raise ValueError(
            f"indexer_q_b_weight must be 2-D; got {indexer_q_b_weight.ndim}-D"
        )
    expected_width = indexer_heads * indexer_head_dim
    if indexer_q_b_weight.shape[0] != qr_norm.numel():
        raise ValueError(
            "indexer query input dimension must match q rank; "
            f"got {indexer_q_b_weight.shape[0]} and {qr_norm.numel()}"
        )
    if indexer_q_b_weight.shape[1] != expected_width:
        raise ValueError(
            "indexer query output dimension must match heads * head_dim; "
            f"got {indexer_q_b_weight.shape[1]} and {expected_width}"
        )
    projected = indexer_q_b_weight.transpose(0, 1).to(torch.float32).matmul(
        qr_norm.to(torch.float32)
    )
    return projected.reshape(indexer_heads, indexer_head_dim)


def indexer_weight_projection_reference(
    hidden: torch.Tensor,
    indexer_proj_weight: torch.Tensor,
) -> torch.Tensor:
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D; got {hidden.ndim}-D")
    if indexer_proj_weight.ndim != 2:
        raise ValueError(
            f"indexer_proj_weight must be 2-D; got {indexer_proj_weight.ndim}-D"
        )
    if indexer_proj_weight.shape[0] != hidden.numel():
        raise ValueError(
            "indexer projection input dimension must match hidden size; "
            f"got {indexer_proj_weight.shape[0]} and {hidden.numel()}"
        )
    return indexer_proj_weight.transpose(0, 1).to(torch.float32).matmul(
        hidden.to(torch.float32)
    )


def indexer_scores_reference(
    indexer_query: torch.Tensor,
    indexer_weights: torch.Tensor,
    compressed_rows: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    if indexer_query.ndim != 2:
        raise ValueError(f"indexer_query must be 2-D; got {indexer_query.ndim}-D")
    if indexer_weights.shape != (indexer_query.shape[0],):
        raise ValueError(
            f"indexer_weights shape must be ({indexer_query.shape[0]},); "
            f"got {tuple(indexer_weights.shape)}"
        )
    if compressed_rows.ndim != 2:
        raise ValueError(
            f"compressed_rows must be 2-D; got {compressed_rows.ndim}-D"
        )
    if compressed_rows.shape[1] != indexer_query.shape[1]:
        raise ValueError(
            "compressed row width must match indexer head dim; "
            f"got {compressed_rows.shape[1]} and {indexer_query.shape[1]}"
        )
    score_scale = (
        float(scale)
        if scale is not None
        else 1.0
        / math.sqrt(float(indexer_query.shape[0] * indexer_query.shape[1]))
    )
    per_head_scores = indexer_query.to(torch.float32).matmul(
        compressed_rows.to(torch.float32).T
    )
    per_head_scores = torch.clamp_min(per_head_scores, 0.0)
    weighted = indexer_weights.to(torch.float32).reshape(-1, 1) * per_head_scores
    return weighted.sum(dim=0) * score_scale


def indexer_topk_reference(scores: torch.Tensor, *, top_k: int) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D; got {scores.ndim}-D")
    if top_k <= 0 or top_k > scores.numel():
        raise ValueError(
            f"top_k must be in [1, {scores.numel()}]; got {top_k}"
        )
    return torch.topk(scores.to(torch.float32), k=top_k, sorted=True).indices


def compressor_pool_state_reference(
    state_kv: torch.Tensor,
    state_score: torch.Tensor,
    *,
    head_dim: int,
    ratio: int,
) -> torch.Tensor:
    if ratio <= 0:
        raise ValueError(f"ratio must be positive; got {ratio}")
    coff = 2 if ratio == 4 else 1
    width = coff * head_dim
    expected_shape = (coff * ratio, width)
    if state_kv.shape != expected_shape:
        raise ValueError(
            f"state_kv shape must be {expected_shape}; got {tuple(state_kv.shape)}"
        )
    if state_score.shape != expected_shape:
        raise ValueError(
            "state_score shape must match state_kv; "
            f"got {tuple(state_score.shape)}"
        )

    pooled = torch.empty(head_dim, dtype=torch.float32, device=state_kv.device)
    kv = state_kv.to(torch.float32)
    score = state_score.to(torch.float32)
    if ratio == 4:
        primary_scores = score[:ratio, :head_dim]
        carry_scores = score[ratio : 2 * ratio, head_dim : 2 * head_dim]
        primary_kv = kv[:ratio, :head_dim]
        carry_kv = kv[ratio : 2 * ratio, head_dim : 2 * head_dim]
        all_scores = torch.cat([primary_scores, carry_scores], dim=0)
        all_kv = torch.cat([primary_kv, carry_kv], dim=0)
    else:
        all_scores = score[:ratio, :head_dim]
        all_kv = kv[:ratio, :head_dim]

    weights = torch.softmax(all_scores, dim=0)
    pooled.copy_((weights * all_kv).sum(dim=0))
    return pooled


def compressor_update_state_reference(
    state_kv: torch.Tensor,
    state_score: torch.Tensor,
    kv_cur: torch.Tensor,
    score_cur: torch.Tensor,
    ape_weight: torch.Tensor,
    *,
    token_idx: int,
    head_dim: int,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if token_idx < 0:
        raise ValueError(f"token_idx must be non-negative; got {token_idx}")
    if ratio <= 0:
        raise ValueError(f"ratio must be positive; got {ratio}")
    coff = 2 if ratio == 4 else 1
    width = coff * head_dim
    expected_shape = (coff * ratio, width)
    if state_kv.shape != expected_shape or state_score.shape != expected_shape:
        raise ValueError(
            f"compressor state shape must be {expected_shape}; "
            f"got {tuple(state_kv.shape)} and {tuple(state_score.shape)}"
        )
    if kv_cur.shape != (width,) or score_cur.shape != (width,):
        raise ValueError(
            f"current compressor rows must have shape ({width},); "
            f"got {tuple(kv_cur.shape)} and {tuple(score_cur.shape)}"
        )
    pos_mod = token_idx % ratio
    if ape_weight.shape == (width, ratio):
        ape_pos = ape_weight[:, pos_mod]
    elif ape_weight.shape == (ratio, width):
        ape_pos = ape_weight[pos_mod, :]
    else:
        raise ValueError(
            f"ape_weight shape must be ({width}, {ratio}) or ({ratio}, {width}); "
            f"got {tuple(ape_weight.shape)}"
        )

    row = ratio + pos_mod if ratio == 4 else pos_mod
    next_kv = state_kv.to(torch.float32).clone()
    next_score = state_score.to(torch.float32).clone()
    next_kv[row] = kv_cur.to(torch.float32)
    next_score[row] = score_cur.to(torch.float32) + ape_pos.to(torch.float32)

    emitted: torch.Tensor | None = None
    if compressor_should_emit_reference(token_idx=token_idx, ratio=ratio):
        emitted = compressor_pool_state_reference(
            next_kv,
            next_score,
            head_dim=head_dim,
            ratio=ratio,
        )
        if ratio == 4:
            next_kv[:ratio] = next_kv[ratio : 2 * ratio]
            next_score[:ratio] = next_score[ratio : 2 * ratio]
            next_kv[ratio : 2 * ratio] = next_kv[:ratio]
            next_score[ratio : 2 * ratio] = next_score[:ratio]
    return next_kv, next_score, emitted
