# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import torch

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
