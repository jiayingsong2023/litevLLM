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
    """Split an already-projected 1-D K/V vector using known semantic widths.

    This helper does not derive the target GGUF ``attn_kv.weight`` split. The
    observed real tensor is 512-wide, and its key/value semantic split must be
    resolved before real attention execution.
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
