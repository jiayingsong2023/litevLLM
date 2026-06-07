# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import torch

from .ops import factorized_linear_reference


def factorized_attention_projection_reference(
    hidden: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
) -> torch.Tensor:
    return factorized_linear_reference(hidden, a_weight, b_weight)


def split_combined_kv_reference(
    kv: torch.Tensor,
    *,
    key_width: int,
    value_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 1-D combined K/V vector using caller-supplied semantic widths."""
    if kv.ndim != 1:
        raise ValueError(f"kv must be 1-D; got {kv.ndim}-D")
    if key_width < 0 or value_width < 0:
        raise ValueError("widths must be non-negative")
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
