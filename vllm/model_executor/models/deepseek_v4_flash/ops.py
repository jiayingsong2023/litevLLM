# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import torch.nn.functional as F


def rms_norm_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D; got {hidden.ndim}-D")
    if weight.ndim != 1:
        raise ValueError(f"weight must be 1-D; got {weight.ndim}-D")
    if weight.numel() != hidden.numel():
        raise ValueError(
            "weight length must match hidden size; "
            f"got {weight.numel()} and {hidden.numel()}"
        )

    hidden_f32 = hidden.to(torch.float32)
    variance = hidden_f32.pow(2).mean()
    return hidden_f32 * torch.rsqrt(variance + eps) * weight.to(torch.float32)


def silu_gate_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError(
            "gate and up shapes must match; "
            f"got {tuple(gate.shape)} and {tuple(up.shape)}"
        )
    return F.silu(gate.to(torch.float32)) * up.to(torch.float32)


def factorized_linear_reference(
    hidden: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D; got {hidden.ndim}-D")
    if a.ndim != 2:
        raise ValueError(f"a must be 2-D; got {a.ndim}-D")
    if b.ndim != 2:
        raise ValueError(f"b must be 2-D; got {b.ndim}-D")
    if a.shape[1] != hidden.numel():
        raise ValueError(
            "a columns must match hidden size; "
            f"got {a.shape[1]} and {hidden.numel()}"
        )
    if b.shape[1] != a.shape[0]:
        raise ValueError(
            "b columns must match a rows; "
            f"got {b.shape[1]} and {a.shape[0]}"
        )

    intermediate = a.to(torch.float32).matmul(hidden.to(torch.float32))
    return b.to(torch.float32).matmul(intermediate)
