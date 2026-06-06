# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def deepseek_linear_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D for batch=1; got {hidden.ndim}-D")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D; got {weight.ndim}-D")
    if weight.shape[1] != hidden.numel():
        raise ValueError(
            "weight columns must match hidden size; "
            f"got {weight.shape[1]} and {hidden.numel()}"
        )
    return weight.to(torch.float32).matmul(hidden.to(torch.float32))
