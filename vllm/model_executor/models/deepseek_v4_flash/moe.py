# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def topk_router_reference(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D; got {hidden.ndim}-D")
    if router_weight.ndim != 2:
        raise ValueError(
            f"router_weight must be 2-D; got {router_weight.ndim}-D"
        )
    if router_weight.shape[1] != hidden.numel():
        raise ValueError(
            "router_weight columns must match hidden size; "
            f"got {router_weight.shape[1]} and {hidden.numel()}"
        )
    num_experts = router_weight.shape[0]
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(
            "top_k must be > 0 and <= number of experts; "
            f"got top_k={top_k}, experts={num_experts}"
        )

    logits = router_weight.to(torch.float32).matmul(hidden.to(torch.float32))
    probs = torch.softmax(logits, dim=0)
    weights, expert_ids = torch.topk(probs, k=top_k, sorted=True)
    return expert_ids.to(torch.int64), weights.to(torch.float32)
