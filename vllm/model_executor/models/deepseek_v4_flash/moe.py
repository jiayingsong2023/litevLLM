# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import torch.nn.functional as F


def _router_scores(logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    if scoring_func == "sqrtsoftplus":
        return F.softplus(logits).sqrt()
    if scoring_func == "softmax":
        return torch.softmax(logits, dim=0)
    raise ValueError(
        "scoring_func must be 'sqrtsoftplus' or 'softmax'; "
        f"got {scoring_func!r}"
    )


def topk_router_reference(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    top_k: int,
    scoring_func: str = "sqrtsoftplus",
    routed_scaling_factor: float = 1.5,
    correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D; got {hidden.ndim}-D")
    if router_weight.ndim != 2:
        raise ValueError(
            f"router_weight must be 2-D; got {router_weight.ndim}-D"
        )
    if hidden.numel() == 0:
        raise ValueError("hidden must contain at least one element")
    if router_weight.shape[1] != hidden.numel():
        raise ValueError(
            "router_weight columns must match hidden size; "
            f"got {router_weight.shape[1]} and {hidden.numel()}"
        )
    num_experts = router_weight.shape[0]
    if num_experts == 0:
        raise ValueError("router_weight must contain at least one expert row")
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(
            "top_k must be > 0 and <= number of experts; "
            f"got top_k={top_k}, experts={num_experts}"
        )
    if correction_bias is not None:
        if correction_bias.ndim != 1:
            raise ValueError(
                "correction_bias must be 1-D; "
                f"got {correction_bias.ndim}-D"
            )
        if correction_bias.numel() != num_experts:
            raise ValueError(
                "correction_bias size must match number of experts; "
                f"got {correction_bias.numel()} and {num_experts}"
            )

    logits = router_weight.to(torch.float32).matmul(hidden.to(torch.float32))
    scores = _router_scores(logits, scoring_func)
    selection_scores = scores
    if correction_bias is not None:
        selection_scores = selection_scores + correction_bias.to(torch.float32)
    expert_ids = torch.topk(selection_scores, k=top_k, sorted=True).indices
    weights = scores.gather(0, expert_ids)
    weights = weights / (weights.sum() + 1e-20)
    weights = weights * float(routed_scaling_factor)
    return expert_ids.to(torch.int64), weights.to(torch.float32)
