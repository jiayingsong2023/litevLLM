# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from .ops import silu_gate_reference


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


def grouped_expert_reference(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D; got {hidden.ndim}-D")
    if gate_weight.ndim != 2:
        raise ValueError(f"gate_weight must be 2-D; got {gate_weight.ndim}-D")
    if up_weight.ndim != 2:
        raise ValueError(f"up_weight must be 2-D; got {up_weight.ndim}-D")
    if down_weight.ndim != 2:
        raise ValueError(f"down_weight must be 2-D; got {down_weight.ndim}-D")
    if hidden.numel() == 0:
        raise ValueError("hidden must contain at least one element")
    if gate_weight.shape[1] != hidden.numel():
        raise ValueError(
            "gate_weight columns must match hidden size; "
            f"got {gate_weight.shape[1]} and {hidden.numel()}"
        )
    if up_weight.shape != gate_weight.shape:
        raise ValueError(
            "up_weight shape must match gate_weight shape; "
            f"got {tuple(up_weight.shape)} and {tuple(gate_weight.shape)}"
        )
    if down_weight.shape[1] != gate_weight.shape[0]:
        raise ValueError(
            "down_weight columns must match expert intermediate size; "
            f"got {down_weight.shape[1]} and {gate_weight.shape[0]}"
        )

    hidden_f32 = hidden.to(torch.float32)
    gate = gate_weight.to(torch.float32).matmul(hidden_f32)
    up = up_weight.to(torch.float32).matmul(hidden_f32)
    activated = silu_gate_reference(gate, up)
    return down_weight.to(torch.float32).matmul(activated)


def routed_moe_reference(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    expert_runner: Callable[[int, torch.Tensor], torch.Tensor],
    *,
    top_k: int,
    correction_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    expert_ids, weights = topk_router_reference(
        hidden,
        router_weight,
        top_k=top_k,
        correction_bias=correction_bias,
    )
    if expert_ids.numel() == 0:
        raise ValueError("routed MoE selected no experts")

    output: torch.Tensor | None = None
    for expert_id, weight in zip(expert_ids.tolist(), weights, strict=True):
        expert_output = expert_runner(expert_id, hidden).to(torch.float32)
        if output is None:
            output = torch.zeros_like(expert_output, dtype=torch.float32)
        if expert_output.shape != output.shape:
            raise ValueError(
                "expert outputs must share one shape; "
                f"got {tuple(expert_output.shape)} and {tuple(output.shape)}"
            )
        output = output + weight * expert_output
    if output is None:
        raise ValueError("routed MoE selected no experts")
    return output
