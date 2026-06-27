# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeepSeekV4FlashHyperConnectionState:
    mixed: torch.Tensor
    post: torch.Tensor
    combine: torch.Tensor


def sinkhorn_reference(
    scores: torch.Tensor,
    *,
    iterations: int = 20,
    eps: float = 1e-6,
) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D; got {scores.ndim}-D")
    if scores.shape[0] != scores.shape[1]:
        raise ValueError(f"scores must be square; got {tuple(scores.shape)}")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    out = torch.softmax(scores.to(torch.float32), dim=-1) + eps
    out = out / (out.sum(dim=0, keepdim=True) + eps)
    for _ in range(1, iterations):
        out = out / (out.sum(dim=-1, keepdim=True) + eps)
        out = out / (out.sum(dim=0, keepdim=True) + eps)
    return out


def hyper_connection_pre_reference(
    streams: torch.Tensor,
    fn_weight: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    sinkhorn_iterations: int = 20,
    eps: float = 1e-6,
) -> DeepSeekV4FlashHyperConnectionState:
    """Run DeepSeek V4 mHC pre-mix for one token.

    ``streams`` is ``(hc_mult, hidden_size)``. The GGUF ``hc_*_fn.weight`` is
    stored as ``(hc_mult * hidden_size, 2 * hc_mult + hc_mult ** 2)``.
    """
    if streams.ndim != 2:
        raise ValueError(f"streams must be 2-D; got {streams.ndim}-D")
    hc_mult, hidden_size = streams.shape
    mix_count = 2 * hc_mult + hc_mult * hc_mult
    flat_size = hc_mult * hidden_size
    if fn_weight.shape == (flat_size, mix_count):
        projection_weight = fn_weight.transpose(0, 1).to(torch.float32)
    elif fn_weight.shape == (mix_count, flat_size):
        projection_weight = fn_weight.to(torch.float32)
    else:
        raise ValueError(
            f"fn_weight shape must be ({flat_size}, {mix_count}) or "
            f"({mix_count}, {flat_size}); "
            f"got {tuple(fn_weight.shape)}"
        )
    if base.shape != (mix_count,):
        raise ValueError(
            f"base shape must be ({mix_count},); got {tuple(base.shape)}"
        )
    if scale.shape != (3,):
        raise ValueError(f"scale shape must be (3,); got {tuple(scale.shape)}")
    if eps <= 0:
        raise ValueError("eps must be positive")

    flat = streams.reshape(flat_size).to(torch.float32)
    rsqrt = torch.rsqrt(flat.pow(2).mean() + eps)
    mixes = projection_weight.matmul(flat) * rsqrt
    mixes = mixes * scale.to(torch.float32).repeat_interleave(
        torch.tensor([hc_mult, hc_mult, hc_mult * hc_mult])
    )
    mixes = mixes + base.to(torch.float32)

    pre = torch.sigmoid(mixes[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(mixes[hc_mult : 2 * hc_mult])
    combine_scores = mixes[2 * hc_mult :].reshape(hc_mult, hc_mult)
    combine = sinkhorn_reference(
        combine_scores,
        iterations=sinkhorn_iterations,
        eps=eps,
    )
    mixed = (pre.reshape(hc_mult, 1) * streams.to(torch.float32)).sum(dim=0)
    return DeepSeekV4FlashHyperConnectionState(
        mixed=mixed,
        post=post,
        combine=combine,
    )


def hyper_connection_post_reference(
    output: torch.Tensor,
    residual_streams: torch.Tensor,
    state: DeepSeekV4FlashHyperConnectionState,
) -> torch.Tensor:
    if output.ndim != 1:
        raise ValueError(f"output must be 1-D; got {output.ndim}-D")
    if residual_streams.ndim != 2:
        raise ValueError(
            f"residual_streams must be 2-D; got {residual_streams.ndim}-D"
        )
    hc_mult, hidden_size = residual_streams.shape
    if output.shape != (hidden_size,):
        raise ValueError(
            f"output shape must be ({hidden_size},); got {tuple(output.shape)}"
        )
    if state.post.shape != (hc_mult,):
        raise ValueError(
            f"state.post shape must be ({hc_mult},); got {tuple(state.post.shape)}"
        )
    if state.combine.shape != (hc_mult, hc_mult):
        raise ValueError(
            "state.combine shape must match residual stream count; "
            f"got {tuple(state.combine.shape)}"
        )

    residual_mix = state.combine.to(torch.float32).T.matmul(
        residual_streams.to(torch.float32)
    )
    return state.post.reshape(hc_mult, 1).to(torch.float32) * output.to(
        torch.float32
    ) + residual_mix
