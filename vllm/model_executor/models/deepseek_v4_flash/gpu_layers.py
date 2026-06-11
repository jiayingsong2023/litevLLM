# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F

from .config import DEEPSEEK_V4_FLASH_SHAPE
from .gguf_reader import DeepSeekV4FlashTensor
from .gpu_backend import DeepSeekV4FlashGPUBackend
from .gpu_weight_staging import DeepSeekV4FlashGPUWeightStager
from .weight_store import (
    DeepSeekV4FlashGroupedExpertTensors,
    DeepSeekV4FlashHyperConnectionTensors,
    DeepSeekV4FlashLayerSemanticBindings,
)


class _SlidingLayerBackend(Protocol):
    def sliding_attention(
        self,
        *,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        attn_sinks: torch.Tensor | None,
        token_idx: int,
    ) -> torch.Tensor: ...

    def routed_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor: ...


def deepseek_v4_flash_rms_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not hidden.is_cuda or not weight.is_cuda:
        raise ValueError("DeepSeek V4 Flash RMSNorm inputs must be CUDA tensors")
    if hidden.shape[-1] != weight.numel():
        raise ValueError(
            "RMSNorm weight size must match hidden size; "
            f"got {weight.numel()} and {hidden.shape[-1]}"
        )
    hidden_f32 = hidden.to(torch.float32)
    variance = hidden_f32.pow(2).mean(dim=-1, keepdim=True)
    return hidden_f32 * torch.rsqrt(variance + eps) * weight.to(torch.float32)


def deepseek_v4_flash_staged_matrix_projection(
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not hidden.is_cuda or not weight.is_cuda:
        raise ValueError("DeepSeek V4 Flash projection inputs must be CUDA tensors")
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D for batch=1; got {hidden.ndim}-D")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D; got {weight.ndim}-D")

    hidden_f32 = hidden.to(torch.float32)
    weight_f32 = weight.to(torch.float32)
    if weight.shape[1] == hidden.numel():
        return weight_f32.matmul(hidden_f32)
    if weight.shape[0] == hidden.numel():
        return hidden_f32.matmul(weight_f32)
    raise ValueError(
        "projection weight must have one dimension matching hidden size; "
        f"got weight={tuple(weight.shape)} and hidden={hidden.numel()}"
    )


def deepseek_v4_flash_router_topk(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    top_k: int,
    correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = deepseek_v4_flash_staged_matrix_projection(hidden, router_weight)
    if logits.ndim != 1:
        raise ValueError(f"router logits must be 1-D; got {logits.ndim}-D")
    if top_k <= 0 or top_k > logits.numel():
        raise ValueError(
            "router top_k must be > 0 and <= number of experts; "
            f"got top_k={top_k}, experts={logits.numel()}"
        )
    scores = F.softplus(logits).sqrt()
    selection_scores = scores
    if correction_bias is not None:
        if not correction_bias.is_cuda:
            raise ValueError("router correction bias must be a CUDA tensor")
        if correction_bias.shape != scores.shape:
            raise ValueError(
                "router correction bias shape must match router scores; "
                f"got {tuple(correction_bias.shape)} and {tuple(scores.shape)}"
            )
        selection_scores = selection_scores + correction_bias.to(torch.float32)

    expert_ids = torch.topk(selection_scores, k=top_k, sorted=True).indices
    expert_weights = scores.gather(0, expert_ids)
    expert_weights = expert_weights / (expert_weights.sum() + 1e-20)
    return expert_ids.to(torch.int64), expert_weights.to(torch.float32)


def deepseek_v4_flash_residual_hyper_connection(
    residual: torch.Tensor,
    update: torch.Tensor,
    *,
    stager: DeepSeekV4FlashGPUWeightStager,
    hyper_connection: DeepSeekV4FlashHyperConnectionTensors | None,
) -> torch.Tensor:
    if not residual.is_cuda or not update.is_cuda:
        raise ValueError("DeepSeek V4 Flash residual inputs must be CUDA tensors")
    if residual.shape != update.shape:
        raise ValueError(
            "residual and update shapes must match; "
            f"got {tuple(residual.shape)} and {tuple(update.shape)}"
        )
    if hyper_connection is None:
        return residual.to(torch.float32) + update.to(torch.float32)

    if residual.ndim == 1:
        return residual.to(torch.float32) + update.to(torch.float32)
    if residual.ndim != 2:
        raise ValueError(
            "hyper-connection residual streams must be 1-D or 2-D; "
            f"got {residual.ndim}-D"
        )

    fn_weight = stager.stage_matrix(hyper_connection.fn)
    base = stager.stage_vector(hyper_connection.base)
    scale = stager.stage_vector(hyper_connection.scale)
    hc_mult, hidden_size = residual.shape
    mix_count = 2 * hc_mult + hc_mult * hc_mult
    flat_size = hc_mult * hidden_size
    if fn_weight.shape not in ((flat_size, mix_count), (mix_count, flat_size)):
        raise ValueError(
            "hyper-connection fn tensor shape does not match residual streams; "
            f"got {tuple(fn_weight.shape)}, expected ({flat_size}, {mix_count})"
        )
    if base.numel() != mix_count:
        raise ValueError(
            "hyper-connection base size does not match residual streams; "
            f"got {base.numel()}, expected {mix_count}"
        )
    if scale.numel() != 3:
        raise ValueError(
            f"hyper-connection scale must contain 3 values; got {scale.numel()}"
        )

    flat = residual.reshape(flat_size).to(torch.float32)
    rsqrt = torch.rsqrt(flat.pow(2).mean() + 1e-6)
    mixes = deepseek_v4_flash_staged_matrix_projection(flat, fn_weight) * rsqrt
    repeat_counts = torch.tensor(
        [hc_mult, hc_mult, hc_mult * hc_mult],
        device=scale.device,
    )
    mixes = mixes * scale.to(torch.float32).repeat_interleave(repeat_counts)
    mixes = mixes + base.to(torch.float32)
    post = torch.sigmoid(mixes[hc_mult : 2 * hc_mult]) + 1e-6
    combine_scores = mixes[2 * hc_mult :].reshape(hc_mult, hc_mult)
    combine = torch.softmax(combine_scores, dim=-1)
    for _ in range(20):
        combine = combine / (combine.sum(dim=-1, keepdim=True) + 1e-6)
        combine = combine / (combine.sum(dim=0, keepdim=True) + 1e-6)
    residual_mix = combine.matmul(residual.to(torch.float32))
    return post.reshape(hc_mult, 1) * update.to(torch.float32) + residual_mix


def deepseek_v4_flash_sliding_layer_forward(
    hidden: torch.Tensor,
    *,
    layer: DeepSeekV4FlashLayerSemanticBindings,
    stager: DeepSeekV4FlashGPUWeightStager,
    backend: DeepSeekV4FlashGPUBackend | _SlidingLayerBackend,
    token_idx: int,
    kv_rows: torch.Tensor | None = None,
    router_top_k: int = DEEPSEEK_V4_FLASH_SHAPE.num_experts_per_tok,
) -> torch.Tensor:
    if not hidden.is_cuda:
        raise ValueError("DeepSeek V4 Flash sliding layer hidden must be CUDA")
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D for batch=1; got {hidden.ndim}-D")

    attention_norm = stager.stage_vector(
        _required_tensor(layer.attention_norm, "attention_norm")
    )
    attention_query = stager.stage_matrix(
        _required_tensor(layer.attention_query, "attention_query")
    )
    attention_output = stager.stage_matrix(
        _required_tensor(layer.attention_output, "attention_output")
    )
    ffn_norm = stager.stage_vector(_required_tensor(layer.ffn_norm, "ffn_norm"))
    router = stager.stage_matrix(_required_tensor(layer.router, "router"))
    grouped_experts = _required_grouped_experts(layer.grouped_experts)

    attn_input = deepseek_v4_flash_rms_norm(hidden, attention_norm)
    query = deepseek_v4_flash_staged_matrix_projection(attn_input, attention_query)
    if layer.attention_query_a_norm is not None:
        query = deepseek_v4_flash_rms_norm(
            query,
            stager.stage_vector(layer.attention_query_a_norm),
        )
    if layer.attention_query_b is not None:
        query = deepseek_v4_flash_staged_matrix_projection(
            query,
            stager.stage_matrix(layer.attention_query_b),
        )
    query = _adapt_attention_width_for_output_projection(
        query,
        attention_output,
        output_size=hidden.numel(),
    )
    if kv_rows is None:
        kv_rows = query.reshape(1, query.numel())
    if not kv_rows.is_cuda:
        raise ValueError("DeepSeek V4 Flash sliding attention KV rows must be CUDA")
    attn_sinks = (
        stager.stage_vector(layer.attention_sinks)
        if layer.attention_sinks is not None
        else None
    )
    attn_update = backend.sliding_attention(
        query=query,
        kv_rows=kv_rows,
        attn_sinks=attn_sinks,
        token_idx=token_idx,
    )
    attn_update = deepseek_v4_flash_staged_matrix_projection(
        attn_update,
        attention_output,
    )
    hidden_after_attn = deepseek_v4_flash_residual_hyper_connection(
        hidden,
        attn_update,
        stager=stager,
        hyper_connection=layer.attention_hyper_connection,
    )

    ffn_input = deepseek_v4_flash_rms_norm(hidden_after_attn, ffn_norm)
    correction_bias = (
        stager.stage_vector(layer.expert_probs_bias)
        if layer.expert_probs_bias is not None
        else None
    )
    expert_ids, expert_weights = deepseek_v4_flash_router_topk(
        ffn_input,
        router,
        top_k=router_top_k,
        correction_bias=correction_bias,
    )
    moe_update = _run_staged_routed_experts(
        ffn_input,
        expert_ids,
        expert_weights,
        grouped_experts=grouped_experts,
        stager=stager,
        backend=backend,
    )
    return deepseek_v4_flash_residual_hyper_connection(
        hidden_after_attn,
        moe_update,
        stager=stager,
        hyper_connection=layer.ffn_hyper_connection,
    )


def _run_staged_routed_experts(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    *,
    grouped_experts: DeepSeekV4FlashGroupedExpertTensors,
    stager: DeepSeekV4FlashGPUWeightStager,
    backend: DeepSeekV4FlashGPUBackend | _SlidingLayerBackend,
) -> torch.Tensor:
    output: torch.Tensor | None = None
    for expert_id_tensor, expert_weight in zip(expert_ids, expert_weights, strict=True):
        expert_id = int(expert_id_tensor.item())
        staged = stager.stage_grouped_expert(grouped_experts, expert_id)
        expert_output = backend.routed_expert_gemm(
            hidden=hidden,
            gate_weight=staged.gate,
            up_weight=staged.up,
            down_weight=staged.down,
        ).to(torch.float32)
        if output is None:
            output = torch.zeros_like(expert_output, dtype=torch.float32)
        if output.shape != expert_output.shape:
            raise ValueError(
                "routed expert outputs must share one shape; "
                f"got {tuple(output.shape)} and {tuple(expert_output.shape)}"
            )
        output = output + expert_weight.to(torch.float32) * expert_output
    if output is None:
        raise ValueError("DeepSeek V4 Flash routed MoE selected no experts")
    return output


def _adapt_attention_width_for_output_projection(
    query: torch.Tensor,
    attention_output: torch.Tensor,
    *,
    output_size: int,
) -> torch.Tensor:
    if attention_output.shape[0] == output_size:
        projection_input_size = attention_output.shape[1]
    elif attention_output.shape[1] == output_size:
        projection_input_size = attention_output.shape[0]
    else:
        projection_input_size = query.numel()

    if query.numel() == projection_input_size:
        return query
    if query.numel() > projection_input_size:
        return query[:projection_input_size]
    raise ValueError(
        "attention query width is smaller than output projection input; "
        f"got query={query.numel()}, projection_input={projection_input_size}"
    )


def _required_tensor(
    tensor: DeepSeekV4FlashTensor | None,
    semantic_name: str,
) -> DeepSeekV4FlashTensor:
    if tensor is None:
        raise ValueError(f"DeepSeek V4 Flash sliding layer missing {semantic_name}")
    return tensor


def _required_grouped_experts(
    tensors: DeepSeekV4FlashGroupedExpertTensors | None,
) -> DeepSeekV4FlashGroupedExpertTensors:
    if tensors is None:
        raise ValueError("DeepSeek V4 Flash sliding layer missing grouped_experts")
    return tensors


__all__ = [
    "deepseek_v4_flash_residual_hyper_connection",
    "deepseek_v4_flash_rms_norm",
    "deepseek_v4_flash_router_topk",
    "deepseek_v4_flash_sliding_layer_forward",
    "deepseek_v4_flash_staged_matrix_projection",
]
