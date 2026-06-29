# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol, cast

import torch
import torch.nn.functional as F

from vllm.kernels.triton.deepseek_v4_flash.cache import (
    DeepSeekV4CacheUpdateInputs,
    deepseek_v4_cache_update,
)
from vllm.kernels.triton.deepseek_v4_flash.q8_linear import (
    q8_0_raw_gate_up_activation,
    q8_0_raw_linear,
)

from .attention import (
    apply_deepseek_layer_rope_to_tail_reference,
    apply_precomputed_rope_to_tail,
    per_head_rms_norm_reference,
    shared_kv_swa_attention_reference,
)
from .config import DEEPSEEK_V4_FLASH_SHAPE, layer_compress_ratio
from .gguf_reader import GGML_TYPE_Q8_0, DeepSeekV4FlashTensor
from .gpu_backend import DeepSeekV4FlashGPUBackend
from .gpu_runtime import DeepSeekV4FlashGPURequestState
from .gpu_weight_staging import (
    DeepSeekV4FlashGPUWeightStager,
    DeepSeekV4FlashSelectedExpertPayloads,
    DeepSeekV4FlashStagedQuantizedExpertPayload,
)
from .ops import deepseek_fp8_kv_qat_reference, deepseek_indexer_qat_reference
from .weight_store import (
    DeepSeekV4FlashCompressorTensors,
    DeepSeekV4FlashGroupedExpertTensors,
    DeepSeekV4FlashHyperConnectionTensors,
    DeepSeekV4FlashLayerSemanticBindings,
)

_Q8_0_BLOCK_SIZE = 32
_Q8_0_BLOCK_BYTES = 2 + _Q8_0_BLOCK_SIZE


@dataclass
class _CompressorRuntimeState:
    kv_rows: torch.Tensor
    score_rows: torch.Tensor
    count: int = 0


@dataclass(frozen=True)
class _GPUHyperConnectionState:
    mixed: torch.Tensor
    post: torch.Tensor
    combine: torch.Tensor


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

    def quantized_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_payload: DeepSeekV4FlashStagedQuantizedExpertPayload,
        up_payload: DeepSeekV4FlashStagedQuantizedExpertPayload,
        down_payload: DeepSeekV4FlashStagedQuantizedExpertPayload,
    ) -> torch.Tensor: ...

    def compressed_attention(
        self,
        *,
        query: torch.Tensor,
        compressed_rows: torch.Tensor,
        selected_rows: torch.Tensor,
    ) -> torch.Tensor: ...


def _stager_profile_section(
    stager: DeepSeekV4FlashGPUWeightStager,
    name: str,
    **metadata: object,
) -> Iterator[None]:
    profiler = getattr(stager, "profiler", None)
    section = getattr(profiler, "section", None)
    if not callable(section):
        return nullcontext()
    return section(name, **metadata)


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
    """Project ``hidden`` with a staged matrix.

    Staged decoded matrices use row-major linear layout
    ``(out_features, in_features)``. For non-square legacy/fp16 tensors whose
    decoded shape is still ``(in_features, out_features)``, only use the
    transpose convention when the row-major convention is impossible.
    """
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


def deepseek_v4_flash_q8_0_tensor_projection(
    hidden: torch.Tensor,
    tensor: DeepSeekV4FlashTensor,
    stager: DeepSeekV4FlashGPUWeightStager,
) -> torch.Tensor:
    """Project a vector with a raw GGUF Q8_0 tensor without fp32 matrix decode."""
    if not hidden.is_cuda:
        raise ValueError("DeepSeek V4 Flash Q8 projection hidden must be CUDA")
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D for Q8 projection; got {hidden.ndim}-D")
    if tensor.tensor_type != GGML_TYPE_Q8_0:
        raise ValueError(
            "DeepSeek V4 Flash Q8 projection expects GGML_TYPE_Q8_0; "
            f"got {tensor.tensor_type}"
        )
    columns, rows = tensor.dims
    if hidden.numel() != columns:
        raise ValueError(
            "Q8 projection hidden width must match tensor input columns; "
            f"got hidden={hidden.numel()} and columns={columns}"
        )
    raw_payload = _stage_q8_0_raw_tensor(tensor, stager)
    expected_bytes = rows * (columns // _Q8_0_BLOCK_SIZE) * _Q8_0_BLOCK_BYTES
    if raw_payload.numel() != expected_bytes:
        raise ValueError(
            "staged raw Q8 payload size must match tensor dims; "
            f"got {raw_payload.numel()} bytes, expected {expected_bytes}"
        )

    return q8_0_raw_linear(
        raw_payload,
        hidden,
        rows=rows,
        columns=columns,
        block_size=_Q8_0_BLOCK_SIZE,
    )


def _stage_q8_0_raw_tensor(
    tensor: DeepSeekV4FlashTensor,
    stager: DeepSeekV4FlashGPUWeightStager,
) -> torch.Tensor:
    if tensor.tensor_type != GGML_TYPE_Q8_0:
        raise ValueError(
            "DeepSeek V4 Flash Q8 staging expects GGML_TYPE_Q8_0; "
            f"got {tensor.tensor_type}"
        )
    if len(tensor.dims) != 2:
        raise ValueError(f"Q8 tensor must be 2-D; got {tensor.dims}")
    columns, rows = tensor.dims
    if columns % _Q8_0_BLOCK_SIZE != 0:
        raise ValueError(
            f"Q8 tensor columns must be divisible by 32; got columns={columns}"
        )
    return stager.stage_q8_raw_payload(tensor)


def _project_tensor(
    hidden: torch.Tensor,
    tensor: DeepSeekV4FlashTensor,
    stager: DeepSeekV4FlashGPUWeightStager,
) -> torch.Tensor:
    if _can_project_q8_0(hidden, tensor):
        return deepseek_v4_flash_q8_0_tensor_projection(hidden, tensor, stager)
    return deepseek_v4_flash_staged_matrix_projection(
        hidden,
        stager.stage_matrix(tensor),
    )


def _can_project_q8_0(hidden: torch.Tensor, tensor: DeepSeekV4FlashTensor) -> bool:
    return (
        tensor.tensor_type == GGML_TYPE_Q8_0
        and len(tensor.dims) == 2
        and hidden.ndim == 1
        and hidden.numel() == tensor.dims[0]
        and tensor.dims[0] % _Q8_0_BLOCK_SIZE == 0
    )


def deepseek_v4_flash_router_topk(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    top_k: int,
    correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = deepseek_v4_flash_staged_matrix_projection(hidden, router_weight)
    return _router_topk_from_logits(
        logits,
        top_k=top_k,
        correction_bias=correction_bias,
    )


def _router_topk_from_logits(
    logits: torch.Tensor,
    *,
    top_k: int,
    correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    expert_weights = expert_weights * 1.5
    return expert_ids.to(torch.int64), expert_weights.to(torch.float32)


def _has_real_sliding_attention_tensors(
    layer: DeepSeekV4FlashLayerSemanticBindings,
) -> bool:
    return all(
        tensor is not None
        for tensor in (
            layer.attention_query_a,
            layer.attention_query_a_norm,
            layer.attention_query_b,
            layer.attention_key_value,
            layer.attention_key_value_a_norm,
            layer.attention_sinks,
            layer.attention_output_a,
            layer.attention_output_b,
        )
    )


def _run_real_sliding_attention(
    attn_input: torch.Tensor,
    *,
    layer: DeepSeekV4FlashLayerSemanticBindings,
    stager: DeepSeekV4FlashGPUWeightStager,
    backend: DeepSeekV4FlashGPUBackend | _SlidingLayerBackend,
    state: DeepSeekV4FlashGPURequestState | None,
    token_idx: int,
    kv_rows: torch.Tensor | None,
    extra_kv_rows: torch.Tensor | None = None,
    use_reference_rope: bool = False,
) -> torch.Tensor:
    if not attn_input.is_cuda:
        raise ValueError("DeepSeek V4 Flash real sliding input must be CUDA")
    query_a = _required_tensor(layer.attention_query_a, "attention_query_a")
    query_a_norm = _required_tensor(
        layer.attention_query_a_norm,
        "attention_query_a_norm",
    )
    query_b = _required_tensor(layer.attention_query_b, "attention_query_b")
    kv_tensor = _required_tensor(layer.attention_key_value, "attention_key_value")
    kv_norm = _required_tensor(
        layer.attention_key_value_a_norm,
        "attention_key_value_a_norm",
    )
    attn_sinks = _required_tensor(layer.attention_sinks, "attention_sinks")

    q_latent = _project_tensor(attn_input, query_a, stager)
    q_latent = deepseek_v4_flash_rms_norm(
        q_latent,
        stager.stage_vector(query_a_norm),
    )
    query = _project_tensor(q_latent, query_b, stager).reshape(
        DEEPSEEK_V4_FLASH_SHAPE.num_attention_heads,
        DEEPSEEK_V4_FLASH_SHAPE.head_dim,
    )
    query = per_head_rms_norm_reference(query)
    if use_reference_rope or state is None:
        query = apply_deepseek_layer_rope_to_tail_reference(
            query,
            token_idx=token_idx,
            layer_idx=layer.layer_index,
            rotary_dim=DEEPSEEK_V4_FLASH_SHAPE.rotary_dim,
        )
    else:
        cos, sin = state.rope_tables_for(layer.layer_index, token_idx)
        query = apply_precomputed_rope_to_tail(query, cos, sin)

    current_kv = _project_tensor(attn_input, kv_tensor, stager)
    current_kv = deepseek_v4_flash_rms_norm(
        current_kv,
        stager.stage_vector(kv_norm),
    )
    if use_reference_rope or state is None:
        current_kv = apply_deepseek_layer_rope_to_tail_reference(
            current_kv,
            token_idx=token_idx,
            layer_idx=layer.layer_index,
            rotary_dim=DEEPSEEK_V4_FLASH_SHAPE.rotary_dim,
        )
    else:
        cos, sin = state.rope_tables_for(layer.layer_index, token_idx)
        current_kv = apply_precomputed_rope_to_tail(current_kv, cos, sin)
    current_kv = deepseek_fp8_kv_qat_reference(
        current_kv,
        head_dim=DEEPSEEK_V4_FLASH_SHAPE.head_dim,
        rotary_dim=DEEPSEEK_V4_FLASH_SHAPE.rotary_dim,
    )
    if current_kv.shape != (DEEPSEEK_V4_FLASH_SHAPE.head_dim,):
        raise ValueError(
            "DeepSeek V4 Flash sliding KV latent shape must be "
            f"({DEEPSEEK_V4_FLASH_SHAPE.head_dim},); got {tuple(current_kv.shape)}"
        )

    if kv_rows is None:
        if state is None:
            kv_rows = current_kv.reshape(1, -1)
        else:
            cache_dtype = state.raw_kv_cache.raw_keys.dtype
            state.raw_kv_cache.append_raw(
                layer.layer_index,
                token_idx,
                current_kv.to(dtype=cache_dtype),
                current_kv.to(dtype=cache_dtype),
            )
            kv_rows, _values = state.raw_kv_cache.read_raw_window(
                layer.layer_index,
                token_idx,
                DEEPSEEK_V4_FLASH_SHAPE.sliding_window,
            )
    if extra_kv_rows is not None:
        kv_rows = torch.cat(
            [kv_rows.to(torch.float32), extra_kv_rows.to(torch.float32)],
            dim=0,
        )
    if not kv_rows.is_cuda:
        raise ValueError("DeepSeek V4 Flash real sliding KV rows must be CUDA")

    staged_sinks = stager.stage_vector(attn_sinks)
    fused_attn = getattr(backend, "fused_sliding_window_attention", None)
    context: torch.Tensor | None = None
    if callable(fused_attn):
        try:
            context = fused_attn(
                query=query,
                kv_rows=kv_rows,
                attn_sinks=staged_sinks,
                token_idx=token_idx,
            )
        except (RuntimeError, NotImplementedError, ValueError):
            context = None
    if context is None:
        context = shared_kv_swa_attention_reference(
            query,
            kv_rows,
            staged_sinks,
        )
    if use_reference_rope or state is None:
        context = apply_deepseek_layer_rope_to_tail_reference(
            context,
            token_idx=token_idx,
            layer_idx=layer.layer_index,
            rotary_dim=DEEPSEEK_V4_FLASH_SHAPE.rotary_dim,
            inverse=True,
        )
    else:
        cos, sin = state.rope_tables_for(layer.layer_index, token_idx)
        context = apply_precomputed_rope_to_tail(
            context,
            cos,
            sin,
            inverse=True,
        )
    return _project_sliding_attention_output(
        context.reshape(-1),
        None,
        attention_output_a=layer.attention_output_a,
        attention_output_b=layer.attention_output_b,
        stager=stager,
    )


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
        raise ValueError(
            "DeepSeek V4 Flash 1-D hyper-connection is unsupported in the "
            "sliding GPU layer; pass stream-shaped hidden state in a later "
            "full-layer path"
        )
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
    post = 2.0 * torch.sigmoid(mixes[hc_mult : 2 * hc_mult])
    combine_scores = mixes[2 * hc_mult :].reshape(hc_mult, hc_mult)
    combine = torch.softmax(combine_scores, dim=-1) + 1e-6
    combine = combine / (combine.sum(dim=0, keepdim=True) + 1e-6)
    for _ in range(1, 20):
        combine = combine / (combine.sum(dim=-1, keepdim=True) + 1e-6)
        combine = combine / (combine.sum(dim=0, keepdim=True) + 1e-6)
    residual_mix = combine.T.matmul(residual.to(torch.float32))
    return post.reshape(hc_mult, 1) * update.to(torch.float32) + residual_mix


def deepseek_v4_flash_sliding_layer_forward(
    hidden: torch.Tensor,
    *,
    layer: DeepSeekV4FlashLayerSemanticBindings,
    stager: DeepSeekV4FlashGPUWeightStager,
    backend: DeepSeekV4FlashGPUBackend | _SlidingLayerBackend,
    state: DeepSeekV4FlashGPURequestState | None = None,
    token_idx: int,
    token_id: int | None = None,
    token_id_tensor: torch.Tensor | None = None,
    kv_rows: torch.Tensor | None = None,
    extra_kv_rows: torch.Tensor | None = None,
    router_top_k: int = DEEPSEEK_V4_FLASH_SHAPE.num_experts_per_tok,
    use_reference_rope: bool = False,
) -> torch.Tensor:
    if not hidden.is_cuda:
        raise ValueError("DeepSeek V4 Flash sliding layer hidden must be CUDA")
    if hidden.ndim not in (1, 2):
        raise ValueError(
            f"hidden must be 1-D or mHC stream-shaped 2-D; got {hidden.ndim}-D"
        )

    attention_norm = stager.stage_vector(
        _required_tensor(layer.attention_norm, "attention_norm")
    )
    use_two_stage_output = (
        layer.attention_output_a is not None and layer.attention_output_b is not None
    )
    attention_output = (
        None
        if use_two_stage_output
        else _required_tensor(layer.attention_output, "attention_output")
    )
    ffn_norm = stager.stage_vector(_required_tensor(layer.ffn_norm, "ffn_norm"))
    grouped_experts = _required_grouped_experts(layer.grouped_experts)

    uses_hyper_connection = (
        layer.attention_hyper_connection is not None
        or layer.ffn_hyper_connection is not None
    )
    attention_residual_streams: torch.Tensor | None = None
    attention_hc_state: _GPUHyperConnectionState | None = None
    if uses_hyper_connection:
        stream_hc = layer.attention_hyper_connection or layer.ffn_hyper_connection
        if stream_hc is None:
            raise AssertionError("hyper-connection state was not validated")
        attention_residual_streams = _ensure_hyper_connection_streams(
            hidden,
            stager=stager,
            hyper_connection=stream_hc,
        )
        if layer.attention_hyper_connection is None:
            attn_source = attention_residual_streams.mean(dim=0).to(torch.float32)
        else:
            attention_hc_state = _hyper_connection_pre_cuda(
                attention_residual_streams,
                stager=stager,
                hyper_connection=layer.attention_hyper_connection,
            )
            attn_source = attention_hc_state.mixed
    else:
        if hidden.ndim != 1:
            raise ValueError(
                "DeepSeek V4 Flash sliding layer accepts 2-D hidden only when "
                "layer hyper-connection tensors are present"
            )
        attn_source = hidden

    with _stager_profile_section(
        stager,
        "layer_attention_norm",
        layer_idx=layer.layer_index,
        layer_type="sliding",
    ):
        attn_input = deepseek_v4_flash_rms_norm(attn_source, attention_norm)
    with _stager_profile_section(
        stager,
        "layer_attention",
        layer_idx=layer.layer_index,
        layer_type="sliding",
    ):
        if _has_real_sliding_attention_tensors(layer):
            attn_update = _run_real_sliding_attention(
                attn_input,
                layer=layer,
                stager=stager,
                backend=backend,
                state=state,
                token_idx=token_idx,
                kv_rows=kv_rows,
                extra_kv_rows=extra_kv_rows,
                use_reference_rope=use_reference_rope,
            )
        else:
            query = _project_tensor(
                attn_input,
                _required_tensor(layer.attention_query, "attention_query"),
                stager,
            )
            if layer.attention_query_a_norm is not None:
                query = deepseek_v4_flash_rms_norm(
                    query,
                    stager.stage_vector(layer.attention_query_a_norm),
                )
            if layer.attention_query_b is not None:
                query = _project_tensor(query, layer.attention_query_b, stager)
            if kv_rows is None:
                kv_rows = query.reshape(1, query.numel())
            if not kv_rows.is_cuda:
                raise ValueError(
                    "DeepSeek V4 Flash sliding attention KV rows must be CUDA"
                )
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
            attn_update = _project_sliding_attention_output(
                attn_update,
                attention_output,
                attention_output_a=layer.attention_output_a,
                attention_output_b=layer.attention_output_b,
                stager=stager,
            )
    if uses_hyper_connection:
        if attention_residual_streams is None:
            raise AssertionError("attention residual streams were not initialized")
        if attention_hc_state is None:
            hidden_after_attn = attention_residual_streams + attn_update.reshape(1, -1)
        else:
            hidden_after_attn = _hyper_connection_post_cuda(
                attn_update,
                attention_residual_streams,
                attention_hc_state,
            )
    else:
        hidden_after_attn = deepseek_v4_flash_residual_hyper_connection(
            attn_source,
            attn_update,
            stager=stager,
            hyper_connection=None,
        )

    if uses_hyper_connection:
        if hidden_after_attn.ndim != 2:
            raise ValueError("mHC sliding path must carry stream-shaped hidden")
        ffn_residual_streams = hidden_after_attn
        if layer.ffn_hyper_connection is None:
            ffn_source = ffn_residual_streams.mean(dim=0).to(torch.float32)
            ffn_hc_state = None
        else:
            ffn_hc_state = _hyper_connection_pre_cuda(
                ffn_residual_streams,
                stager=stager,
                hyper_connection=layer.ffn_hyper_connection,
            )
            ffn_source = ffn_hc_state.mixed
    else:
        ffn_source = hidden_after_attn
        ffn_hc_state = None
        ffn_residual_streams = None

    with _stager_profile_section(
        stager,
        "layer_ffn_norm",
        layer_idx=layer.layer_index,
        layer_type="sliding",
    ):
        ffn_input = deepseek_v4_flash_rms_norm(ffn_source, ffn_norm)
    with _stager_profile_section(
        stager,
        "layer_moe",
        layer_idx=layer.layer_index,
        layer_type="sliding",
    ):
        moe_update = _run_sliding_moe(
            ffn_input,
            layer=layer,
            grouped_experts=grouped_experts,
            stager=stager,
            backend=backend,
            state=state,
            token_id=token_idx if token_id is None else token_id,
            token_id_tensor=token_id_tensor,
            router_top_k=router_top_k,
        )
    if uses_hyper_connection:
        if ffn_residual_streams is None:
            raise AssertionError("FFN residual streams were not initialized")
        if ffn_hc_state is None:
            return ffn_residual_streams + moe_update.reshape(1, -1)
        return _hyper_connection_post_cuda(
            moe_update,
            ffn_residual_streams,
            ffn_hc_state,
        )
    return deepseek_v4_flash_residual_hyper_connection(
        hidden_after_attn,
        moe_update,
        stager=stager,
        hyper_connection=None,
    )


def deepseek_v4_flash_layer_forward(
    hidden: torch.Tensor,
    *,
    layer: DeepSeekV4FlashLayerSemanticBindings,
    stager: DeepSeekV4FlashGPUWeightStager,
    backend: DeepSeekV4FlashGPUBackend | _SlidingLayerBackend,
    state: DeepSeekV4FlashGPURequestState,
    token_idx: int,
    token_id: int | None = None,
    token_id_tensor: torch.Tensor | None = None,
    kv_rows: torch.Tensor | None = None,
    extra_kv_rows: torch.Tensor | None = None,
    router_top_k: int = DEEPSEEK_V4_FLASH_SHAPE.num_experts_per_tok,
    use_reference_rope: bool = False,
) -> torch.Tensor:
    ratio = layer_compress_ratio(layer.layer_index)
    if ratio == 0:
        return deepseek_v4_flash_sliding_layer_forward(
            hidden,
            layer=layer,
            stager=stager,
            backend=backend,
            state=state,
            token_idx=token_idx,
            token_id=token_id,
            token_id_tensor=token_id_tensor,
            kv_rows=kv_rows,
            extra_kv_rows=extra_kv_rows,
            router_top_k=router_top_k,
            use_reference_rope=use_reference_rope,
        )
    return deepseek_v4_flash_compressed_layer_forward(
        hidden,
        layer=layer,
        stager=stager,
        backend=backend,
        state=state,
        token_idx=token_idx,
        token_id=token_id,
        token_id_tensor=token_id_tensor,
        kv_rows=kv_rows,
        extra_kv_rows=extra_kv_rows,
        router_top_k=router_top_k,
        use_reference_rope=use_reference_rope,
    )


def deepseek_v4_flash_compressed_layer_forward(
    hidden: torch.Tensor,
    *,
    layer: DeepSeekV4FlashLayerSemanticBindings,
    stager: DeepSeekV4FlashGPUWeightStager,
    backend: DeepSeekV4FlashGPUBackend | _SlidingLayerBackend,
    state: DeepSeekV4FlashGPURequestState,
    token_idx: int,
    token_id: int | None = None,
    token_id_tensor: torch.Tensor | None = None,
    kv_rows: torch.Tensor | None = None,
    extra_kv_rows: torch.Tensor | None = None,
    router_top_k: int = DEEPSEEK_V4_FLASH_SHAPE.num_experts_per_tok,
    use_reference_rope: bool = False,
) -> torch.Tensor:
    if not hidden.is_cuda:
        raise ValueError("DeepSeek V4 Flash compressed layer hidden must be CUDA")
    if hidden.ndim not in (1, 2):
        raise ValueError(
            f"hidden must be 1-D or mHC stream-shaped 2-D; got {hidden.ndim}-D"
        )
    ratio = layer_compress_ratio(layer.layer_index)
    if ratio == 0:
        raise ValueError(
            "DeepSeek V4 Flash compressed forward received sliding-only "
            f"layer {layer.layer_index}"
        )
    state.require_capacity(token_idx)

    attention_norm = stager.stage_vector(
        _required_tensor(layer.attention_norm, "attention_norm")
    )
    ffn_norm = stager.stage_vector(_required_tensor(layer.ffn_norm, "ffn_norm"))
    grouped_experts = _required_grouped_experts(layer.grouped_experts)
    compressor = _required_compressor(layer.attention_compressor)
    if ratio == 4 and layer.indexer is None:
        raise ValueError("DeepSeek V4 Flash ratio-4 compressed layer missing indexer")

    uses_hyper_connection = (
        layer.attention_hyper_connection is not None
        or layer.ffn_hyper_connection is not None
    )
    attention_residual_streams: torch.Tensor | None = None
    attention_hc_state: _GPUHyperConnectionState | None = None
    if uses_hyper_connection:
        stream_hc = layer.attention_hyper_connection or layer.ffn_hyper_connection
        if stream_hc is None:
            raise AssertionError("hyper-connection state was not validated")
        attention_residual_streams = _ensure_hyper_connection_streams(
            hidden,
            stager=stager,
            hyper_connection=stream_hc,
        )
        if layer.attention_hyper_connection is None:
            attn_source = attention_residual_streams.mean(dim=0).to(torch.float32)
        else:
            attention_hc_state = _hyper_connection_pre_cuda(
                attention_residual_streams,
                stager=stager,
                hyper_connection=layer.attention_hyper_connection,
            )
            attn_source = attention_hc_state.mixed
    else:
        if hidden.ndim != 1:
            raise ValueError(
                "DeepSeek V4 Flash compressed layer accepts 2-D hidden only "
                "when layer hyper-connection tensors are present"
            )
        attn_source = hidden

    with _stager_profile_section(
        stager,
        "layer_attention_norm",
        layer_idx=layer.layer_index,
        layer_type="compressed",
        ratio=ratio,
    ):
        attn_input = deepseek_v4_flash_rms_norm(attn_source, attention_norm)
    with _stager_profile_section(
        stager,
        "compressed_kv_update",
        layer_idx=layer.layer_index,
        ratio=ratio,
    ):
        candidate_row, emitted_row = _update_compressor_state(
            attn_input,
            state=state,
            layer_idx=layer.layer_index,
            state_name="attention",
            compressor=compressor,
            stager=stager,
            token_idx=token_idx,
            ratio=ratio,
            use_reference_rope=use_reference_rope,
        )
    indexer_row: torch.Tensor | None = None
    if layer.indexer is not None:
        with _stager_profile_section(
            stager,
            "compressed_indexer_update",
            layer_idx=layer.layer_index,
            ratio=ratio,
        ):
            _index_candidate, indexer_row = _update_compressor_state(
                attn_input,
                state=state,
                layer_idx=layer.layer_index,
                state_name="indexer",
                compressor=layer.indexer.compressor,
                stager=stager,
                token_idx=token_idx,
                ratio=4,
                use_reference_rope=use_reference_rope,
            )

    if emitted_row is not None:
        if ratio == 4 and indexer_row is None:
            raise ValueError(
                "DeepSeek V4 Flash ratio-4 compressed layer did not emit "
                "an indexer row with the attention row"
            )
        _write_compressed_runtime_row(
            state,
            layer_idx=layer.layer_index,
            token_idx=token_idx,
            compressed_row=emitted_row,
            indexer_row=indexer_row,
        )

    prior_rows = state.compressed_kv_cache.read_compressed(layer.layer_index)
    if prior_rows.shape[0] == 0:
        compressed_attention_rows: torch.Tensor | None = None
        compressed_extra_rows: torch.Tensor | None = None
        selected_rows = torch.zeros(1, dtype=torch.int64, device=hidden.device)
    elif layer.indexer is not None:
        with _stager_profile_section(
            stager,
            "compressed_indexer_select",
            layer_idx=layer.layer_index,
            ratio=ratio,
        ):
            selected_rows = _select_compressed_rows_with_indexer(
                attn_input,
                layer=layer,
                stager=stager,
                indexer_rows=state.compressed_kv_cache.read_indexer_rows(
                    layer.layer_index
                ),
                state=state,
                token_idx=token_idx,
                use_reference_rope=use_reference_rope,
            )
        compressed_attention_rows = prior_rows
        compressed_extra_rows = state.compressed_kv_cache.read_compressed(
            layer.layer_index,
            row_indices=selected_rows,
        )
    else:
        compressed_attention_rows = prior_rows
        compressed_extra_rows = prior_rows
        selected_rows = torch.arange(
            prior_rows.shape[0],
            dtype=torch.int64,
            device=hidden.device,
        )

    with _stager_profile_section(
        stager,
        "layer_attention",
        layer_idx=layer.layer_index,
        layer_type="compressed",
        ratio=ratio,
    ):
        if _has_real_sliding_attention_tensors(layer):
            if kv_rows is None:
                effective_kv_rows: torch.Tensor | None = None
                effective_extra_kv_rows = compressed_extra_rows
            else:
                effective_kv_rows = kv_rows
                effective_extra_kv_rows = extra_kv_rows
            attn_update = _run_real_sliding_attention(
                attn_input,
                layer=layer,
                stager=stager,
                backend=backend,
                state=state,
                token_idx=token_idx,
                kv_rows=effective_kv_rows,
                extra_kv_rows=effective_extra_kv_rows,
                use_reference_rope=use_reference_rope,
            )
        else:
            attention_rows = compressed_attention_rows
            if attention_rows is None:
                attention_rows = (
                    candidate_row if emitted_row is None else emitted_row
                ).reshape(1, -1)
            query = _compressed_attention_query(attn_input, layer=layer, stager=stager)
            context = backend.compressed_attention(
                query=query,
                compressed_rows=attention_rows,
                selected_rows=selected_rows,
            )
            context = _expand_shared_attention_context_for_output(
                context,
                layer=layer,
                stager=stager,
            )
            attn_update = _project_sliding_attention_output(
                context,
                None
                if layer.attention_output_a is not None
                and layer.attention_output_b is not None
                else _required_tensor(layer.attention_output, "attention_output"),
                attention_output_a=layer.attention_output_a,
                attention_output_b=layer.attention_output_b,
                stager=stager,
            )
    if uses_hyper_connection:
        if attention_residual_streams is None:
            raise AssertionError("attention residual streams were not initialized")
        if attention_hc_state is None:
            hidden_after_attn = attention_residual_streams + attn_update.reshape(1, -1)
        else:
            hidden_after_attn = _hyper_connection_post_cuda(
                attn_update,
                attention_residual_streams,
                attention_hc_state,
            )
    else:
        hidden_after_attn = deepseek_v4_flash_residual_hyper_connection(
            attn_source,
            attn_update,
            stager=stager,
            hyper_connection=None,
        )

    if uses_hyper_connection:
        if hidden_after_attn.ndim != 2:
            raise ValueError("mHC compressed path must carry stream-shaped hidden")
        ffn_residual_streams = hidden_after_attn
        if layer.ffn_hyper_connection is None:
            ffn_source = ffn_residual_streams.mean(dim=0).to(torch.float32)
            ffn_hc_state = None
        else:
            ffn_hc_state = _hyper_connection_pre_cuda(
                ffn_residual_streams,
                stager=stager,
                hyper_connection=layer.ffn_hyper_connection,
            )
            ffn_source = ffn_hc_state.mixed
    else:
        ffn_source = hidden_after_attn
        ffn_hc_state = None
        ffn_residual_streams = None

    with _stager_profile_section(
        stager,
        "layer_ffn_norm",
        layer_idx=layer.layer_index,
        layer_type="compressed",
        ratio=ratio,
    ):
        ffn_input = deepseek_v4_flash_rms_norm(ffn_source, ffn_norm)
    with _stager_profile_section(
        stager,
        "layer_moe",
        layer_idx=layer.layer_index,
        layer_type="compressed",
        ratio=ratio,
    ):
        moe_update = _run_sliding_moe(
            ffn_input,
            layer=layer,
            grouped_experts=grouped_experts,
            stager=stager,
            backend=backend,
            state=state,
            token_id=token_idx if token_id is None else token_id,
            token_id_tensor=token_id_tensor,
            router_top_k=router_top_k,
        )
    if uses_hyper_connection:
        if ffn_residual_streams is None:
            raise AssertionError("FFN residual streams were not initialized")
        if ffn_hc_state is None:
            return ffn_residual_streams + moe_update.reshape(1, -1)
        return _hyper_connection_post_cuda(
            moe_update,
            ffn_residual_streams,
            ffn_hc_state,
        )
    return deepseek_v4_flash_residual_hyper_connection(
        hidden_after_attn,
        moe_update,
        stager=stager,
        hyper_connection=None,
    )


def _project_sliding_attention_output(
    context: torch.Tensor,
    attention_output: DeepSeekV4FlashTensor | None,
    *,
    attention_output_a: DeepSeekV4FlashTensor | None,
    attention_output_b: DeepSeekV4FlashTensor | None,
    stager: DeepSeekV4FlashGPUWeightStager,
) -> torch.Tensor:
    if attention_output_a is not None or attention_output_b is not None:
        if attention_output_a is None or attention_output_b is None:
            raise ValueError(
                "DeepSeek V4 Flash sliding output projection requires both "
                "attention_output_a and attention_output_b"
            )
        return _grouped_output_projection(
            context,
            attention_output_a,
            attention_output_b,
            output_groups=DEEPSEEK_V4_FLASH_SHAPE.output_groups,
            stager=stager,
        )
    if attention_output is None:
        raise ValueError("DeepSeek V4 Flash sliding layer missing attention_output")
    return _project_tensor(context, attention_output, stager)


def _grouped_output_projection(
    context: torch.Tensor,
    output_a: DeepSeekV4FlashTensor,
    output_b: DeepSeekV4FlashTensor,
    *,
    output_groups: int,
    stager: DeepSeekV4FlashGPUWeightStager,
) -> torch.Tensor:
    if context.ndim != 1:
        raise ValueError(f"attention context must be 1-D; got {context.ndim}-D")
    if len(output_a.dims) != 2 or len(output_b.dims) != 2:
        raise ValueError("attention output projection tensors must be 2-D")
    if output_groups <= 0:
        raise ValueError(f"output_groups must be positive; got {output_groups}")
    if context.numel() % output_groups != 0:
        raise ValueError(
            "attention context width must be divisible by output_groups; "
            f"got {context.numel()} and {output_groups}"
        )

    group_input = context.numel() // output_groups
    if output_a.dims[0] != group_input:
        raise ValueError(
            "attention_output_a input width must match grouped context; "
            f"got {output_a.dims[0]} and {group_input}"
        )
    if output_a.dims[1] % output_groups != 0:
        raise ValueError(
            "attention_output_a output width must be divisible by output_groups; "
            f"got {output_a.dims[1]} and {output_groups}"
        )
    rank_per_group = output_a.dims[1] // output_groups
    if output_b.dims[0] != output_groups * rank_per_group:
        raise ValueError(
            "attention_output_b input width must match grouped output rank; "
            f"got {output_b.dims[0]} and {output_groups * rank_per_group}"
        )

    grouped = context.to(torch.float32).reshape(output_groups, group_input)
    if (
        output_a.tensor_type == GGML_TYPE_Q8_0
        and output_a.dims[0] % _Q8_0_BLOCK_SIZE == 0
    ):
        raw_payload = _stage_q8_0_raw_tensor(output_a, stager)
        row_bytes = (output_a.dims[0] // _Q8_0_BLOCK_SIZE) * _Q8_0_BLOCK_BYTES
        low_rank_rows = []
        for group_idx in range(output_groups):
            row_start = group_idx * rank_per_group
            row_end = row_start + rank_per_group
            byte_start = row_start * row_bytes
            byte_end = row_end * row_bytes
            low_rank_rows.append(
                q8_0_raw_linear(
                    raw_payload[byte_start:byte_end],
                    grouped[group_idx],
                    rows=rank_per_group,
                    columns=group_input,
                    block_size=_Q8_0_BLOCK_SIZE,
                )
            )
        low_rank = torch.stack(low_rank_rows, dim=0)
    else:
        output_a_weight = stager.stage_matrix(output_a)
        a_by_group = output_a_weight.to(torch.float32).reshape(
            output_groups,
            rank_per_group,
            group_input,
        )
        low_rank = torch.einsum("gi,gri->gr", grouped, a_by_group)
    return _project_tensor(low_rank.reshape(-1), output_b, stager)


def _ensure_hyper_connection_streams(
    hidden: torch.Tensor,
    *,
    stager: DeepSeekV4FlashGPUWeightStager,
    hyper_connection: DeepSeekV4FlashHyperConnectionTensors,
) -> torch.Tensor:
    base = stager.stage_vector(hyper_connection.base)
    hc_mult = _hyper_connection_stream_count(base)
    if hidden.ndim == 2:
        if hidden.shape[0] != hc_mult:
            raise ValueError(
                "mHC stream count must match hyper-connection tensors; "
                f"got {hidden.shape[0]} and {hc_mult}"
            )
        return hidden.to(torch.float32)
    if hidden.ndim != 1:
        raise ValueError(
            f"hidden must be 1-D or mHC stream-shaped 2-D; got {hidden.ndim}-D"
        )
    return hidden.to(torch.float32).reshape(1, -1).expand(hc_mult, -1).clone()


def _hyper_connection_stream_count(base: torch.Tensor) -> int:
    mix_count = base.numel()
    hc_mult = 1
    while 2 * hc_mult + hc_mult * hc_mult < mix_count:
        hc_mult += 1
    if 2 * hc_mult + hc_mult * hc_mult != mix_count:
        raise ValueError(
            f"hyper-connection base size does not match 2*h + h*h; got {mix_count}"
        )
    return hc_mult


def _hyper_connection_pre_cuda(
    streams: torch.Tensor,
    *,
    stager: DeepSeekV4FlashGPUWeightStager,
    hyper_connection: DeepSeekV4FlashHyperConnectionTensors,
) -> _GPUHyperConnectionState:
    if not streams.is_cuda:
        raise ValueError("DeepSeek V4 Flash mHC streams must be CUDA tensors")
    if streams.ndim != 2:
        raise ValueError(f"mHC streams must be 2-D; got {streams.ndim}-D")

    fn_weight = stager.stage_matrix(hyper_connection.fn)
    base = stager.stage_vector(hyper_connection.base)
    scale = stager.stage_vector(hyper_connection.scale)
    hc_mult, hidden_size = streams.shape
    mix_count = 2 * hc_mult + hc_mult * hc_mult
    flat_size = hc_mult * hidden_size
    if fn_weight.shape == (flat_size, mix_count):
        projection_weight = fn_weight.to(torch.float32).T
    elif fn_weight.shape == (mix_count, flat_size):
        projection_weight = fn_weight.to(torch.float32)
    else:
        raise ValueError(
            "hyper-connection fn tensor shape does not match streams; "
            f"got {tuple(fn_weight.shape)}, expected ({flat_size}, {mix_count})"
        )
    if base.shape != (mix_count,):
        raise ValueError(
            f"hyper-connection base shape must be ({mix_count},); "
            f"got {tuple(base.shape)}"
        )
    if scale.shape != (3,):
        raise ValueError(
            f"hyper-connection scale shape must be (3,); got {tuple(scale.shape)}"
        )

    flat = streams.reshape(flat_size).to(torch.float32)
    mixes = projection_weight.matmul(flat) * torch.rsqrt(flat.pow(2).mean() + 1e-6)
    repeat_counts = torch.tensor(
        [hc_mult, hc_mult, hc_mult * hc_mult],
        dtype=torch.long,
        device=scale.device,
    )
    mixes = mixes * scale.to(torch.float32).repeat_interleave(repeat_counts)
    mixes = mixes + base.to(torch.float32)
    pre = torch.sigmoid(mixes[:hc_mult]) + 1e-6
    post = 2.0 * torch.sigmoid(mixes[hc_mult : 2 * hc_mult])
    combine_scores = mixes[2 * hc_mult :].reshape(hc_mult, hc_mult)
    combine = torch.softmax(combine_scores, dim=-1) + 1e-6
    combine = combine / (combine.sum(dim=0, keepdim=True) + 1e-6)
    for _ in range(1, 20):
        combine = combine / (combine.sum(dim=-1, keepdim=True) + 1e-6)
        combine = combine / (combine.sum(dim=0, keepdim=True) + 1e-6)
    mixed = (pre.reshape(hc_mult, 1) * streams.to(torch.float32)).sum(dim=0)
    return _GPUHyperConnectionState(mixed=mixed, post=post, combine=combine)


def _hyper_connection_post_cuda(
    output: torch.Tensor,
    residual_streams: torch.Tensor,
    state: _GPUHyperConnectionState,
) -> torch.Tensor:
    if not output.is_cuda or not residual_streams.is_cuda:
        raise ValueError("DeepSeek V4 Flash mHC post inputs must be CUDA tensors")
    if output.ndim != 1:
        raise ValueError(f"mHC output must be 1-D; got {output.ndim}-D")
    if residual_streams.ndim != 2:
        raise ValueError(
            f"mHC residual streams must be 2-D; got {residual_streams.ndim}-D"
        )
    hc_mult, hidden_size = residual_streams.shape
    if output.shape != (hidden_size,):
        raise ValueError(
            f"mHC output shape must be ({hidden_size},); got {tuple(output.shape)}"
        )
    if state.post.shape != (hc_mult,):
        raise ValueError(
            f"mHC post shape must be ({hc_mult},); got {tuple(state.post.shape)}"
        )
    if state.combine.shape != (hc_mult, hc_mult):
        raise ValueError(
            "mHC combine shape must match residual streams; "
            f"got {tuple(state.combine.shape)}"
        )
    residual_mix = state.combine.to(torch.float32).T.matmul(
        residual_streams.to(torch.float32)
    )
    return state.post.reshape(hc_mult, 1) * output.to(torch.float32) + residual_mix


def _expand_shared_attention_context_for_output(
    context: torch.Tensor,
    *,
    layer: DeepSeekV4FlashLayerSemanticBindings,
    stager: DeepSeekV4FlashGPUWeightStager,
) -> torch.Tensor:
    if context.ndim != 1:
        raise ValueError(f"attention context must be 1-D; got {context.ndim}-D")
    if layer.attention_output_a is None or layer.attention_output_b is None:
        return context
    expected_context = (
        layer.attention_output_a.dims[0] * DEEPSEEK_V4_FLASH_SHAPE.output_groups
    )
    if context.numel() == expected_context:
        return context
    attn_sinks = (
        stager.stage_vector(layer.attention_sinks)
        if layer.attention_sinks is not None
        else None
    )
    if (
        attn_sinks is not None
        and context.numel() * attn_sinks.numel() == expected_context
    ):
        return context.reshape(1, -1).expand(attn_sinks.numel(), -1).reshape(-1)
    return context


def _compressed_attention_query(
    hidden: torch.Tensor,
    *,
    layer: DeepSeekV4FlashLayerSemanticBindings,
    stager: DeepSeekV4FlashGPUWeightStager,
) -> torch.Tensor:
    if layer.attention_key_value is not None:
        query = _project_tensor(hidden, layer.attention_key_value, stager)
        if layer.attention_key_value_a_norm is not None:
            query = deepseek_v4_flash_rms_norm(
                query,
                stager.stage_vector(layer.attention_key_value_a_norm),
            )
        return query

    query = _project_tensor(
        hidden,
        _required_tensor(layer.attention_query, "attention_query"),
        stager,
    )
    if layer.attention_query_a_norm is not None:
        query = deepseek_v4_flash_rms_norm(
            query,
            stager.stage_vector(layer.attention_query_a_norm),
        )
    if layer.attention_query_b is not None:
        query = _project_tensor(query, layer.attention_query_b, stager)
    return query


def _update_compressor_state(
    hidden: torch.Tensor,
    *,
    state: DeepSeekV4FlashGPURequestState,
    layer_idx: int,
    state_name: str,
    compressor: DeepSeekV4FlashCompressorTensors,
    stager: DeepSeekV4FlashGPUWeightStager,
    token_idx: int,
    ratio: int,
    use_reference_rope: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if token_idx < 0:
        raise ValueError("token_idx must be non-negative")
    if ratio <= 0:
        raise ValueError(f"compressor ratio must be positive; got {ratio}")
    kv_cur = _project_tensor(hidden, compressor.kv, stager)
    score_cur = _project_tensor(hidden, compressor.gate, stager)
    if kv_cur.shape != score_cur.shape:
        raise ValueError(
            "compressor KV and gate projections must share one shape; "
            f"got {tuple(kv_cur.shape)} and {tuple(score_cur.shape)}"
        )

    norm = stager.stage_vector(compressor.norm)
    row_width = norm.numel()
    projection_width = kv_cur.numel()
    if ratio == 4 and projection_width == 2 * row_width:
        state_rows = 2 * ratio
        use_ratio4_carry = True
    elif projection_width == row_width:
        state_rows = ratio
        use_ratio4_carry = False
    else:
        raise ValueError(
            "unsupported DeepSeek V4 Flash compressor projection layout: "
            "projection width must be row_width or 2 * row_width for ratio-4; "
            f"got projection_width={projection_width}, row_width={row_width}, "
            f"ratio={ratio}"
        )
    pos_mod = token_idx % ratio
    ape = stager.stage_matrix(compressor.ape)
    if ape.ndim != 2:
        raise ValueError(f"compressor ape must be 2-D; got {ape.ndim}-D")
    if ape.shape == (projection_width, ratio):
        ape_pos = ape[:, pos_mod]
    elif ape.shape == (ratio, projection_width):
        ape_pos = ape[pos_mod, :]
    else:
        raise ValueError(
            "unsupported DeepSeek V4 Flash compressor APE layout: "
            f"expected ({projection_width}, {ratio}) or "
            f"({ratio}, {projection_width}), got {tuple(ape.shape)}"
        )

    runtime_state = _get_compressor_runtime_state(
        state,
        layer_idx=layer_idx,
        state_name=state_name,
        state_rows=state_rows,
        projection_width=projection_width,
        device=hidden.device,
    )
    target_row = ratio + pos_mod if use_ratio4_carry else pos_mod
    runtime_state.kv_rows[target_row].copy_(kv_cur.to(torch.float32))
    runtime_state.score_rows[target_row].copy_(
        score_cur.to(torch.float32) + ape_pos.to(torch.float32)
    )
    runtime_state.count = min(runtime_state.count + 1, ratio)

    candidate_source = kv_cur[row_width : 2 * row_width] if use_ratio4_carry else kv_cur
    candidate = deepseek_v4_flash_rms_norm(
        candidate_source.to(torch.float32),
        norm,
    ).to(stager.dtype)
    if (token_idx + 1) % ratio != 0 or runtime_state.count < ratio:
        return candidate, None

    if use_ratio4_carry:
        primary_scores = runtime_state.score_rows[:ratio, :row_width]
        carry_scores = runtime_state.score_rows[ratio : 2 * ratio, row_width:]
        primary_kv = runtime_state.kv_rows[:ratio, :row_width]
        carry_kv = runtime_state.kv_rows[ratio : 2 * ratio, row_width:]
        score_rows = torch.cat([primary_scores, carry_scores], dim=0)
        kv_rows = torch.cat([primary_kv, carry_kv], dim=0)
    else:
        score_rows = runtime_state.score_rows
        kv_rows = runtime_state.kv_rows

    weights = torch.softmax(score_rows.to(torch.float32), dim=0)
    pooled = (weights * kv_rows.to(torch.float32)).sum(dim=0)
    emitted = deepseek_v4_flash_rms_norm(pooled, norm).to(stager.dtype)
    if emitted.numel() >= DEEPSEEK_V4_FLASH_SHAPE.rotary_dim:
        rope_token_idx = token_idx + 1 - ratio
        if use_reference_rope:
            emitted = apply_deepseek_layer_rope_to_tail_reference(
                emitted,
                token_idx=rope_token_idx,
                layer_idx=layer_idx,
                rotary_dim=DEEPSEEK_V4_FLASH_SHAPE.rotary_dim,
            ).to(stager.dtype)
        else:
            cos, sin = state.rope_tables_for(layer_idx, rope_token_idx)
            emitted = apply_precomputed_rope_to_tail(
                emitted,
                cos,
                sin,
            ).to(stager.dtype)
    if emitted.numel() == DEEPSEEK_V4_FLASH_SHAPE.head_dim:
        emitted = deepseek_fp8_kv_qat_reference(
            emitted,
            head_dim=DEEPSEEK_V4_FLASH_SHAPE.head_dim,
            rotary_dim=DEEPSEEK_V4_FLASH_SHAPE.rotary_dim,
        ).to(stager.dtype)
    elif emitted.numel() == DEEPSEEK_V4_FLASH_SHAPE.indexer_head_dim:
        emitted = deepseek_indexer_qat_reference(emitted).to(stager.dtype)
    if use_ratio4_carry:
        carry_kv_full = runtime_state.kv_rows[ratio : 2 * ratio].clone()
        carry_score_full = runtime_state.score_rows[ratio : 2 * ratio].clone()
        runtime_state.kv_rows[:ratio].copy_(carry_kv_full)
        runtime_state.score_rows[:ratio].copy_(carry_score_full)
        runtime_state.kv_rows[ratio : 2 * ratio].copy_(carry_kv_full)
        runtime_state.score_rows[ratio : 2 * ratio].copy_(carry_score_full)
    runtime_state.count = 0
    return candidate, emitted


def _get_compressor_runtime_state(
    state: DeepSeekV4FlashGPURequestState,
    *,
    layer_idx: int,
    state_name: str,
    state_rows: int,
    projection_width: int,
    device: torch.device,
) -> _CompressorRuntimeState:
    attr = "_deepseek_v4_flash_compressor_states"
    states = getattr(state, attr, None)
    if states is None:
        states = {}
        setattr(state, attr, states)
    states = cast(dict[tuple[int, str], _CompressorRuntimeState], states)
    key = (layer_idx, state_name)
    runtime_state = states.get(key)
    if runtime_state is None:
        runtime_state = _CompressorRuntimeState(
            kv_rows=torch.zeros(
                (state_rows, projection_width),
                dtype=torch.float32,
                device=device,
            ),
            score_rows=torch.empty(
                (state_rows, projection_width),
                dtype=torch.float32,
                device=device,
            ).fill_(-1000.0),
        )
        states[key] = runtime_state
        return runtime_state
    if runtime_state.kv_rows.shape != (state_rows, projection_width):
        raise ValueError(
            "compressor runtime state shape changed for layer "
            f"{layer_idx} {state_name}: got {tuple(runtime_state.kv_rows.shape)}, "
            f"expected ({state_rows}, {projection_width})"
        )
    if runtime_state.kv_rows.device != device:
        raise ValueError(
            "compressor runtime state device changed for layer "
            f"{layer_idx} {state_name}: got {runtime_state.kv_rows.device}, "
            f"expected {device}"
        )
    return runtime_state


def _select_compressed_rows_with_indexer(
    hidden: torch.Tensor,
    *,
    layer: DeepSeekV4FlashLayerSemanticBindings,
    stager: DeepSeekV4FlashGPUWeightStager,
    indexer_rows: torch.Tensor,
    state: DeepSeekV4FlashGPURequestState | None = None,
    token_idx: int,
    use_reference_rope: bool = False,
) -> torch.Tensor:
    if layer.indexer is None:
        raise ValueError("DeepSeek V4 Flash compressed row selection requires indexer")
    if indexer_rows.ndim != 2:
        raise ValueError(f"indexer_rows must be 2-D; got {indexer_rows.ndim}-D")
    if indexer_rows.shape[0] == 0:
        raise ValueError("indexer_rows must contain at least one row")
    if not indexer_rows.is_cuda:
        raise ValueError("indexer_rows must be CUDA")
    top_k = min(DEEPSEEK_V4_FLASH_SHAPE.indexer_top_k, indexer_rows.shape[0])
    if top_k <= 0:
        raise ValueError("indexer selected no compressed rows")
    if indexer_rows.shape[0] <= top_k:
        return torch.arange(
            indexer_rows.shape[0],
            dtype=torch.int64,
            device=indexer_rows.device,
        )

    index_weights = _project_tensor(hidden, layer.indexer.projection, stager)
    if index_weights.ndim != 1:
        raise ValueError(f"indexer weights must be 1-D; got {index_weights.ndim}-D")

    if layer.attention_query_a is None:
        query_source = hidden
    else:
        query_a_norm = _required_tensor(
            layer.attention_query_a_norm,
            "attention_query_a_norm",
        )
        query_source = _project_tensor(hidden, layer.attention_query_a, stager)
        query_source = deepseek_v4_flash_rms_norm(
            query_source,
            stager.stage_vector(query_a_norm),
        )
    query_flat = _project_tensor(query_source, layer.indexer.query_b, stager)
    row_width = indexer_rows.shape[1]
    if query_flat.numel() % row_width != 0:
        raise ValueError(
            "indexer query width must be divisible by indexer row width; "
            f"got {query_flat.numel()} and {row_width}"
        )
    heads = query_flat.numel() // row_width
    if index_weights.numel() != heads:
        raise ValueError(
            "indexer projection width must match query heads; "
            f"got {index_weights.numel()} and {heads}"
        )
    index_query = query_flat.reshape(heads, row_width)
    if use_reference_rope or state is None:
        index_query = apply_deepseek_layer_rope_to_tail_reference(
            index_query,
            token_idx=token_idx,
            layer_idx=layer.layer_index,
            rotary_dim=DEEPSEEK_V4_FLASH_SHAPE.rotary_dim,
        )
    else:
        cos, sin = state.rope_tables_for(layer.layer_index, token_idx)
        index_query = apply_precomputed_rope_to_tail(index_query, cos, sin)
    if row_width == DEEPSEEK_V4_FLASH_SHAPE.indexer_head_dim:
        index_query = torch.stack(
            [deepseek_indexer_qat_reference(row) for row in index_query],
            dim=0,
        ).to(index_query.device, dtype=index_query.dtype)
    per_head_scores = index_query.to(torch.float32).matmul(
        indexer_rows.to(torch.float32).T
    )
    per_head_scores = torch.clamp_min(per_head_scores, 0.0)
    scale = 1.0 / float(heads * row_width) ** 0.5
    scores = (index_weights.to(torch.float32).reshape(heads, 1) * per_head_scores).sum(
        dim=0
    ) * scale
    return torch.topk(scores, k=top_k, sorted=True).indices.to(torch.int64)


def _write_compressed_runtime_row(
    state: DeepSeekV4FlashGPURequestState,
    *,
    layer_idx: int,
    token_idx: int,
    compressed_row: torch.Tensor,
    indexer_row: torch.Tensor | None,
) -> None:
    cache = state.compressed_kv_cache
    slot = int(cache._compressed_counts[layer_idx].item())
    if slot >= cache.compressed_rows.shape[1]:
        raise ValueError("compressed cache capacity exceeded")
    if compressed_row.shape != (cache.hidden_size,):
        raise ValueError(
            f"compressed row shape must be ({cache.hidden_size},); "
            f"got {tuple(compressed_row.shape)}"
        )
    page_table = torch.zeros(
        cache.compressed_rows.shape[1],
        dtype=torch.int32,
        device=compressed_row.device,
    )
    deepseek_v4_cache_update(
        DeepSeekV4CacheUpdateInputs(
            page_table=page_table,
            kv_row=compressed_row.to(cache.compressed_rows.dtype),
            cache_storage=cache.compressed_rows[layer_idx].unsqueeze(0),
            logical_row=slot,
        )
    )
    cache.compressed_token_indices[layer_idx, slot] = token_idx
    if indexer_row is not None:
        if indexer_row.shape != (cache.indexer_rows.shape[-1],):
            raise ValueError(
                "indexer row shape must match runtime indexer width; "
                f"got {tuple(indexer_row.shape)} and {cache.indexer_rows.shape[-1]}"
            )
        deepseek_v4_cache_update(
            DeepSeekV4CacheUpdateInputs(
                page_table=page_table,
                kv_row=indexer_row.to(cache.indexer_rows.dtype),
                cache_storage=cache.indexer_rows[layer_idx].unsqueeze(0),
                logical_row=slot,
            )
        )
    cache._compressed_counts[layer_idx] += 1


def _run_sliding_moe(
    hidden: torch.Tensor,
    *,
    layer: DeepSeekV4FlashLayerSemanticBindings,
    grouped_experts: DeepSeekV4FlashGroupedExpertTensors,
    stager: DeepSeekV4FlashGPUWeightStager,
    backend: DeepSeekV4FlashGPUBackend | _SlidingLayerBackend,
    state: DeepSeekV4FlashGPURequestState | None = None,
    token_id: int,
    token_id_tensor: torch.Tensor | None = None,
    router_top_k: int,
) -> torch.Tensor:
    with _stager_profile_section(
        stager,
        "moe_hash_router",
        layer_idx=layer.layer_index,
    ):
        routed = _run_hash_routed_experts(
            hidden,
            layer=layer,
            grouped_experts=grouped_experts,
            stager=stager,
            backend=backend,
            state=state,
            token_id=token_id,
            token_id_tensor=token_id_tensor,
        )
    if routed is None:
        router = _required_tensor(layer.router, "router")
        correction_bias = (
            stager.stage_vector(layer.expert_probs_bias)
            if layer.expert_probs_bias is not None
            else None
        )
        with _stager_profile_section(
            stager,
            "moe_router_topk",
            layer_idx=layer.layer_index,
        ):
            logits = _project_tensor(hidden, router, stager)
            expert_ids, expert_weights = _router_topk_from_logits(
                logits,
                top_k=router_top_k,
                correction_bias=correction_bias,
            )
        with _stager_profile_section(
            stager,
            "moe_routed_experts",
            layer_idx=layer.layer_index,
        ):
            routed = _run_staged_routed_experts(
                hidden,
                expert_ids,
                expert_weights,
                grouped_experts=grouped_experts,
                stager=stager,
                backend=backend,
                state=state,
                layer_idx=layer.layer_index,
            )

    if layer.shared_experts is None:
        return routed
    with _stager_profile_section(
        stager,
        "moe_shared_expert",
        layer_idx=layer.layer_index,
    ):
        shared = _run_staged_shared_expert(
            hidden,
            layer.shared_experts,
            stager=stager,
            backend=backend,
        )
    if shared.shape != routed.shape:
        raise ValueError(
            "shared and routed MoE outputs must share one shape; "
            f"got {tuple(shared.shape)} and {tuple(routed.shape)}"
        )
    return shared.to(torch.float32) + routed.to(torch.float32)


def _run_hash_routed_experts(
    hidden: torch.Tensor,
    *,
    layer: DeepSeekV4FlashLayerSemanticBindings,
    grouped_experts: DeepSeekV4FlashGroupedExpertTensors,
    stager: DeepSeekV4FlashGPUWeightStager,
    backend: DeepSeekV4FlashGPUBackend | _SlidingLayerBackend,
    state: DeepSeekV4FlashGPURequestState | None = None,
    token_id: int,
    token_id_tensor: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if layer.expert_token_to_expert_ids is None:
        return None
    table_device = hidden.device if token_id_tensor is not None else torch.device("cpu")
    table = _stage_expert_token_table(
        stager,
        layer.expert_token_to_expert_ids,
        device=table_device,
    )
    if table.ndim != 2:
        raise ValueError(f"expert token table must be 2-D; got {table.ndim}-D")
    if token_id_tensor is None:
        if token_id < 0 or token_id >= table.shape[1]:
            raise ValueError(
                f"token_id out of range: {token_id}; expected [0, {table.shape[1]})"
            )
        expert_ids = table[:, token_id].to(torch.int64)
    else:
        token_id_tensor = token_id_tensor.to(device=hidden.device, dtype=torch.long)
        token_id_tensor = token_id_tensor.reshape(())
        torch._assert_async(
            (token_id_tensor >= 0) & (token_id_tensor < table.shape[1]),
            f"token_id out of range; expected [0, {table.shape[1]})",
        )
        expert_ids = table.index_select(1, token_id_tensor.reshape(1)).reshape(-1)
        expert_ids = expert_ids.to(torch.int64)
    if expert_ids.is_cuda:
        torch._assert_async(
            torch.all(expert_ids >= 0),
            "hash-routed expert ids must be non-negative",
        )
    elif torch.any(expert_ids < 0):
        raise ValueError("hash-routed expert ids must be non-negative")
    if layer.router is None:
        weights = torch.full(
            expert_ids.shape,
            1.5 / float(expert_ids.numel()),
            dtype=torch.float32,
            device=hidden.device,
        )
    else:
        if hidden.is_cuda:
            logits = _project_tensor(hidden, layer.router, stager)
        else:
            router_weight = stager.stage_matrix(layer.router)
            hidden_f32 = hidden.to(torch.float32)
            router_f32 = router_weight.to(torch.float32)
            if router_f32.shape[1] == hidden.numel():
                logits = router_f32.matmul(hidden_f32)
            elif router_f32.shape[0] == hidden.numel():
                logits = hidden_f32.matmul(router_f32)
            else:
                raise ValueError(
                    "hash router weight must have one dimension matching hidden; "
                    f"got {tuple(router_f32.shape)} and {hidden.numel()}"
                )
        scores = F.softplus(logits).sqrt()
        weights = scores.gather(0, expert_ids.to(device=scores.device))
        weights = weights / torch.clamp(weights.sum(), min=6.103515625e-5)
        weights = weights * 1.5
    return _run_staged_routed_experts(
        hidden,
        expert_ids,
        weights,
        grouped_experts=grouped_experts,
        stager=stager,
        backend=backend,
        state=state,
        layer_idx=layer.layer_index,
    )


def _run_staged_shared_expert(
    hidden: torch.Tensor,
    tensors: DeepSeekV4FlashGroupedExpertTensors,
    *,
    stager: DeepSeekV4FlashGPUWeightStager,
    backend: DeepSeekV4FlashGPUBackend | _SlidingLayerBackend,
) -> torch.Tensor:
    if (
        _can_project_q8_0(hidden, tensors.gate)
        and _can_project_q8_0(hidden, tensors.up)
        and tensors.down.tensor_type == GGML_TYPE_Q8_0
        and len(tensors.down.dims) == 2
    ):
        columns, rows = tensors.gate.dims
        activated = q8_0_raw_gate_up_activation(
            _stage_q8_0_raw_tensor(tensors.gate, stager),
            _stage_q8_0_raw_tensor(tensors.up, stager),
            hidden,
            rows=rows,
            columns=columns,
            block_size=_Q8_0_BLOCK_SIZE,
        )
        if _can_project_q8_0(activated, tensors.down):
            return _project_tensor(activated, tensors.down, stager).to(torch.float32)

    return backend.routed_expert_gemm(
        hidden=hidden,
        gate_weight=stager.stage_matrix(tensors.gate),
        up_weight=stager.stage_matrix(tensors.up),
        down_weight=stager.stage_matrix(tensors.down),
    ).to(torch.float32)


def _run_staged_routed_experts(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    *,
    grouped_experts: DeepSeekV4FlashGroupedExpertTensors,
    stager: DeepSeekV4FlashGPUWeightStager,
    backend: DeepSeekV4FlashGPUBackend | _SlidingLayerBackend,
    state: DeepSeekV4FlashGPURequestState | None = None,
    layer_idx: int | None = None,
) -> torch.Tensor:
    output: torch.Tensor | None = None
    try:
        selected_payloads = _stage_selected_expert_payloads(
            expert_ids,
            grouped_experts=grouped_experts,
            stager=stager,
            layer_idx=layer_idx,
            max_experts=expert_weights.numel(),
        )
    except (AttributeError, NotImplementedError):
        return _run_dense_staged_routed_experts(
            hidden,
            expert_ids,
            expert_weights,
            grouped_experts=grouped_experts,
            stager=stager,
            backend=backend,
            layer_idx=layer_idx,
        )
    selected_gemm = getattr(backend, "fused_quantized_selected_experts_gemm", None)
    if callable(selected_gemm) and state is not None:
        try:
            workspace = state.moe_workspace(
                num_experts=len(selected_payloads),
                intermediate_size=grouped_experts.gate.dims[1],
                device=hidden.device,
            )
            return selected_gemm(
                hidden=hidden,
                expert_weights=expert_weights.reshape(-1),
                payloads=selected_payloads,
                workspace=workspace,
            ).to(torch.float32)
        except (RuntimeError, NotImplementedError, ValueError):
            pass
    selected_gemm = getattr(backend, "quantized_selected_experts_gemm", None)
    if callable(selected_gemm):
        with _stager_profile_section(
            stager,
            "router_selected_experts_kernel",
            layer_idx=layer_idx,
            expert_count=len(selected_payloads),
        ):
            return selected_gemm(
                hidden=hidden,
                expert_weights=expert_weights.reshape(-1),
                payloads=selected_payloads,
            ).to(torch.float32)
    for payload_index, (
        expert_id,
        gate_payload,
        up_payload,
        down_payload,
    ) in enumerate(selected_payloads):
        expert_weight = expert_weights.reshape(-1)[payload_index]
        try:
            with _stager_profile_section(
                stager,
                "router_expert_kernel",
                layer_idx=layer_idx,
                expert_id=expert_id,
            ):
                expert_output = backend.quantized_expert_gemm(
                    hidden=hidden,
                    gate_payload=gate_payload,
                    up_payload=up_payload,
                    down_payload=down_payload,
                ).to(torch.float32)
        except (AttributeError, NotImplementedError):
            with _stager_profile_section(
                stager,
                "router_expert_stage",
                layer_idx=layer_idx,
                expert_id=expert_id,
            ):
                staged = stager.stage_grouped_expert(
                    grouped_experts,
                    expert_id,
                    layer_idx=layer_idx,
                )
            with _stager_profile_section(
                stager,
                "router_expert_kernel",
                layer_idx=layer_idx,
                expert_id=expert_id,
            ):
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
        output = output + expert_weight.to(torch.float32) * expert_output.to(
            torch.float32
        )
    if output is None:
        raise ValueError("DeepSeek V4 Flash routed MoE selected no experts")
    return output


def _run_dense_staged_routed_experts(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    *,
    grouped_experts: DeepSeekV4FlashGroupedExpertTensors,
    stager: DeepSeekV4FlashGPUWeightStager,
    backend: DeepSeekV4FlashGPUBackend | _SlidingLayerBackend,
    layer_idx: int | None,
) -> torch.Tensor:
    output: torch.Tensor | None = None
    selected_expert_ids = _materialize_selected_expert_ids(
        expert_ids,
        max_experts=expert_weights.numel(),
        stager=stager,
    )
    for expert_id, expert_weight in zip(
        selected_expert_ids,
        expert_weights.reshape(-1),
        strict=True,
    ):
        with _stager_profile_section(
            stager,
            "router_expert_stage",
            layer_idx=layer_idx,
            expert_id=expert_id,
        ):
            staged = stager.stage_grouped_expert(
                grouped_experts,
                expert_id,
                layer_idx=layer_idx,
            )
        with _stager_profile_section(
            stager,
            "router_expert_kernel",
            layer_idx=layer_idx,
            expert_id=expert_id,
        ):
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
        output = output + expert_weight.to(torch.float32) * expert_output.to(
            torch.float32
        )
    if output is None:
        raise ValueError("DeepSeek V4 Flash routed MoE selected no experts")
    return output


def _stage_selected_expert_payloads(
    expert_ids: torch.Tensor,
    *,
    grouped_experts: DeepSeekV4FlashGroupedExpertTensors,
    stager: DeepSeekV4FlashGPUWeightStager,
    layer_idx: int | None,
    max_experts: int,
) -> list[DeepSeekV4FlashSelectedExpertPayloads]:
    if max_experts <= 0:
        raise ValueError("routed expert materialization requires at least one expert")
    if expert_ids.detach().reshape(-1).numel() > max_experts:
        raise ValueError(
            "routed expert id tensor exceeds bounded top-k materialization; "
            f"got {expert_ids.numel()} ids for max_experts={max_experts}"
        )
    stage_payloads_for_ids = getattr(
        stager,
        "stage_grouped_expert_payloads_for_ids",
        None,
    )
    if callable(stage_payloads_for_ids):
        with _stager_profile_section(
            stager,
            "router_expert_stage",
            layer_idx=layer_idx,
            expert_id=-1,
        ):
            return stage_payloads_for_ids(
                grouped_experts,
                expert_ids,
                layer_idx=layer_idx,
            )
    selected_expert_ids = _materialize_selected_expert_ids(
        expert_ids,
        max_experts=max_experts,
        stager=stager,
    )
    payloads: list[DeepSeekV4FlashSelectedExpertPayloads] = []
    for expert_id in selected_expert_ids:
        with _stager_profile_section(
            stager,
            "router_expert_stage",
            layer_idx=layer_idx,
            expert_id=expert_id,
        ):
            payloads.append(
                (
                    expert_id,
                    stager.stage_grouped_expert_payload(
                        grouped_experts.gate,
                        expert_id,
                        layer_idx=layer_idx,
                    ),
                    stager.stage_grouped_expert_payload(
                        grouped_experts.up,
                        expert_id,
                        layer_idx=layer_idx,
                    ),
                    stager.stage_grouped_expert_payload(
                        grouped_experts.down,
                        expert_id,
                        layer_idx=layer_idx,
                    ),
                )
            )
    return payloads


def _stage_expert_token_table(
    stager: DeepSeekV4FlashGPUWeightStager,
    tensor: DeepSeekV4FlashTensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    cache = getattr(stager, "_dynamic_cache", None)
    cache_key_fn = getattr(stager, "_dynamic_cache_key", None)
    if not isinstance(cache, dict) or not callable(cache_key_fn):
        return stager.store.tensor_to_torch(tensor, dtype=torch.int32).to(
            device=device,
            dtype=torch.int32,
            non_blocking=True,
        )

    cache_key = cache_key_fn(
        tensor,
        dtype=torch.int32,
        extra=("expert_token_table",),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        record_lru_hit = getattr(stager, "_record_lru_hit", None)
        if callable(record_lru_hit):
            record_lru_hit(cache_key)
        stager.record_cache_hit("dynamic", tensor_name=tensor.name)
        return cached

    decoded = stager.store.tensor_to_torch(tensor, dtype=torch.int32)
    nbytes = decoded.numel() * decoded.element_size()
    prepare_insert = getattr(stager, "_prepare_cache_insert", None)
    if callable(prepare_insert) and not prepare_insert(nbytes):
        record_streamed_bytes = getattr(stager, "_record_streamed_bytes", None)
        if callable(record_streamed_bytes):
            record_streamed_bytes(nbytes)
        return decoded.to(device=device, dtype=torch.int32, non_blocking=True)

    stager.record_cache_miss("dynamic", nbytes, tensor_name=tensor.name)
    staged = decoded.to(device=device, dtype=torch.int32, non_blocking=True)
    register_cached_entry = getattr(stager, "_register_cached_entry", None)
    if callable(register_cached_entry):
        register_cached_entry(cache_key, staged, nbytes)
    else:
        cache[cache_key] = staged
    return staged


def _materialize_selected_expert_ids(
    expert_ids: torch.Tensor,
    *,
    max_experts: int,
    stager: DeepSeekV4FlashGPUWeightStager,
) -> list[int]:
    if max_experts <= 0:
        raise ValueError("routed expert materialization requires at least one expert")
    flattened = expert_ids.detach().reshape(-1)
    if flattened.numel() > max_experts:
        raise ValueError(
            "routed expert id tensor exceeds bounded top-k materialization; "
            f"got {flattened.numel()} ids for max_experts={max_experts}"
        )
    if flattened.is_cuda:
        _increment_stager_counter(stager, "routed_expert_id_materializations")
        flattened = flattened.to(device="cpu", dtype=torch.int64)
    else:
        flattened = flattened.to(dtype=torch.int64)
    return [int(expert_id) for expert_id in flattened.tolist()]


def _increment_stager_counter(
    stager: DeepSeekV4FlashGPUWeightStager,
    counter_name: str,
) -> None:
    stats = getattr(stager, "_cache_stats", None)
    if isinstance(stats, dict):
        stats[counter_name] = int(stats.get(counter_name, 0)) + 1


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


def _required_compressor(
    tensors: DeepSeekV4FlashCompressorTensors | None,
) -> DeepSeekV4FlashCompressorTensors:
    if tensors is None:
        raise ValueError("DeepSeek V4 Flash compressed layer missing compressor")
    return tensors


__all__ = [
    "deepseek_v4_flash_compressed_layer_forward",
    "deepseek_v4_flash_layer_forward",
    "deepseek_v4_flash_q8_0_tensor_projection",
    "deepseek_v4_flash_residual_hyper_connection",
    "deepseek_v4_flash_rms_norm",
    "deepseek_v4_flash_router_topk",
    "deepseek_v4_flash_sliding_layer_forward",
    "deepseek_v4_flash_staged_matrix_projection",
]
