# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

from vllm.kernels.triton.deepseek_v4_flash.attention import (
    DeepSeekV4AttentionKernelInputs,
    deepseek_v4_attention,
)
from vllm.kernels.triton.deepseek_v4_flash.compressed_attention import (
    DeepSeekV4CompressedAttentionTensorInputs,
    deepseek_v4_compressed_attention,
)
from vllm.kernels.triton.deepseek_v4_flash.moe import (
    DeepSeekV4MoEKernelInputs,
    deepseek_v4_moe,
)
from vllm.kernels.triton.deepseek_v4_flash.output import (
    DeepSeekV4OutputKernelInputs,
    deepseek_v4_output_argmax,
    deepseek_v4_output_projection,
)
from vllm.kernels.triton.deepseek_v4_flash.q2_iq2_moe import (
    deepseek_v4_iq2_xxs_gate_up,
    deepseek_v4_iq2_xxs_matvec,
    deepseek_v4_q2_k_matvec,
)
from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_Q2_K,
)


class DeepSeekV4FlashQuantizedExpertPayloadLike(Protocol):
    ggml_type: int
    rows: int
    columns: int
    payload: torch.Tensor


@dataclass(frozen=True)
class DeepSeekV4FlashGPUCapabilities:
    q8_linear: bool = True
    attention: bool = False
    compressed_attention: bool = False
    cache_update: bool = False
    moe: bool = False
    output: bool = False

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, enabled in (
                ("q8_linear", self.q8_linear),
                ("attention", self.attention),
                ("compressed_attention", self.compressed_attention),
                ("cache_update", self.cache_update),
                ("moe", self.moe),
                ("output", self.output),
            )
            if not enabled
        )

    @property
    def is_ready(self) -> bool:
        return not self.missing


class DeepSeekV4FlashGPUBackend:
    def __init__(
        self,
        *,
        capabilities: DeepSeekV4FlashGPUCapabilities | None = None,
    ) -> None:
        self.capabilities = capabilities or DeepSeekV4FlashGPUCapabilities()
        self._stats = {
            "quantized_expert_calls": 0,
            "q2_k_triton_calls": 0,
            "iq2_xxs_triton_calls": 0,
            "q2_iq2_reference_fallback_calls": 0,
        }

    @property
    def is_ready(self) -> bool:
        return self.capabilities.is_ready

    @property
    def missing_kernels(self) -> tuple[str, ...]:
        return self.capabilities.missing

    def require_ready(self) -> None:
        if not self.is_ready:
            missing = ", ".join(self.missing_kernels)
            raise RuntimeError(f"DeepSeek V4 Flash missing GPU kernels: {missing}")

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def output_logits(
        self,
        *,
        streams: torch.Tensor,
        lm_head_values: torch.Tensor,
        lm_head_scales: torch.Tensor,
        output_hc_weight: torch.Tensor,
        output_hc_scale: torch.Tensor,
        output_hc_base: torch.Tensor,
        output_norm_weight: torch.Tensor,
        block_size: int = 32,
    ) -> torch.Tensor:
        tensors = (
            streams,
            lm_head_values,
            lm_head_scales,
            output_hc_weight,
            output_hc_scale,
            output_hc_base,
            output_norm_weight,
        )
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError("DeepSeek V4 Flash output inputs must be CUDA tensors")
        return deepseek_v4_output_projection(
            DeepSeekV4OutputKernelInputs(
                streams=streams,
                lm_head_values=lm_head_values,
                lm_head_scales=lm_head_scales,
                output_hc_weight=output_hc_weight,
                output_hc_scale=output_hc_scale,
                output_hc_base=output_hc_base,
                output_norm_weight=output_norm_weight,
                block_size=block_size,
            )
        )

    def output_argmax(
        self,
        *,
        streams: torch.Tensor,
        lm_head_values: torch.Tensor,
        lm_head_scales: torch.Tensor,
        output_hc_weight: torch.Tensor,
        output_hc_scale: torch.Tensor,
        output_hc_base: torch.Tensor,
        output_norm_weight: torch.Tensor,
        block_size: int = 32,
        row_offset: int = 0,
    ) -> torch.Tensor:
        tensors = (
            streams,
            lm_head_values,
            lm_head_scales,
            output_hc_weight,
            output_hc_scale,
            output_hc_base,
            output_norm_weight,
        )
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError("DeepSeek V4 Flash output inputs must be CUDA tensors")
        return deepseek_v4_output_argmax(
            DeepSeekV4OutputKernelInputs(
                streams=streams,
                lm_head_values=lm_head_values,
                lm_head_scales=lm_head_scales,
                output_hc_weight=output_hc_weight,
                output_hc_scale=output_hc_scale,
                output_hc_base=output_hc_base,
                output_norm_weight=output_norm_weight,
                block_size=block_size,
            ),
            row_offset=row_offset,
        )

    def sliding_attention(
        self,
        *,
        query: torch.Tensor,
        kv_rows: torch.Tensor,
        attn_sinks: torch.Tensor | None,
        token_idx: int,
    ) -> torch.Tensor:
        tensors = (query, kv_rows, attn_sinks)
        if any(tensor is not None and not tensor.is_cuda for tensor in tensors):
            raise ValueError(
                "DeepSeek V4 Flash sliding attention inputs must be CUDA tensors"
            )
        return deepseek_v4_attention(
            DeepSeekV4AttentionKernelInputs(
                hidden=query,
                kv_rows=kv_rows,
                token_idx=token_idx,
                attn_sinks=attn_sinks,
            )
        )

    def routed_moe(
        self,
        *,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_outputs: torch.Tensor,
    ) -> torch.Tensor:
        tensors = (hidden, expert_ids, expert_weights, expert_outputs)
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError("DeepSeek V4 Flash MoE inputs must be CUDA tensors")
        return deepseek_v4_moe(
            DeepSeekV4MoEKernelInputs(
                hidden=hidden,
                expert_ids=expert_ids,
                expert_weights=expert_weights,
                expert_outputs=expert_outputs,
            )
        )

    def routed_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> torch.Tensor:
        tensors = (hidden, gate_weight, up_weight, down_weight)
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError(
                "DeepSeek V4 Flash routed expert GEMM inputs must be CUDA tensors"
            )
        hidden_f32 = hidden.to(torch.float32)
        gate = gate_weight.to(torch.float32).matmul(hidden_f32)
        up = up_weight.to(torch.float32).matmul(hidden_f32)
        return down_weight.to(torch.float32).matmul(F.silu(gate) * up)

    def quantized_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_payload: DeepSeekV4FlashQuantizedExpertPayloadLike,
        up_payload: DeepSeekV4FlashQuantizedExpertPayloadLike,
        down_payload: DeepSeekV4FlashQuantizedExpertPayloadLike,
    ) -> torch.Tensor:
        self._stats["quantized_expert_calls"] += 1
        if (
            gate_payload.ggml_type == GGML_TYPE_IQ2_XXS
            and up_payload.ggml_type == GGML_TYPE_IQ2_XXS
            and gate_payload.rows == up_payload.rows
            and gate_payload.columns == up_payload.columns
        ):
            self._stats["iq2_xxs_triton_calls"] += 2
            gate, up = deepseek_v4_iq2_xxs_gate_up(
                gate_payload.payload,
                up_payload.payload,
                hidden,
                rows=gate_payload.rows,
                columns=gate_payload.columns,
            )
        else:
            gate = self._quantized_expert_matvec(gate_payload, hidden)
            up = self._quantized_expert_matvec(up_payload, hidden)
        return self._quantized_expert_matvec(down_payload, F.silu(gate) * up)

    def _quantized_expert_matvec(
        self,
        staged: DeepSeekV4FlashQuantizedExpertPayloadLike,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        if staged.ggml_type == GGML_TYPE_Q2_K:
            self._stats["q2_k_triton_calls"] += 1
            return deepseek_v4_q2_k_matvec(
                staged.payload,
                hidden,
                rows=staged.rows,
                columns=staged.columns,
            )
        if staged.ggml_type == GGML_TYPE_IQ2_XXS:
            self._stats["iq2_xxs_triton_calls"] += 1
            return deepseek_v4_iq2_xxs_matvec(
                staged.payload,
                hidden,
                rows=staged.rows,
                columns=staged.columns,
            )
        raise NotImplementedError(
            "DeepSeek V4 Flash quantized expert GEMM does not support "
            f"GGML type {staged.ggml_type}"
        )

    def compressed_attention(
        self,
        *,
        query: torch.Tensor,
        compressed_rows: torch.Tensor,
        selected_rows: torch.Tensor,
    ) -> torch.Tensor:
        tensors = (query, compressed_rows, selected_rows)
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError(
                "DeepSeek V4 Flash compressed attention inputs must be CUDA tensors"
            )
        return deepseek_v4_compressed_attention(
            DeepSeekV4CompressedAttentionTensorInputs(
                query=query,
                compressed_rows=compressed_rows,
                selected_rows=selected_rows,
            )
        )
