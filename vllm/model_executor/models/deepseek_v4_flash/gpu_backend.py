# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from vllm.kernels.triton.deepseek_v4_flash.attention import (
    DeepSeekV4AttentionKernelInputs,
    deepseek_v4_attention,
    deepseek_v4_fused_sliding_window_attention,
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
    deepseek_v4_output_argmax_with_value,
    deepseek_v4_output_hidden,
    deepseek_v4_output_projection,
    deepseek_v4_q8_0_output_argmax_with_value,
    deepseek_v4_q8_0_output_logits,
)
from vllm.kernels.triton.deepseek_v4_flash.q2_iq2_moe import (
    deepseek_v4_iq2_xxs_gate_up,
    deepseek_v4_iq2_xxs_gate_up_activation,
    deepseek_v4_iq2_xxs_matvec,
    deepseek_v4_iq2_xxs_selected_experts_activation_direct,
    deepseek_v4_q2_k_matvec,
    deepseek_v4_q2_k_selected_experts_down_projection_direct,
)
from vllm.model_executor.models.deepseek_v4_flash.gguf_reader import (
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_Q2_K,
)
from vllm.model_executor.models.deepseek_v4_flash.ops import (
    deepseek_q8_k_roundtrip_reference,
)

from .config import DEEPSEEK_V4_FLASH_SHAPE
from .gpu_weight_staging import (
    DeepSeekV4FlashGPUWeightStager,
    DeepSeekV4FlashSelectedExpertPayloads,
)
from .weight_store import DeepSeekV4FlashGroupedExpertTensors


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
            "iq2_xxs_gate_up_fused_calls": 0,
            "q2_iq2_reference_fallback_calls": 0,
            "cpu_token_sync_points": 0,
            "fused_sliding_attention_api_calls": 0,
        }
        self.profiler: Any | None = None

    def _profile_section(self, name: str, **metadata: Any) -> Iterator[None]:
        if self.profiler is None:
            return nullcontext()
        section = getattr(self.profiler, "section", None)
        if not callable(section):
            return nullcontext()
        return section(name, **metadata)

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

    def output_hidden(
        self,
        *,
        streams: torch.Tensor,
        output_hc_weight: torch.Tensor,
        output_hc_scale: torch.Tensor,
        output_hc_base: torch.Tensor,
        output_norm_weight: torch.Tensor,
        lm_head_values: torch.Tensor,
        lm_head_scales: torch.Tensor,
        block_size: int = 32,
    ) -> torch.Tensor:
        tensors = (
            streams,
            output_hc_weight,
            output_hc_scale,
            output_hc_base,
            output_norm_weight,
            lm_head_values,
            lm_head_scales,
        )
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError(
                "DeepSeek V4 Flash output hidden inputs must be CUDA tensors"
            )
        return deepseek_v4_output_hidden(
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

    def output_logits_from_hidden(
        self,
        *,
        hidden: torch.Tensor,
        lm_head_values: torch.Tensor,
        lm_head_scales: torch.Tensor,
        block_size: int = 32,
    ) -> torch.Tensor:
        tensors = (hidden, lm_head_values, lm_head_scales)
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError("DeepSeek V4 Flash output inputs must be CUDA tensors")
        return deepseek_v4_q8_0_output_logits(
            hidden=hidden,
            lm_head_values=lm_head_values,
            lm_head_scales=lm_head_scales,
            block_size=block_size,
        )

    def output_argmax_from_hidden(
        self,
        *,
        hidden: torch.Tensor,
        lm_head_values: torch.Tensor,
        lm_head_scales: torch.Tensor,
        block_size: int = 32,
        row_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = (hidden, lm_head_values, lm_head_scales)
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError("DeepSeek V4 Flash output inputs must be CUDA tensors")
        return deepseek_v4_q8_0_output_argmax_with_value(
            hidden=hidden,
            lm_head_values=lm_head_values,
            lm_head_scales=lm_head_scales,
            block_size=block_size,
            row_offset=row_offset,
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

    def output_argmax_with_value(
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return deepseek_v4_output_argmax_with_value(
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

    def fused_sliding_window_attention(
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
                "DeepSeek V4 Flash fused sliding attention inputs must be CUDA tensors"
            )
        self._stats["fused_sliding_attention_api_calls"] += 1
        return deepseek_v4_fused_sliding_window_attention(
            query=query.to(torch.float32),
            kv_rows=kv_rows.to(torch.float32),
            attn_sinks=attn_sinks.to(torch.float32) if attn_sinks is not None else None,
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
        with self._profile_section("selected_expert_up_gate"):
            gate = gate_weight.to(torch.float32).matmul(hidden_f32)
            up = up_weight.to(torch.float32).matmul(hidden_f32)
            clamp = float(DEEPSEEK_V4_FLASH_SHAPE.swiglu_clamp)
            if clamp > 1.0e-6:
                gate = torch.clamp(gate, max=clamp)
                up = torch.clamp(up, min=-clamp, max=clamp)
            activated = F.silu(gate) * up
        with self._profile_section("selected_expert_down"):
            return down_weight.to(torch.float32).matmul(activated)

    def quantized_expert_gemm(
        self,
        *,
        hidden: torch.Tensor,
        gate_payload: DeepSeekV4FlashQuantizedExpertPayloadLike,
        up_payload: DeepSeekV4FlashQuantizedExpertPayloadLike,
        down_payload: DeepSeekV4FlashQuantizedExpertPayloadLike,
    ) -> torch.Tensor:
        self._stats["quantized_expert_calls"] += 1
        return self._quantized_expert_gemm_from_q8_input(
            expert_input=deepseek_q8_k_roundtrip_reference(hidden),
            gate_payload=gate_payload,
            up_payload=up_payload,
            down_payload=down_payload,
        )

    def _quantized_expert_gemm_from_q8_input(
        self,
        *,
        expert_input: torch.Tensor,
        gate_payload: DeepSeekV4FlashQuantizedExpertPayloadLike,
        up_payload: DeepSeekV4FlashQuantizedExpertPayloadLike,
        down_payload: DeepSeekV4FlashQuantizedExpertPayloadLike,
    ) -> torch.Tensor:
        with self._profile_section("selected_expert_up_gate"):
            if (
                gate_payload.ggml_type == GGML_TYPE_IQ2_XXS
                and up_payload.ggml_type == GGML_TYPE_IQ2_XXS
                and gate_payload.rows == up_payload.rows
                and gate_payload.columns == up_payload.columns
                and gate_payload.columns % 256 == 0
            ):
                self._stats["iq2_xxs_gate_up_fused_calls"] += 1
                activated = deepseek_v4_iq2_xxs_gate_up_activation(
                    gate_payload.payload,
                    up_payload.payload,
                    expert_input,
                    rows=gate_payload.rows,
                    columns=gate_payload.columns,
                )
                activated = deepseek_q8_k_roundtrip_reference(activated)
            else:
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
                        expert_input,
                        rows=gate_payload.rows,
                        columns=gate_payload.columns,
                    )
                else:
                    gate = self._quantized_expert_matvec(gate_payload, expert_input)
                    up = self._quantized_expert_matvec(up_payload, expert_input)
                clamp = float(DEEPSEEK_V4_FLASH_SHAPE.swiglu_clamp)
                if clamp > 1.0e-6:
                    gate = torch.clamp(gate, max=clamp)
                    up = torch.clamp(up, min=-clamp, max=clamp)
                activated = F.silu(gate) * up
                activated = deepseek_q8_k_roundtrip_reference(activated)
        with self._profile_section("selected_expert_down"):
            return self._quantized_expert_matvec(down_payload, activated)

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

    def quantized_selected_experts_gemm(
        self,
        *,
        hidden: torch.Tensor,
        expert_weights: torch.Tensor,
        payloads: list[
            tuple[
                int,
                DeepSeekV4FlashQuantizedExpertPayloadLike,
                DeepSeekV4FlashQuantizedExpertPayloadLike,
                DeepSeekV4FlashQuantizedExpertPayloadLike,
            ]
        ],
    ) -> torch.Tensor:
        if not hidden.is_cuda or not expert_weights.is_cuda:
            raise ValueError(
                "DeepSeek V4 Flash selected expert GEMM inputs must be CUDA tensors"
            )
        self._stats["fused_selected_expert_api_calls"] = (
            self._stats.get("fused_selected_expert_api_calls", 0) + 1
        )
        if not payloads:
            raise ValueError("DeepSeek V4 Flash selected expert GEMM got no payloads")
        output: torch.Tensor | None = None
        weights = expert_weights.reshape(-1)
        expert_input = deepseek_q8_k_roundtrip_reference(hidden)
        for payload_index, (
            _expert_id,
            gate_payload,
            up_payload,
            down_payload,
        ) in enumerate(payloads):
            self._stats["quantized_expert_calls"] += 1
            expert_output = self._quantized_expert_gemm_from_q8_input(
                expert_input=expert_input,
                gate_payload=gate_payload,
                up_payload=up_payload,
                down_payload=down_payload,
            ).to(torch.float32)
            if output is None:
                output = torch.zeros_like(expert_output, dtype=torch.float32)
            with self._profile_section("selected_expert_combine"):
                output = (
                    output + weights[payload_index].to(torch.float32) * expert_output
                )
        if output is None:
            raise ValueError("DeepSeek V4 Flash selected expert GEMM got no payloads")
        return output

    def fused_quantized_selected_experts_gemm(
        self,
        *,
        hidden: torch.Tensor,
        expert_weights: torch.Tensor,
        payloads: list[
            tuple[
                int,
                DeepSeekV4FlashQuantizedExpertPayloadLike,
                DeepSeekV4FlashQuantizedExpertPayloadLike,
                DeepSeekV4FlashQuantizedExpertPayloadLike,
            ]
        ],
        workspace: torch.Tensor,
        gate_stack: torch.Tensor | None = None,
        up_stack: torch.Tensor | None = None,
        down_stack: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not hidden.is_cuda or not expert_weights.is_cuda or not workspace.is_cuda:
            raise ValueError("fused selected expert GEMM inputs must be CUDA tensors")
        if not payloads:
            raise ValueError("fused selected expert GEMM got no payloads")
        self._stats["fused_selected_expert_api_calls"] = (
            self._stats.get("fused_selected_expert_api_calls", 0) + 1
        )
        gate_payloads = []
        up_payloads = []
        down_payloads = []
        for payload_index, (_expert_id, gate, up, down) in enumerate(payloads):
            gate_payloads.append(gate.payload)
            up_payloads.append(up.payload)
            down_payloads.append(down.payload)
        rows = payloads[0][1].rows
        columns = payloads[0][1].columns
        with self._profile_section("selected_expert_up_gate"):
            deepseek_v4_iq2_xxs_selected_experts_activation_direct(
                hidden=hidden.to(torch.float32),
                payloads=list(zip(gate_payloads, up_payloads)),
                workspace=workspace,
                rows=rows,
                columns=columns,
            )
        down_rows = payloads[0][3].rows
        down_columns = payloads[0][3].columns
        output = torch.empty((down_rows,), dtype=torch.float32, device=hidden.device)
        with self._profile_section("selected_expert_down"):
            deepseek_v4_q2_k_selected_experts_down_projection_direct(
                workspace=workspace,
                down_payloads=down_payloads,
                expert_weights=expert_weights.reshape(-1),
                output=output,
                rows=down_rows,
                columns=down_columns,
            )
        return output

    def stage_selected_expert_payloads(
        self,
        stager: DeepSeekV4FlashGPUWeightStager,
        grouped_experts: DeepSeekV4FlashGroupedExpertTensors,
        expert_ids: torch.Tensor,
        *,
        layer_idx: int | None = None,
    ) -> list[DeepSeekV4FlashSelectedExpertPayloads]:
        return stager.copy_selected_expert_payload_bytes(
            grouped_experts,
            expert_ids,
            layer_idx=layer_idx,
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
