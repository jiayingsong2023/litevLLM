# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.kernels.triton.deepseek_v4_flash.output import (
    DeepSeekV4OutputKernelInputs,
    deepseek_v4_output_projection,
)


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
