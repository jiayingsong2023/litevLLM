# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeepSeekV4OutputKernelInputs:
    """Kernel input contract for DeepSeek V4 output collapse and projection.

    Memory layout:
    - streams is [4, hidden_size] after the final transformer layer.
    - lm_head_values is [vocab_size, hidden_size] in a GPU-friendly packed form.
    - lm_head_scales is [vocab_size, hidden_size / block_size].

    Tiling:
    - Future kernels should first collapse hyper-connection streams to one
      hidden vector, then tile the Q8 output projection by vocab rows.
    - The output tensor is [vocab_size] fp32 logits for batch=1 decode.
    """

    streams: torch.Tensor
    lm_head_values: torch.Tensor
    lm_head_scales: torch.Tensor

    def __post_init__(self) -> None:
        if self.streams.ndim != 2 or self.streams.shape[0] != 4:
            raise ValueError("streams must have shape [4, hidden_size]")
        if self.lm_head_values.ndim != 2:
            raise ValueError(
                f"lm_head_values must be 2-D; got {self.lm_head_values.ndim}-D"
            )
        if self.lm_head_scales.ndim != 2:
            raise ValueError(
                f"lm_head_scales must be 2-D; got {self.lm_head_scales.ndim}-D"
            )


def deepseek_v4_output_projection(inputs: DeepSeekV4OutputKernelInputs) -> torch.Tensor:
    del inputs
    raise NotImplementedError("DeepSeek V4 Flash output kernel is not implemented")
