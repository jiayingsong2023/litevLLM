# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeepSeekV4MoEKernelInputs:
    """Kernel input contract for DeepSeek V4 routed MoE execution.

    Memory layout:
    - hidden is [hidden_size] for batch=1 decode.
    - expert_ids is [top_k] and contains routed expert ids for this token.
    - expert_weights is [top_k] and contains router probabilities.

    Tiling:
    - Future kernels should group selected experts and tile expert matvec by
      output channel and quantization block.
    - Shared expert and routed expert accumulation should return one
      [hidden_size] vector to the layer runner.
    """

    hidden: torch.Tensor
    expert_ids: torch.Tensor
    expert_weights: torch.Tensor

    def __post_init__(self) -> None:
        if self.hidden.ndim != 1:
            raise ValueError(f"hidden must be 1-D; got {self.hidden.ndim}-D")
        if self.expert_ids.ndim != 1:
            raise ValueError(f"expert_ids must be 1-D; got {self.expert_ids.ndim}-D")
        if self.expert_weights.ndim != 1:
            raise ValueError(
                f"expert_weights must be 1-D; got {self.expert_weights.ndim}-D"
            )
        if self.expert_ids.numel() != self.expert_weights.numel():
            raise ValueError("expert_ids and expert_weights must have the same length")


def deepseek_v4_moe(inputs: DeepSeekV4MoEKernelInputs) -> torch.Tensor:
    del inputs
    raise NotImplementedError("DeepSeek V4 Flash MoE kernel is not implemented")
