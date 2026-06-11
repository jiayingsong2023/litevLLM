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
    expert_outputs: torch.Tensor | None = None

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
        if self.expert_outputs is not None:
            if self.expert_outputs.ndim != 2:
                raise ValueError(
                    f"expert_outputs must be 2-D; got {self.expert_outputs.ndim}-D"
                )
            if self.expert_outputs.shape[1] != self.hidden.numel():
                raise ValueError("expert_outputs width must match hidden size")
            if int(torch.max(self.expert_ids).item()) >= self.expert_outputs.shape[0]:
                raise ValueError("expert_ids exceed expert_outputs rows")


def deepseek_v4_moe(inputs: DeepSeekV4MoEKernelInputs) -> torch.Tensor:
    tensors = [
        inputs.hidden,
        inputs.expert_ids,
        inputs.expert_weights,
        inputs.expert_outputs,
    ]
    if any(tensor is not None and not tensor.is_cuda for tensor in tensors):
        raise ValueError("DeepSeek V4 MoE inputs must be CUDA tensors")
    if inputs.expert_outputs is None:
        raise NotImplementedError("DeepSeek V4 Flash expert GEMM is not implemented")
    selected = inputs.expert_outputs.index_select(0, inputs.expert_ids.to(torch.long))
    weights = inputs.expert_weights.to(torch.float32).reshape(-1, 1)
    return (selected.to(torch.float32) * weights).sum(dim=0)
