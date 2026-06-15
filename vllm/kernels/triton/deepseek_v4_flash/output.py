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
    output_hc_weight: torch.Tensor | None = None
    output_hc_scale: torch.Tensor | None = None
    output_hc_base: torch.Tensor | None = None
    output_norm_weight: torch.Tensor | None = None
    block_size: int = 32

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
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive; got {self.block_size}")
        if self.lm_head_values.shape[1] % self.block_size != 0:
            raise ValueError("lm_head_values columns must be divisible by block_size")
        expected_scales = (
            self.lm_head_values.shape[0],
            self.lm_head_values.shape[1] // self.block_size,
        )
        if tuple(self.lm_head_scales.shape) != expected_scales:
            raise ValueError(
                "lm_head_scales shape must be "
                f"{expected_scales}; got {tuple(self.lm_head_scales.shape)}"
            )
        if self.output_hc_weight is not None:
            expected_hc = (4, int(self.streams.numel()))
            if tuple(self.output_hc_weight.shape) != expected_hc:
                raise ValueError(
                    f"output_hc_weight shape must be {expected_hc}; "
                    f"got {tuple(self.output_hc_weight.shape)}"
                )
        if (
            self.output_hc_scale is not None
            and tuple(self.output_hc_scale.shape) != (1,)
        ):
            raise ValueError("output_hc_scale must have shape [1]")
        if (
            self.output_hc_base is not None
            and tuple(self.output_hc_base.shape) != (4,)
        ):
            raise ValueError("output_hc_base must have shape [4]")
        if self.output_norm_weight is not None:
            expected_norm = (self.streams.shape[1],)
            if tuple(self.output_norm_weight.shape) != expected_norm:
                raise ValueError(
                    f"output_norm_weight shape must be {expected_norm}; "
                    f"got {tuple(self.output_norm_weight.shape)}"
                )


def deepseek_v4_output_projection(inputs: DeepSeekV4OutputKernelInputs) -> torch.Tensor:
    tensors = [
        inputs.streams,
        inputs.lm_head_values,
        inputs.lm_head_scales,
        inputs.output_hc_weight,
        inputs.output_hc_scale,
        inputs.output_hc_base,
        inputs.output_norm_weight,
    ]
    if any(tensor is not None and not tensor.is_cuda for tensor in tensors):
        raise ValueError("DeepSeek V4 output inputs must be CUDA tensors")

    hidden = inputs.streams.to(torch.float32)
    if inputs.output_hc_weight is not None:
        if inputs.output_hc_scale is None or inputs.output_hc_base is None:
            raise ValueError(
                "output_hc_scale and output_hc_base are required with output_hc_weight"
            )
        flat = hidden.reshape(-1)
        flat = flat * torch.rsqrt(flat.pow(2).mean() + 1e-6)
        pre = inputs.output_hc_weight.to(torch.float32).matmul(flat)
        weights = (
            torch.sigmoid(
                pre * inputs.output_hc_scale.to(torch.float32)[0]
                + inputs.output_hc_base.to(torch.float32)
            )
            + 1e-6
        )
        hidden = (weights.reshape(4, 1) * hidden).sum(dim=0)
    else:
        hidden = hidden.mean(dim=0)

    if inputs.output_norm_weight is not None:
        hidden = hidden * torch.rsqrt(hidden.pow(2).mean() + 1e-6)
        hidden = hidden * inputs.output_norm_weight.to(torch.float32)

    rows, columns = inputs.lm_head_values.shape
    blocks_per_row = columns // inputs.block_size
    decoded = inputs.lm_head_values.to(torch.float32).reshape(
        rows,
        blocks_per_row,
        inputs.block_size,
    )
    scales = inputs.lm_head_scales.to(torch.float32).reshape(rows, blocks_per_row, 1)
    matrix = (decoded * scales).reshape(rows, columns)
    return matrix.matmul(hidden.to(torch.float32))


def deepseek_v4_output_argmax(
    inputs: DeepSeekV4OutputKernelInputs,
    *,
    row_offset: int,
) -> torch.Tensor:
    token, _value = deepseek_v4_output_argmax_with_value(
        inputs,
        row_offset=row_offset,
    )
    return token


def deepseek_v4_output_argmax_with_value(
    inputs: DeepSeekV4OutputKernelInputs,
    *,
    row_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = deepseek_v4_output_projection(inputs)
    value, index = torch.max(logits, dim=0)
    token = index.to(torch.long) + int(row_offset)
    return token.reshape(()), value.to(torch.float32).reshape(())
