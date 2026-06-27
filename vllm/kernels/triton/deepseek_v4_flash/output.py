# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


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


def deepseek_v4_output_hidden(inputs: DeepSeekV4OutputKernelInputs) -> torch.Tensor:
    tensors = [
        inputs.streams,
        inputs.output_hc_weight,
        inputs.output_hc_scale,
        inputs.output_hc_base,
        inputs.output_norm_weight,
    ]
    if any(tensor is not None and not tensor.is_cuda for tensor in tensors):
        raise ValueError("DeepSeek V4 output hidden inputs must be CUDA tensors")

    streams = inputs.streams.to(torch.float32)
    if inputs.output_hc_weight is not None:
        if inputs.output_hc_scale is None or inputs.output_hc_base is None:
            raise ValueError(
                "output_hc_scale and output_hc_base are required with output_hc_weight"
            )
        flat = streams.reshape(-1)
        flat = flat * torch.rsqrt(flat.pow(2).mean() + 1e-6)
        pre = inputs.output_hc_weight.to(torch.float32).matmul(flat)
        weights = (
            torch.sigmoid(
                pre * inputs.output_hc_scale.to(torch.float32)[0]
                + inputs.output_hc_base.to(torch.float32)
            )
            + 1e-6
        )
        hidden = (weights.reshape(4, 1) * streams).sum(dim=0)
    else:
        hidden = streams.mean(dim=0)

    if inputs.output_norm_weight is not None:
        hidden = hidden * torch.rsqrt(hidden.pow(2).mean() + 1e-6)
        hidden = hidden * inputs.output_norm_weight.to(torch.float32)
    return hidden.to(torch.float32)


# Q8_0 output row layout after staging:
# - lm_head_values is row-major int8 [vocab_rows, hidden_size].
# - lm_head_scales is row-major fp32 [vocab_rows, hidden_size / 32].
# One Triton program handles one vocab row. It loops over 32-value Q8_0
# blocks, multiplies dequantized int8 values by the dense hidden vector, and
# stores a single fp32 logit. This avoids materializing the decoded matrix.
@triton.jit
def _q8_0_output_matvec_kernel(
    values_ptr,
    scales_ptr,
    hidden_ptr,
    logits_ptr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_ROW: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    total = tl.full((), 0.0, tl.float32)
    for block_idx in tl.static_range(0, BLOCKS_PER_ROW):
        value_offsets = row * BLOCKS_PER_ROW * BLOCK_SIZE + block_idx * BLOCK_SIZE
        values = tl.load(values_ptr + value_offsets + offsets).to(tl.float32)
        scale = tl.load(scales_ptr + row * BLOCKS_PER_ROW + block_idx).to(tl.float32)
        hidden = tl.load(hidden_ptr + block_idx * BLOCK_SIZE + offsets).to(
            tl.float32
        )
        total += tl.sum(values * scale * hidden, axis=0)
    tl.store(logits_ptr + row, total)


def _q8_0_output_matvec_triton_cuda(
    values: torch.Tensor,
    scales: torch.Tensor,
    hidden: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    if block_size != 32:
        raise ValueError("DeepSeek V4 Q8_0 output kernel requires block_size=32")
    rows, columns = values.shape
    blocks_per_row = columns // block_size
    logits = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    _q8_0_output_matvec_kernel[(rows,)](
        values.contiguous(),
        scales.contiguous(),
        hidden.contiguous(),
        logits,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_ROW=blocks_per_row,
        num_warps=1,
    )
    return logits


def _q8_0_output_matvec(
    values: torch.Tensor,
    scales: torch.Tensor,
    hidden: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    if block_size == 32:
        return _q8_0_output_matvec_triton_cuda(
            values,
            scales,
            hidden,
            block_size=block_size,
        )
    rows, columns = values.shape
    blocks_per_row = columns // block_size
    decoded = values.to(torch.float32).reshape(rows, blocks_per_row, block_size)
    row_scales = scales.to(torch.float32).reshape(rows, blocks_per_row, 1)
    decoded_matrix = (decoded * row_scales).reshape(rows, columns)
    return decoded_matrix.matmul(hidden.to(torch.float32))


def deepseek_v4_q8_0_output_logits(
    *,
    hidden: torch.Tensor,
    lm_head_values: torch.Tensor,
    lm_head_scales: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    if not hidden.is_cuda or not lm_head_values.is_cuda or not lm_head_scales.is_cuda:
        raise ValueError("DeepSeek V4 Q8_0 output inputs must be CUDA tensors")
    return _q8_0_output_matvec(
        lm_head_values,
        lm_head_scales,
        hidden.to(torch.float32),
        block_size=block_size,
    )


def deepseek_v4_q8_0_output_argmax_with_value(
    *,
    hidden: torch.Tensor,
    lm_head_values: torch.Tensor,
    lm_head_scales: torch.Tensor,
    block_size: int = 32,
    row_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = deepseek_v4_q8_0_output_logits(
        hidden=hidden,
        lm_head_values=lm_head_values,
        lm_head_scales=lm_head_scales,
        block_size=block_size,
    )
    value, index = torch.max(logits, dim=0)
    token = index.to(torch.long) + int(row_offset)
    return token.reshape(()), value.to(torch.float32).reshape(())


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

    return _q8_0_output_matvec(
        inputs.lm_head_values,
        inputs.lm_head_scales,
        deepseek_v4_output_hidden(inputs),
        block_size=inputs.block_size,
    )


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

    logits = _q8_0_output_matvec(
        inputs.lm_head_values,
        inputs.lm_head_scales,
        deepseek_v4_output_hidden(inputs),
        block_size=inputs.block_size,
    )
    value, index = torch.max(logits, dim=0)
    token = index.to(torch.long) + int(row_offset)
    return token.reshape(()), value.to(torch.float32).reshape(())
