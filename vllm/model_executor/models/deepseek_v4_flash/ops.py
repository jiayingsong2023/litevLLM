# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import torch.nn.functional as F

_E4M3FN_VALUES = (
    0.0,
    0.001953125,
    0.00390625,
    0.005859375,
    0.0078125,
    0.009765625,
    0.01171875,
    0.013671875,
    0.015625,
    0.017578125,
    0.01953125,
    0.021484375,
    0.0234375,
    0.025390625,
    0.02734375,
    0.029296875,
    0.03125,
    0.03515625,
    0.0390625,
    0.04296875,
    0.046875,
    0.05078125,
    0.0546875,
    0.05859375,
    0.0625,
    0.0703125,
    0.078125,
    0.0859375,
    0.09375,
    0.1015625,
    0.109375,
    0.1171875,
    0.125,
    0.140625,
    0.15625,
    0.171875,
    0.1875,
    0.203125,
    0.21875,
    0.234375,
    0.25,
    0.28125,
    0.3125,
    0.34375,
    0.375,
    0.40625,
    0.4375,
    0.46875,
    0.5,
    0.5625,
    0.625,
    0.6875,
    0.75,
    0.8125,
    0.875,
    0.9375,
    1.0,
    1.125,
    1.25,
    1.375,
    1.5,
    1.625,
    1.75,
    1.875,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    18.0,
    20.0,
    22.0,
    24.0,
    26.0,
    28.0,
    30.0,
    32.0,
    36.0,
    40.0,
    44.0,
    48.0,
    52.0,
    56.0,
    60.0,
    64.0,
    72.0,
    80.0,
    88.0,
    96.0,
    104.0,
    112.0,
    120.0,
    128.0,
    144.0,
    160.0,
    176.0,
    192.0,
    208.0,
    224.0,
    240.0,
    256.0,
    288.0,
    320.0,
    352.0,
    384.0,
    416.0,
    448.0,
)

_E2M1FN_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _nearest_table_values(
    values: torch.Tensor,
    table_values: tuple[float, ...],
) -> torch.Tensor:
    values_f32 = values.to(torch.float32)
    sign = torch.where(values_f32 < 0.0, -1.0, 1.0)
    table = torch.tensor(table_values, dtype=torch.float32, device=values.device)
    max_value = float(table[-1].item())
    abs_values = torch.clamp(values_f32.abs(), max=max_value)
    diffs = torch.abs(abs_values.reshape(-1, 1) - table.reshape(1, -1))
    odd_penalty = (
        torch.arange(table.numel(), device=values.device, dtype=torch.float32) % 2.0
    ) * 1.0e-12
    indices = torch.argmin(diffs + odd_penalty.reshape(1, -1), dim=1)
    return (sign.reshape(-1) * table.index_select(0, indices)).reshape_as(values_f32)


def e4m3fn_dequant_reference(values: torch.Tensor) -> torch.Tensor:
    return _nearest_table_values(values, _E4M3FN_VALUES)


def _e2m1fn_dequant_reference(values: torch.Tensor) -> torch.Tensor:
    return _nearest_table_values(values, _E2M1FN_VALUES)


def deepseek_fp8_kv_qat_reference(
    row: torch.Tensor,
    *,
    head_dim: int,
    rotary_dim: int,
) -> torch.Tensor:
    if row.ndim != 1:
        raise ValueError(f"KV QAT row must be 1-D; got {row.ndim}-D")
    if row.numel() != head_dim:
        raise ValueError(f"KV QAT row width must be {head_dim}; got {row.numel()}")
    if rotary_dim < 0 or rotary_dim > head_dim:
        raise ValueError(
            f"rotary_dim must be within [0, {head_dim}]; got {rotary_dim}"
        )
    n_nope = head_dim - rotary_dim
    if n_nope % 64 != 0:
        raise ValueError(f"non-RoPE KV width must be 64-aligned; got {n_nope}")
    out = row.to(torch.float32).clone()
    for offset in range(0, n_nope, 64):
        block = out[offset : offset + 64]
        amax = torch.clamp(block.abs().max(), min=1.0e-4)
        scale = torch.pow(
            torch.tensor(2.0, dtype=torch.float32, device=row.device),
            torch.ceil(torch.log2(amax / 448.0)),
        )
        quant_input = torch.clamp(block / scale, min=-448.0, max=448.0)
        out[offset : offset + 64] = e4m3fn_dequant_reference(quant_input) * scale
    return out


def _hadamard128_reference(row: torch.Tensor) -> torch.Tensor:
    if row.shape != (128,):
        raise ValueError(f"Hadamard row must have shape (128,); got {tuple(row.shape)}")
    out = row.to(torch.float32).clone()
    stride = 1
    while stride < 128:
        for base in range(0, 128, 2 * stride):
            left = out[base : base + stride].clone()
            right = out[base + stride : base + 2 * stride].clone()
            out[base : base + stride] = left + right
            out[base + stride : base + 2 * stride] = left - right
        stride <<= 1
    return out * 0.08838834764831845


def deepseek_indexer_qat_reference(row: torch.Tensor) -> torch.Tensor:
    if row.shape != (128,):
        raise ValueError(
            "indexer QAT row must have shape (128,); "
            f"got {tuple(row.shape)}"
        )
    out = _hadamard128_reference(row)
    for offset in range(0, 128, 32):
        block = out[offset : offset + 32]
        amax = torch.clamp(block.abs().max(), min=7.052966104933725e-38)
        scale = torch.pow(
            torch.tensor(2.0, dtype=torch.float32, device=row.device),
            torch.ceil(torch.log2(amax / 6.0)),
        )
        quant_input = torch.clamp(block / scale, min=-6.0, max=6.0)
        out[offset : offset + 32] = _e2m1fn_dequant_reference(quant_input) * scale
    return out


def deepseek_q8_k_roundtrip_reference(
    row: torch.Tensor,
    *,
    block_size: int = 256,
) -> torch.Tensor:
    """Round-trip a vector through DS4's routed-expert Q8_K quantizer."""
    if row.ndim != 1:
        raise ValueError(f"Q8_K row must be 1-D; got {row.ndim}-D")
    if block_size <= 0:
        raise ValueError(f"Q8_K block_size must be positive; got {block_size}")
    if row.numel() % block_size != 0:
        raise ValueError(
            "Q8_K row length must be divisible by block_size; "
            f"got {row.numel()} and {block_size}"
        )

    blocks = row.to(torch.float32).reshape(-1, block_size)
    max_indices = torch.argmax(blocks.abs(), dim=1)
    signed_max = blocks.gather(1, max_indices.reshape(-1, 1)).reshape(-1, 1)
    nonzero = signed_max.abs() > 0.0
    iscale = torch.where(
        nonzero,
        torch.full_like(signed_max, -127.0) / signed_max,
        torch.zeros_like(signed_max),
    )
    quantized = torch.round(blocks * iscale).clamp(min=-128, max=127)
    scale = torch.where(nonzero, 1.0 / iscale, torch.zeros_like(iscale))
    return (quantized * scale).reshape_as(row).to(torch.float32)


def rms_norm_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D; got {hidden.ndim}-D")
    if weight.ndim != 1:
        raise ValueError(f"weight must be 1-D; got {weight.ndim}-D")
    if weight.numel() != hidden.numel():
        raise ValueError(
            "weight length must match hidden size; "
            f"got {weight.numel()} and {hidden.numel()}"
        )

    hidden_f32 = hidden.to(torch.float32)
    variance = hidden_f32.pow(2).mean()
    return hidden_f32 * torch.rsqrt(variance + eps) * weight.to(torch.float32)


def silu_gate_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    clamp: float | None = None,
) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError(
            "gate and up shapes must match; "
            f"got {tuple(gate.shape)} and {tuple(up.shape)}"
        )
    gate_f32 = gate.to(torch.float32)
    up_f32 = up.to(torch.float32)
    if clamp is not None and clamp > 1.0e-6:
        gate_f32 = torch.clamp(gate_f32, max=float(clamp))
        up_f32 = torch.clamp(up_f32, min=-float(clamp), max=float(clamp))
    return F.silu(gate_f32) * up_f32


def factorized_linear_reference(
    hidden: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    if hidden.ndim != 1:
        raise ValueError(f"hidden must be 1-D; got {hidden.ndim}-D")
    if a.ndim != 2:
        raise ValueError(f"a must be 2-D; got {a.ndim}-D")
    if b.ndim != 2:
        raise ValueError(f"b must be 2-D; got {b.ndim}-D")
    if a.shape[1] != hidden.numel():
        raise ValueError(
            "a columns must match hidden size; "
            f"got {a.shape[1]} and {hidden.numel()}"
        )
    if b.shape[1] != a.shape[0]:
        raise ValueError(
            "b columns must match a rows; "
            f"got {b.shape[1]} and {a.shape[0]}"
        )

    intermediate = a.to(torch.float32).matmul(hidden.to(torch.float32))
    return b.to(torch.float32).matmul(intermediate)
