# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import time

import torch

from vllm.triton_utils import tl, triton

_TUNED_GATE_UP_BLOCK_PACKS = 8
_TUNED_DOWN_BLOCK_PACKS = 8
_TUNED_DOWN_BLOCK_H_SMALL_H = 64
_TUNED_DOWN_BLOCK_H_LARGE_H = 128
_CHUNKED_BLOCK_I = 256
_CHUNKED_BLOCK_H = 64
_CHUNKED_BLOCK_PACKS_H = 8
_CHUNKED_BLOCK_PACKS_I = 8
_PROFILE_STATS: dict[str, dict[str, float]] = {}


def _profile_enabled() -> bool:
    return os.environ.get(
        "FASTINFERENCE_GEMMA4_MOE_KERNEL_PROFILE", ""
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def reset_moe_kernel_profile_stats() -> None:
    _PROFILE_STATS.clear()


def get_moe_kernel_profile_stats() -> dict[str, dict[str, float]]:
    return {
        scope: {"time_s": float(row["time_s"]), "count": float(row["count"])}
        for scope, row in _PROFILE_STATS.items()
    }


def _record_profile(scope: str, elapsed_s: float) -> None:
    row = _PROFILE_STATS.setdefault(scope, {"time_s": 0.0, "count": 0.0})
    row["time_s"] += float(elapsed_s)
    row["count"] += 1.0


class _ProfileSpan:
    def __init__(self, scope: str):
        self.scope = scope
        self._enabled = False
        self._start = 0.0

    def __enter__(self) -> "_ProfileSpan":
        self._enabled = _profile_enabled()
        if self._enabled:
            torch.cuda.synchronize()
            self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._enabled:
            torch.cuda.synchronize()
            _record_profile(self.scope, time.perf_counter() - self._start)


def _profile_span(scope: str) -> _ProfileSpan:
    return _ProfileSpan(scope)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _activation_kind(activation: str) -> int:
    name = str(activation).strip().lower()
    if name in ("gelu", "gelu_pytorch_tanh"):
        return 1
    return 0


def _activation_supported(activation: str) -> bool:
    return str(activation).strip().lower() in (
        "silu",
        "swish",
        "gelu",
        "gelu_pytorch_tanh",
    )


@triton.jit
def _gemma4_moe_gate_up_m1_kernel(
    x_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    gate_up_q_ptr,
    gate_up_s_ptr,
    tmp_ptr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    stride_gu_e: tl.constexpr,
    stride_gu_n: tl.constexpr,
    stride_gu_k: tl.constexpr,
    stride_gus_e: tl.constexpr,
    stride_gus_n: tl.constexpr,
    stride_gus_g: tl.constexpr,
    stride_tmp_k: tl.constexpr,
    stride_tmp_i: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_PACKS: tl.constexpr,
    ACT_KIND: tl.constexpr,
):
    """
    Decode-only Gemma4 MoE gate/up stage for expert-major symmetric int4.

    Layout:
      x: [1, H]
      topk_ids/topk_weights: [1, top_k]
      gate_up_q: [E, 2 * intermediate, H/8], eight signed int4 values per
        packed int32
      gate_up_s: [E, 2 * intermediate, H/32], one scale per 32 input columns
      tmp: [top_k, intermediate], fp32 activated gate-up values already
        multiplied by the routing weight.

    Tiling:
      One program handles one top-k slot and BLOCK_I intermediate rows.
      It streams packed H columns in BLOCK_PACKS chunks and unpacks eight
      nibbles per packed int32 inside the program.
    """
    pid_k = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_i = offs_i < INTERMEDIATE
    offs_p = tl.arange(0, BLOCK_PACKS)

    expert_id = tl.load(topk_ids_ptr + pid_k).to(tl.int64)
    route_weight = tl.load(topk_weights_ptr + pid_k).to(tl.float32)
    num_packs = H // 8
    gate_acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_I,), dtype=tl.float32)

    for pack_base in range(0, tl.cdiv(num_packs, BLOCK_PACKS)):
        pack_idx = pack_base * BLOCK_PACKS + offs_p
        mask_p = pack_idx < num_packs
        group_idx = pack_idx // 4
        gate_packed = tl.load(
            gate_up_q_ptr
            + expert_id * stride_gu_e
            + offs_i[:, None] * stride_gu_n
            + pack_idx[None, :] * stride_gu_k,
            mask=mask_i[:, None] & mask_p[None, :],
            other=0,
        ).to(tl.int32)
        up_packed = tl.load(
            gate_up_q_ptr
            + expert_id * stride_gu_e
            + (INTERMEDIATE + offs_i[:, None]) * stride_gu_n
            + pack_idx[None, :] * stride_gu_k,
            mask=mask_i[:, None] & mask_p[None, :],
            other=0,
        ).to(tl.int32)
        gate_scale = tl.load(
            gate_up_s_ptr
            + expert_id * stride_gus_e
            + offs_i[:, None] * stride_gus_n
            + group_idx[None, :] * stride_gus_g,
            mask=mask_i[:, None] & mask_p[None, :],
            other=0.0,
        ).to(tl.float32)
        up_scale = tl.load(
            gate_up_s_ptr
            + expert_id * stride_gus_e
            + (INTERMEDIATE + offs_i[:, None]) * stride_gus_n
            + group_idx[None, :] * stride_gus_g,
            mask=mask_i[:, None] & mask_p[None, :],
            other=0.0,
        ).to(tl.float32)
        gate_partial = tl.zeros((BLOCK_I, BLOCK_PACKS), dtype=tl.float32)
        up_partial = tl.zeros((BLOCK_I, BLOCK_PACKS), dtype=tl.float32)
        for nibble in tl.static_range(0, 8):
            h_idx = pack_idx * 8 + nibble
            x_val = tl.load(x_ptr + h_idx, mask=mask_p & (h_idx < H), other=0.0)
            gate_q = (gate_packed >> (nibble * 4)) & 0xF
            up_q = (up_packed >> (nibble * 4)) & 0xF
            gate_partial += (
                x_val[None, :].to(tl.float32)
                * (gate_q.to(tl.float32) - 8.0)
                * gate_scale
            )
            up_partial += (
                x_val[None, :].to(tl.float32) * (up_q.to(tl.float32) - 8.0) * up_scale
            )
        gate_acc += tl.sum(gate_partial, axis=1)
        up_acc += tl.sum(up_partial, axis=1)

    if ACT_KIND == 1:
        # GELU tanh approximation used by Gemma's gelu_pytorch_tanh.
        # Use sigmoid(2x) identity because tanh is not available everywhere.
        x3 = gate_acc * gate_acc * gate_acc
        inner = 0.7978845608028654 * (gate_acc + 0.044715 * x3)
        act = gate_acc / (1.0 + tl.exp(-2.0 * inner))
    else:
        act = gate_acc / (1.0 + tl.exp(-gate_acc))
    out = act * up_acc * route_weight
    tl.store(
        tmp_ptr + pid_k * stride_tmp_k + offs_i * stride_tmp_i,
        out,
        mask=mask_i,
    )


@triton.jit
def _gemma4_moe_down_reduce_m1_kernel(
    topk_ids_ptr,
    down_q_ptr,
    down_s_ptr,
    tmp_ptr,
    out_ptr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_d_e: tl.constexpr,
    stride_d_h: tl.constexpr,
    stride_d_k: tl.constexpr,
    stride_ds_e: tl.constexpr,
    stride_ds_h: tl.constexpr,
    stride_ds_g: tl.constexpr,
    stride_tmp_k: tl.constexpr,
    stride_tmp_i: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_PACKS: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
):
    """
    Decode-only Gemma4 MoE down projection and top-k reduction.

    Layout:
      down_q: [E, H, intermediate/8], eight signed int4 values per packed int32
      down_s: [E, H, intermediate/32], one scale per 32 intermediate columns
      tmp: [top_k, intermediate], fp32 route-weighted gate/up values
      out: [1, H]

    Tiling:
      One program computes BLOCK_H output rows. It loops over top-k experts
      and streams the packed intermediate dimension in BLOCK_PACKS chunks.
    """
    pid_h = tl.program_id(0)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    offs_p = tl.arange(0, BLOCK_PACKS)
    num_packs = INTERMEDIATE // 8
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for topk_pos in tl.static_range(0, TOP_K):
        expert_id = tl.load(topk_ids_ptr + topk_pos).to(tl.int64)
        expert_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for pack_base in range(0, tl.cdiv(num_packs, BLOCK_PACKS)):
            pack_idx = pack_base * BLOCK_PACKS + offs_p
            mask_p = pack_idx < num_packs
            group_idx = pack_idx // 4
            packed = tl.load(
                down_q_ptr
                + expert_id * stride_d_e
                + offs_h[:, None] * stride_d_h
                + pack_idx[None, :] * stride_d_k,
                mask=mask_h[:, None] & mask_p[None, :],
                other=0,
            ).to(tl.int32)
            scale = tl.load(
                down_s_ptr
                + expert_id * stride_ds_e
                + offs_h[:, None] * stride_ds_h
                + group_idx[None, :] * stride_ds_g,
                mask=mask_h[:, None] & mask_p[None, :],
                other=0.0,
            ).to(tl.float32)
            partial = tl.zeros((BLOCK_H, BLOCK_PACKS), dtype=tl.float32)
            for nibble in tl.static_range(0, 8):
                i_idx = pack_idx * 8 + nibble
                hidden = tl.load(
                    tmp_ptr + topk_pos * stride_tmp_k + i_idx * stride_tmp_i,
                    mask=mask_p & (i_idx < INTERMEDIATE),
                    other=0.0,
                )
                q = (packed >> (nibble * 4)) & 0xF
                partial += (
                    hidden[None, :].to(tl.float32) * (q.to(tl.float32) - 8.0) * scale
                )
            expert_acc += tl.sum(partial, axis=1)
        acc += expert_acc

    out = acc.to(tl.bfloat16) if USE_BF16_OUTPUT else acc.to(tl.float16)
    tl.store(out_ptr + offs_h, out, mask=mask_h)


@triton.jit
def _gemma4_moe_gate_up_batched_kernel(
    x_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    gate_up_q_ptr,
    gate_up_s_ptr,
    tmp_ptr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    stride_x_m: tl.constexpr,
    stride_ids_m: tl.constexpr,
    stride_ids_k: tl.constexpr,
    stride_w_m: tl.constexpr,
    stride_w_k: tl.constexpr,
    stride_gu_e: tl.constexpr,
    stride_gu_n: tl.constexpr,
    stride_gu_k: tl.constexpr,
    stride_gus_e: tl.constexpr,
    stride_gus_n: tl.constexpr,
    stride_gus_g: tl.constexpr,
    stride_tmp_m: tl.constexpr,
    stride_tmp_k: tl.constexpr,
    stride_tmp_i: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_PACKS: tl.constexpr,
    ACT_KIND: tl.constexpr,
):
    """
    Batched decode Gemma4 MoE gate/up stage for expert-major symmetric int4.

    Layout:
      x: [M, H]
      topk_ids/topk_weights: [M, top_k]
      gate_up_q: [E, 2 * intermediate, H/8]
      gate_up_s: [E, 2 * intermediate, H/32]
      tmp: [M, top_k, intermediate], fp32 route-weighted activations.

    Tiling:
      One program handles one token, one top-k slot, and BLOCK_I intermediate
      rows. H is streamed as packed int32 groups and unpacked in-register.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_i = tl.program_id(2)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_i = offs_i < INTERMEDIATE
    offs_p = tl.arange(0, BLOCK_PACKS)

    expert_id = tl.load(topk_ids_ptr + pid_m * stride_ids_m + pid_k * stride_ids_k).to(
        tl.int64
    )
    route_weight = tl.load(
        topk_weights_ptr + pid_m * stride_w_m + pid_k * stride_w_k
    ).to(tl.float32)
    num_packs = H // 8
    gate_acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_I,), dtype=tl.float32)

    for pack_base in range(0, tl.cdiv(num_packs, BLOCK_PACKS)):
        pack_idx = pack_base * BLOCK_PACKS + offs_p
        mask_p = pack_idx < num_packs
        group_idx = pack_idx // 4
        gate_packed = tl.load(
            gate_up_q_ptr
            + expert_id * stride_gu_e
            + offs_i[:, None] * stride_gu_n
            + pack_idx[None, :] * stride_gu_k,
            mask=mask_i[:, None] & mask_p[None, :],
            other=0,
        ).to(tl.int32)
        up_packed = tl.load(
            gate_up_q_ptr
            + expert_id * stride_gu_e
            + (INTERMEDIATE + offs_i[:, None]) * stride_gu_n
            + pack_idx[None, :] * stride_gu_k,
            mask=mask_i[:, None] & mask_p[None, :],
            other=0,
        ).to(tl.int32)
        gate_scale = tl.load(
            gate_up_s_ptr
            + expert_id * stride_gus_e
            + offs_i[:, None] * stride_gus_n
            + group_idx[None, :] * stride_gus_g,
            mask=mask_i[:, None] & mask_p[None, :],
            other=0.0,
        ).to(tl.float32)
        up_scale = tl.load(
            gate_up_s_ptr
            + expert_id * stride_gus_e
            + (INTERMEDIATE + offs_i[:, None]) * stride_gus_n
            + group_idx[None, :] * stride_gus_g,
            mask=mask_i[:, None] & mask_p[None, :],
            other=0.0,
        ).to(tl.float32)
        gate_partial = tl.zeros((BLOCK_I, BLOCK_PACKS), dtype=tl.float32)
        up_partial = tl.zeros((BLOCK_I, BLOCK_PACKS), dtype=tl.float32)
        for nibble in tl.static_range(0, 8):
            h_idx = pack_idx * 8 + nibble
            x_val = tl.load(
                x_ptr + pid_m * stride_x_m + h_idx,
                mask=mask_p & (h_idx < H),
                other=0.0,
            )
            gate_q = (gate_packed >> (nibble * 4)) & 0xF
            up_q = (up_packed >> (nibble * 4)) & 0xF
            gate_partial += (
                x_val[None, :].to(tl.float32)
                * (gate_q.to(tl.float32) - 8.0)
                * gate_scale
            )
            up_partial += (
                x_val[None, :].to(tl.float32) * (up_q.to(tl.float32) - 8.0) * up_scale
            )
        gate_acc += tl.sum(gate_partial, axis=1)
        up_acc += tl.sum(up_partial, axis=1)

    if ACT_KIND == 1:
        # GELU tanh approximation used by Gemma's gelu_pytorch_tanh.
        # Use sigmoid(2x) identity because tanh is not available everywhere.
        x3 = gate_acc * gate_acc * gate_acc
        inner = 0.7978845608028654 * (gate_acc + 0.044715 * x3)
        act = gate_acc / (1.0 + tl.exp(-2.0 * inner))
    else:
        act = gate_acc / (1.0 + tl.exp(-gate_acc))
    out = act * up_acc * route_weight
    tl.store(
        tmp_ptr + pid_m * stride_tmp_m + pid_k * stride_tmp_k + offs_i * stride_tmp_i,
        out,
        mask=mask_i,
    )


@triton.jit
def _gemma4_moe_down_reduce_batched_kernel(
    topk_ids_ptr,
    down_q_ptr,
    down_s_ptr,
    tmp_ptr,
    out_ptr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_ids_m: tl.constexpr,
    stride_ids_k: tl.constexpr,
    stride_d_e: tl.constexpr,
    stride_d_h: tl.constexpr,
    stride_d_k: tl.constexpr,
    stride_ds_e: tl.constexpr,
    stride_ds_h: tl.constexpr,
    stride_ds_g: tl.constexpr,
    stride_tmp_m: tl.constexpr,
    stride_tmp_k: tl.constexpr,
    stride_tmp_i: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_PACKS: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
):
    """
    Batched decode Gemma4 MoE down projection and per-token top-k reduction.

    Layout:
      topk_ids: [M, top_k]
      down_q: [E, H, intermediate/8]
      down_s: [E, H, intermediate/32]
      tmp: [M, top_k, intermediate], fp32 route-weighted activations
      out: [M, H]

    Tiling:
      One program computes one token and BLOCK_H output rows. It loops over
      top-k experts and streams packed intermediate columns for each expert.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    offs_p = tl.arange(0, BLOCK_PACKS)
    num_packs = INTERMEDIATE // 8
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for topk_pos in tl.static_range(0, TOP_K):
        expert_id = tl.load(
            topk_ids_ptr + pid_m * stride_ids_m + topk_pos * stride_ids_k
        ).to(tl.int64)
        expert_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for pack_base in range(0, tl.cdiv(num_packs, BLOCK_PACKS)):
            pack_idx = pack_base * BLOCK_PACKS + offs_p
            mask_p = pack_idx < num_packs
            group_idx = pack_idx // 4
            packed = tl.load(
                down_q_ptr
                + expert_id * stride_d_e
                + offs_h[:, None] * stride_d_h
                + pack_idx[None, :] * stride_d_k,
                mask=mask_h[:, None] & mask_p[None, :],
                other=0,
            ).to(tl.int32)
            scale = tl.load(
                down_s_ptr
                + expert_id * stride_ds_e
                + offs_h[:, None] * stride_ds_h
                + group_idx[None, :] * stride_ds_g,
                mask=mask_h[:, None] & mask_p[None, :],
                other=0.0,
            ).to(tl.float32)
            partial = tl.zeros((BLOCK_H, BLOCK_PACKS), dtype=tl.float32)
            for nibble in tl.static_range(0, 8):
                i_idx = pack_idx * 8 + nibble
                hidden = tl.load(
                    tmp_ptr
                    + pid_m * stride_tmp_m
                    + topk_pos * stride_tmp_k
                    + i_idx * stride_tmp_i,
                    mask=mask_p & (i_idx < INTERMEDIATE),
                    other=0.0,
                )
                q = (packed >> (nibble * 4)) & 0xF
                partial += (
                    hidden[None, :].to(tl.float32) * (q.to(tl.float32) - 8.0) * scale
                )
            expert_acc += tl.sum(partial, axis=1)
        acc += expert_acc

    out = acc.to(tl.bfloat16) if USE_BF16_OUTPUT else acc.to(tl.float16)
    tl.store(
        out_ptr + pid_m * stride_out_m + offs_h * stride_out_h,
        out,
        mask=mask_h,
    )


@triton.jit
def _gemma4_moe_down_reduce_batched_token_major_kernel(
    topk_ids_ptr,
    down_q_ptr,
    down_s_ptr,
    tmp_ptr,
    out_ptr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_ids_m: tl.constexpr,
    stride_ids_k: tl.constexpr,
    stride_d_e: tl.constexpr,
    stride_d_h: tl.constexpr,
    stride_d_k: tl.constexpr,
    stride_ds_e: tl.constexpr,
    stride_ds_h: tl.constexpr,
    stride_ds_g: tl.constexpr,
    stride_tmp_m: tl.constexpr,
    stride_tmp_k: tl.constexpr,
    stride_tmp_i: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_PACKS: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
):
    """
    Token-major batched decode down projection for symmetric int4 MoE.

    Layout:
      topk_ids: [M, top_k]
      down_q: [E, H, intermediate/8]
      down_s: [E, H, intermediate/32]
      tmp: [M, top_k, intermediate], fp32 route-weighted activations
      out: [M, H]

    Tiling:
      Grid is launched as [H tiles, M] so adjacent programs walk all H tiles
      for one token before advancing to the next token. This favors reuse of a
      token's tmp rows in cache while preserving the same per-program math as
      the conservative batched down/reduce kernel.
    """
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    offs_p = tl.arange(0, BLOCK_PACKS)
    num_packs = INTERMEDIATE // 8
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for topk_pos in tl.static_range(0, TOP_K):
        expert_id = tl.load(
            topk_ids_ptr + pid_m * stride_ids_m + topk_pos * stride_ids_k
        ).to(tl.int64)
        expert_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for pack_base in range(0, tl.cdiv(num_packs, BLOCK_PACKS)):
            pack_idx = pack_base * BLOCK_PACKS + offs_p
            mask_p = pack_idx < num_packs
            group_idx = pack_idx // 4
            packed = tl.load(
                down_q_ptr
                + expert_id * stride_d_e
                + offs_h[:, None] * stride_d_h
                + pack_idx[None, :] * stride_d_k,
                mask=mask_h[:, None] & mask_p[None, :],
                other=0,
            ).to(tl.int32)
            scale = tl.load(
                down_s_ptr
                + expert_id * stride_ds_e
                + offs_h[:, None] * stride_ds_h
                + group_idx[None, :] * stride_ds_g,
                mask=mask_h[:, None] & mask_p[None, :],
                other=0.0,
            ).to(tl.float32)
            partial = tl.zeros((BLOCK_H, BLOCK_PACKS), dtype=tl.float32)
            for nibble in tl.static_range(0, 8):
                i_idx = pack_idx * 8 + nibble
                hidden = tl.load(
                    tmp_ptr
                    + pid_m * stride_tmp_m
                    + topk_pos * stride_tmp_k
                    + i_idx * stride_tmp_i,
                    mask=mask_p & (i_idx < INTERMEDIATE),
                    other=0.0,
                )
                q = (packed >> (nibble * 4)) & 0xF
                partial += (
                    hidden[None, :].to(tl.float32) * (q.to(tl.float32) - 8.0) * scale
                )
            expert_acc += tl.sum(partial, axis=1)
        acc += expert_acc

    out = acc.to(tl.bfloat16) if USE_BF16_OUTPUT else acc.to(tl.float16)
    tl.store(
        out_ptr + pid_m * stride_out_m + offs_h * stride_out_h,
        out,
        mask=mask_h,
    )


@triton.jit
def _gemma4_moe_gate_up_batched_chunk_kernel(
    x_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    gate_up_q_ptr,
    gate_up_s_ptr,
    tmp_ptr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    I_OFFSET: tl.constexpr,
    TMP_OFFSET: tl.constexpr,
    stride_x_m: tl.constexpr,
    stride_ids_m: tl.constexpr,
    stride_ids_k: tl.constexpr,
    stride_w_m: tl.constexpr,
    stride_w_k: tl.constexpr,
    stride_gu_e: tl.constexpr,
    stride_gu_n: tl.constexpr,
    stride_gu_k: tl.constexpr,
    stride_gus_e: tl.constexpr,
    stride_gus_n: tl.constexpr,
    stride_gus_g: tl.constexpr,
    stride_tmp_m: tl.constexpr,
    stride_tmp_k: tl.constexpr,
    stride_tmp_i: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_PACKS: tl.constexpr,
    ACT_KIND: tl.constexpr,
):
    """
    Batched decode gate/up for one intermediate chunk.

    Layout:
      x: [M, H]
      topk_ids/topk_weights: [M, top_k]
      gate_up_q: [E, 2 * intermediate, H/8]
      gate_up_s: [E, 2 * intermediate, H/32]
      tmp: [M, top_k, >= TMP_OFFSET + BLOCK_I], fp32 route-weighted chunk
        activations.

    Tiling:
      One program handles one token and one top-k slot for a contiguous
      BLOCK_I slice of the intermediate dimension. H is streamed in packed
      int32 groups so only this chunk, not the full hidden buffer, is written
      to global memory.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_i = tl.arange(0, BLOCK_I)
    i_idx = I_OFFSET + offs_i
    mask_i = i_idx < INTERMEDIATE
    offs_p = tl.arange(0, BLOCK_PACKS)

    expert_id = tl.load(topk_ids_ptr + pid_m * stride_ids_m + pid_k * stride_ids_k).to(
        tl.int64
    )
    route_weight = tl.load(
        topk_weights_ptr + pid_m * stride_w_m + pid_k * stride_w_k
    ).to(tl.float32)
    num_packs = H // 8
    gate_acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_I,), dtype=tl.float32)

    for pack_base in range(0, tl.cdiv(num_packs, BLOCK_PACKS)):
        pack_idx = pack_base * BLOCK_PACKS + offs_p
        mask_p = pack_idx < num_packs
        group_idx = pack_idx // 4
        gate_packed = tl.load(
            gate_up_q_ptr
            + expert_id * stride_gu_e
            + i_idx[:, None] * stride_gu_n
            + pack_idx[None, :] * stride_gu_k,
            mask=mask_i[:, None] & mask_p[None, :],
            other=0,
        ).to(tl.int32)
        up_packed = tl.load(
            gate_up_q_ptr
            + expert_id * stride_gu_e
            + (INTERMEDIATE + i_idx[:, None]) * stride_gu_n
            + pack_idx[None, :] * stride_gu_k,
            mask=mask_i[:, None] & mask_p[None, :],
            other=0,
        ).to(tl.int32)
        gate_scale = tl.load(
            gate_up_s_ptr
            + expert_id * stride_gus_e
            + i_idx[:, None] * stride_gus_n
            + group_idx[None, :] * stride_gus_g,
            mask=mask_i[:, None] & mask_p[None, :],
            other=0.0,
        ).to(tl.float32)
        up_scale = tl.load(
            gate_up_s_ptr
            + expert_id * stride_gus_e
            + (INTERMEDIATE + i_idx[:, None]) * stride_gus_n
            + group_idx[None, :] * stride_gus_g,
            mask=mask_i[:, None] & mask_p[None, :],
            other=0.0,
        ).to(tl.float32)
        gate_partial = tl.zeros((BLOCK_I, BLOCK_PACKS), dtype=tl.float32)
        up_partial = tl.zeros((BLOCK_I, BLOCK_PACKS), dtype=tl.float32)
        for nibble in tl.static_range(0, 8):
            h_idx = pack_idx * 8 + nibble
            x_val = tl.load(
                x_ptr + pid_m * stride_x_m + h_idx,
                mask=mask_p & (h_idx < H),
                other=0.0,
            )
            gate_q = (gate_packed >> (nibble * 4)) & 0xF
            up_q = (up_packed >> (nibble * 4)) & 0xF
            gate_partial += (
                x_val[None, :].to(tl.float32)
                * (gate_q.to(tl.float32) - 8.0)
                * gate_scale
            )
            up_partial += (
                x_val[None, :].to(tl.float32) * (up_q.to(tl.float32) - 8.0) * up_scale
            )
        gate_acc += tl.sum(gate_partial, axis=1)
        up_acc += tl.sum(up_partial, axis=1)

    if ACT_KIND == 1:
        # GELU tanh approximation used by Gemma's gelu_pytorch_tanh.
        # Use sigmoid(2x) identity because tanh is not available everywhere.
        x3 = gate_acc * gate_acc * gate_acc
        inner = 0.7978845608028654 * (gate_acc + 0.044715 * x3)
        act = gate_acc / (1.0 + tl.exp(-2.0 * inner))
    else:
        act = gate_acc / (1.0 + tl.exp(-gate_acc))
    out = act * up_acc * route_weight
    tl.store(
        tmp_ptr
        + pid_m * stride_tmp_m
        + pid_k * stride_tmp_k
        + (TMP_OFFSET + offs_i) * stride_tmp_i,
        out,
        mask=mask_i,
    )


@triton.jit
def _gemma4_moe_down_reduce_batched_chunk_kernel(
    topk_ids_ptr,
    down_q_ptr,
    down_s_ptr,
    tmp_ptr,
    acc_ptr,
    out_ptr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TOP_K: tl.constexpr,
    I_OFFSET: tl.constexpr,
    TMP_OFFSET: tl.constexpr,
    stride_ids_m: tl.constexpr,
    stride_ids_k: tl.constexpr,
    stride_d_e: tl.constexpr,
    stride_d_h: tl.constexpr,
    stride_d_k: tl.constexpr,
    stride_ds_e: tl.constexpr,
    stride_ds_h: tl.constexpr,
    stride_ds_g: tl.constexpr,
    stride_tmp_m: tl.constexpr,
    stride_tmp_k: tl.constexpr,
    stride_tmp_i: tl.constexpr,
    stride_acc_m: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_PACKS: tl.constexpr,
    IS_FIRST_CHUNK: tl.constexpr,
    IS_LAST_CHUNK: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
):
    """
    Batched decode down/reduce for one intermediate chunk.

    Layout:
      topk_ids: [M, top_k]
      down_q: [E, H, intermediate/8]
      down_s: [E, H, intermediate/32]
      tmp: [M, top_k, BLOCK_I], fp32 route-weighted chunk activations
      acc: [M, H], fp32 output accumulator
      out: [M, H], final dtype output.

    Tiling:
      Grid is [H tiles, M]. Each program consumes a BLOCK_I intermediate
      chunk for all top-k experts. The first chunk initializes the fp32
      accumulator without a separate zero-fill launch; the last chunk writes
      the final cast output directly, removing a separate accumulator cast
      kernel.
    """
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    offs_p = tl.arange(0, BLOCK_PACKS)
    num_chunk_packs = BLOCK_I // 8
    chunk_valid_i = tl.minimum(BLOCK_I, INTERMEDIATE - I_OFFSET)
    chunk_packs = tl.cdiv(chunk_valid_i, 8)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for topk_pos in tl.static_range(0, TOP_K):
        expert_id = tl.load(
            topk_ids_ptr + pid_m * stride_ids_m + topk_pos * stride_ids_k
        ).to(tl.int64)
        expert_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for pack_base in range(0, tl.cdiv(num_chunk_packs, BLOCK_PACKS)):
            pack_idx = pack_base * BLOCK_PACKS + offs_p
            mask_p = pack_idx < chunk_packs
            global_pack_idx = (I_OFFSET // 8) + pack_idx
            group_idx = (I_OFFSET // 32) + (pack_idx // 4)
            packed = tl.load(
                down_q_ptr
                + expert_id * stride_d_e
                + offs_h[:, None] * stride_d_h
                + global_pack_idx[None, :] * stride_d_k,
                mask=mask_h[:, None] & mask_p[None, :],
                other=0,
            ).to(tl.int32)
            scale = tl.load(
                down_s_ptr
                + expert_id * stride_ds_e
                + offs_h[:, None] * stride_ds_h
                + group_idx[None, :] * stride_ds_g,
                mask=mask_h[:, None] & mask_p[None, :],
                other=0.0,
            ).to(tl.float32)
            partial = tl.zeros((BLOCK_H, BLOCK_PACKS), dtype=tl.float32)
            for nibble in tl.static_range(0, 8):
                i_local = pack_idx * 8 + nibble
                hidden = tl.load(
                    tmp_ptr
                    + pid_m * stride_tmp_m
                    + topk_pos * stride_tmp_k
                    + (TMP_OFFSET + i_local) * stride_tmp_i,
                    mask=mask_p & (i_local < chunk_valid_i),
                    other=0.0,
                )
                q = (packed >> (nibble * 4)) & 0xF
                partial += (
                    hidden[None, :].to(tl.float32) * (q.to(tl.float32) - 8.0) * scale
                )
            expert_acc += tl.sum(partial, axis=1)
        acc += expert_acc

    if IS_FIRST_CHUNK:
        total = acc
    else:
        old = tl.load(
            acc_ptr + pid_m * stride_acc_m + offs_h * stride_acc_h,
            mask=mask_h,
            other=0.0,
        )
        total = old + acc

    if IS_LAST_CHUNK:
        out = total.to(tl.bfloat16) if USE_BF16_OUTPUT else total.to(tl.float16)
        tl.store(
            out_ptr + pid_m * stride_out_m + offs_h * stride_out_h,
            out,
            mask=mask_h,
        )
    else:
        tl.store(
            acc_ptr + pid_m * stride_acc_m + offs_h * stride_acc_h,
            total,
            mask=mask_h,
        )


@triton.jit
def _gemma4_moe_down_reduce_batched_grouped_chunk_kernel(
    topk_ids_ptr,
    down_q_ptr,
    down_s_ptr,
    tmp_ptr,
    acc_ptr,
    out_ptr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TOP_K: tl.constexpr,
    I_OFFSET: tl.constexpr,
    stride_ids_m: tl.constexpr,
    stride_ids_k: tl.constexpr,
    stride_d_e: tl.constexpr,
    stride_d_h: tl.constexpr,
    stride_d_k: tl.constexpr,
    stride_ds_e: tl.constexpr,
    stride_ds_h: tl.constexpr,
    stride_ds_g: tl.constexpr,
    stride_tmp_m: tl.constexpr,
    stride_tmp_i: tl.constexpr,
    stride_tmp_k: tl.constexpr,
    stride_acc_m: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_PACKS: tl.constexpr,
    IS_FIRST_CHUNK: tl.constexpr,
    IS_LAST_CHUNK: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
):
    """
    Batched decode grouped tmp down/reduce for one intermediate chunk.

    Layout:
      topk_ids: [M, top_k]
      down_q: [E, H, intermediate/8]
      down_s: [E, H, intermediate/32]
      tmp: [M, BLOCK_I, top_k], compact route-weighted activations with
        top-k contiguous for each intermediate element
      acc: [M, H], fp32 output accumulator
      out: [M, H], final dtype output.

    Tiling:
      Grid is [H tiles, M]. Each program streams one intermediate chunk and
      visits all top-k experts for each packed intermediate tile. This layout
      makes the top-k group adjacent in tmp while keeping gate/up computed
      once per token, top-k slot, and intermediate chunk.
    """
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    offs_p = tl.arange(0, BLOCK_PACKS)
    num_chunk_packs = BLOCK_I // 8
    chunk_valid_i = tl.minimum(BLOCK_I, INTERMEDIATE - I_OFFSET)
    chunk_packs = tl.cdiv(chunk_valid_i, 8)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for pack_base in range(0, tl.cdiv(num_chunk_packs, BLOCK_PACKS)):
        pack_idx = pack_base * BLOCK_PACKS + offs_p
        mask_p = pack_idx < chunk_packs
        global_pack_idx = (I_OFFSET // 8) + pack_idx
        group_idx = (I_OFFSET // 32) + (pack_idx // 4)
        i_local = pack_idx * 8

        for topk_pos in tl.static_range(0, TOP_K):
            expert_id = tl.load(
                topk_ids_ptr + pid_m * stride_ids_m + topk_pos * stride_ids_k
            ).to(tl.int64)
            packed = tl.load(
                down_q_ptr
                + expert_id * stride_d_e
                + offs_h[:, None] * stride_d_h
                + global_pack_idx[None, :] * stride_d_k,
                mask=mask_h[:, None] & mask_p[None, :],
                other=0,
            ).to(tl.int32)
            scale = tl.load(
                down_s_ptr
                + expert_id * stride_ds_e
                + offs_h[:, None] * stride_ds_h
                + group_idx[None, :] * stride_ds_g,
                mask=mask_h[:, None] & mask_p[None, :],
                other=0.0,
            ).to(tl.float32)
            partial = tl.zeros((BLOCK_H, BLOCK_PACKS), dtype=tl.float32)
            for nibble in tl.static_range(0, 8):
                i_offset = i_local + nibble
                hidden = tl.load(
                    tmp_ptr
                    + pid_m * stride_tmp_m
                    + i_offset * stride_tmp_i
                    + topk_pos * stride_tmp_k,
                    mask=mask_p & (i_offset < chunk_valid_i),
                    other=0.0,
                )
                q = (packed >> (nibble * 4)) & 0xF
                partial += (
                    hidden[None, :].to(tl.float32) * (q.to(tl.float32) - 8.0) * scale
                )
            acc += tl.sum(partial, axis=1)

    if IS_FIRST_CHUNK:
        total = acc
    else:
        old = tl.load(
            acc_ptr + pid_m * stride_acc_m + offs_h * stride_acc_h,
            mask=mask_h,
            other=0.0,
        )
        total = old + acc

    if IS_LAST_CHUNK:
        out = total.to(tl.bfloat16) if USE_BF16_OUTPUT else total.to(tl.float16)
        tl.store(
            out_ptr + pid_m * stride_out_m + offs_h * stride_out_h,
            out,
            mask=mask_h,
        )
    else:
        tl.store(
            acc_ptr + pid_m * stride_acc_m + offs_h * stride_acc_h,
            total,
            mask=mask_h,
        )


@triton.jit
def _gemma4_moe_down_reduce_batched_chunk_pair_kernel(
    topk_ids_ptr,
    down_q_ptr,
    down_s_ptr,
    tmp_ptr,
    acc_ptr,
    out_ptr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TOP_K: tl.constexpr,
    I_OFFSET: tl.constexpr,
    stride_ids_m: tl.constexpr,
    stride_ids_k: tl.constexpr,
    stride_d_e: tl.constexpr,
    stride_d_h: tl.constexpr,
    stride_d_k: tl.constexpr,
    stride_ds_e: tl.constexpr,
    stride_ds_h: tl.constexpr,
    stride_ds_g: tl.constexpr,
    stride_tmp_m: tl.constexpr,
    stride_tmp_k: tl.constexpr,
    stride_tmp_i: tl.constexpr,
    stride_acc_m: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_PACKS: tl.constexpr,
    IS_FIRST_PAIR: tl.constexpr,
    IS_LAST_PAIR: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
):
    """
    Batched decode down/reduce for two intermediate chunks in one launch.

    Layout:
      topk_ids: [M, top_k]
      down_q: [E, H, intermediate/8]
      down_s: [E, H, intermediate/32]
      tmp: [M, top_k, 2 * BLOCK_I], fp32 route-weighted activations
      acc: [M, H], fp32 output accumulator
      out: [M, H], final dtype output.

    Tiling:
      Grid is [H tiles, M]. Each program consumes up to two adjacent BLOCK_I
      intermediate chunks for all top-k experts. This cuts one down/reduce
      launch per pair while reusing the gate/up tmp written by the paired
      chunk producer, so gate/up is not recomputed per output tile.
    """
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    offs_p = tl.arange(0, BLOCK_PACKS)
    num_chunk_packs = BLOCK_I // 8
    pair_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for topk_pos in tl.static_range(0, TOP_K):
        expert_id = tl.load(
            topk_ids_ptr + pid_m * stride_ids_m + topk_pos * stride_ids_k
        ).to(tl.int64)
        expert_pair_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for chunk_pos in tl.static_range(0, 2):
            chunk_i_offset = I_OFFSET + chunk_pos * BLOCK_I
            if chunk_i_offset < INTERMEDIATE:
                chunk_valid_i = tl.minimum(BLOCK_I, INTERMEDIATE - chunk_i_offset)
                chunk_packs = tl.cdiv(chunk_valid_i, 8)
                chunk_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
                for pack_base in range(0, tl.cdiv(num_chunk_packs, BLOCK_PACKS)):
                    pack_idx = pack_base * BLOCK_PACKS + offs_p
                    mask_p = pack_idx < chunk_packs
                    global_pack_idx = (chunk_i_offset // 8) + pack_idx
                    group_idx = (chunk_i_offset // 32) + (pack_idx // 4)
                    packed = tl.load(
                        down_q_ptr
                        + expert_id * stride_d_e
                        + offs_h[:, None] * stride_d_h
                        + global_pack_idx[None, :] * stride_d_k,
                        mask=mask_h[:, None] & mask_p[None, :],
                        other=0,
                    ).to(tl.int32)
                    scale = tl.load(
                        down_s_ptr
                        + expert_id * stride_ds_e
                        + offs_h[:, None] * stride_ds_h
                        + group_idx[None, :] * stride_ds_g,
                        mask=mask_h[:, None] & mask_p[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    partial = tl.zeros((BLOCK_H, BLOCK_PACKS), dtype=tl.float32)
                    for nibble in tl.static_range(0, 8):
                        i_local = pack_idx * 8 + nibble
                        tmp_i = chunk_pos * BLOCK_I + i_local
                        hidden = tl.load(
                            tmp_ptr
                            + pid_m * stride_tmp_m
                            + topk_pos * stride_tmp_k
                            + tmp_i * stride_tmp_i,
                            mask=mask_p & (i_local < chunk_valid_i),
                            other=0.0,
                        )
                        q = (packed >> (nibble * 4)) & 0xF
                        partial += (
                            hidden[None, :].to(tl.float32)
                            * (q.to(tl.float32) - 8.0)
                            * scale
                        )
                    chunk_acc += tl.sum(partial, axis=1)
                expert_pair_acc += chunk_acc
        pair_acc += expert_pair_acc

    if IS_FIRST_PAIR:
        total = pair_acc
    else:
        old = tl.load(
            acc_ptr + pid_m * stride_acc_m + offs_h * stride_acc_h,
            mask=mask_h,
            other=0.0,
        )
        total = old + pair_acc

    if IS_LAST_PAIR:
        out = total.to(tl.bfloat16) if USE_BF16_OUTPUT else total.to(tl.float16)
        tl.store(
            out_ptr + pid_m * stride_out_m + offs_h * stride_out_h,
            out,
            mask=mask_h,
        )
    else:
        tl.store(
            acc_ptr + pid_m * stride_acc_m + offs_h * stride_acc_h,
            total,
            mask=mask_h,
        )


@triton.jit
def _gemma4_moe_single_fused_m1_kernel(
    x_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    gate_up_q_ptr,
    gate_up_s_ptr,
    down_q_ptr,
    down_s_ptr,
    out_ptr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_gu_e: tl.constexpr,
    stride_gu_n: tl.constexpr,
    stride_gu_k: tl.constexpr,
    stride_gus_e: tl.constexpr,
    stride_gus_n: tl.constexpr,
    stride_gus_g: tl.constexpr,
    stride_d_e: tl.constexpr,
    stride_d_h: tl.constexpr,
    stride_d_k: tl.constexpr,
    stride_ds_e: tl.constexpr,
    stride_ds_h: tl.constexpr,
    stride_ds_g: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_PACKS_H: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
    ACT_KIND: tl.constexpr,
):
    """
    Single-launch Gemma4 MoE decode kernel for expert-major symmetric int4.

    Layout:
      x: [1, H]
      topk_ids/topk_weights: [1, top_k]
      gate_up_q: [E, 2 * intermediate, H/8]
      gate_up_s: [E, 2 * intermediate, H/32]
      down_q: [E, H, intermediate/8]
      down_s: [E, H, intermediate/32]
      out: [1, H]

    Tiling:
      One program computes BLOCK_H output rows. For each selected expert it
      streams intermediate rows in BLOCK_I chunks. Each chunk recomputes the
      gate/up dot products for that BLOCK_H tile and immediately applies the
      matching packed down-projection nibbles. This removes global tmp traffic
      and the second launch, at the cost of recomputing gate/up per H tile.
    """
    pid_h = tl.program_id(0)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    offs_i = tl.arange(0, BLOCK_I)
    offs_hp = tl.arange(0, BLOCK_PACKS_H)
    num_h_packs = H // 8
    acc_h = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for topk_pos in tl.static_range(0, TOP_K):
        expert_id = tl.load(topk_ids_ptr + topk_pos).to(tl.int64)
        route_weight = tl.load(topk_weights_ptr + topk_pos).to(tl.float32)
        for i_base in range(0, tl.cdiv(INTERMEDIATE, BLOCK_I)):
            i_idx = i_base * BLOCK_I + offs_i
            mask_i = i_idx < INTERMEDIATE
            gate_acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
            up_acc = tl.zeros((BLOCK_I,), dtype=tl.float32)

            for h_pack_base in range(0, tl.cdiv(num_h_packs, BLOCK_PACKS_H)):
                h_pack_idx = h_pack_base * BLOCK_PACKS_H + offs_hp
                mask_hp = h_pack_idx < num_h_packs
                h_group_idx = h_pack_idx // 4
                gate_packed = tl.load(
                    gate_up_q_ptr
                    + expert_id * stride_gu_e
                    + i_idx[:, None] * stride_gu_n
                    + h_pack_idx[None, :] * stride_gu_k,
                    mask=mask_i[:, None] & mask_hp[None, :],
                    other=0,
                ).to(tl.int32)
                up_packed = tl.load(
                    gate_up_q_ptr
                    + expert_id * stride_gu_e
                    + (INTERMEDIATE + i_idx[:, None]) * stride_gu_n
                    + h_pack_idx[None, :] * stride_gu_k,
                    mask=mask_i[:, None] & mask_hp[None, :],
                    other=0,
                ).to(tl.int32)
                gate_scale = tl.load(
                    gate_up_s_ptr
                    + expert_id * stride_gus_e
                    + i_idx[:, None] * stride_gus_n
                    + h_group_idx[None, :] * stride_gus_g,
                    mask=mask_i[:, None] & mask_hp[None, :],
                    other=0.0,
                ).to(tl.float32)
                up_scale = tl.load(
                    gate_up_s_ptr
                    + expert_id * stride_gus_e
                    + (INTERMEDIATE + i_idx[:, None]) * stride_gus_n
                    + h_group_idx[None, :] * stride_gus_g,
                    mask=mask_i[:, None] & mask_hp[None, :],
                    other=0.0,
                ).to(tl.float32)
                gate_partial = tl.zeros((BLOCK_I, BLOCK_PACKS_H), dtype=tl.float32)
                up_partial = tl.zeros((BLOCK_I, BLOCK_PACKS_H), dtype=tl.float32)
                for nibble_h in tl.static_range(0, 8):
                    h_idx = h_pack_idx * 8 + nibble_h
                    x_val = tl.load(
                        x_ptr + h_idx,
                        mask=mask_hp & (h_idx < H),
                        other=0.0,
                    )
                    gate_q = (gate_packed >> (nibble_h * 4)) & 0xF
                    up_q = (up_packed >> (nibble_h * 4)) & 0xF
                    gate_partial += (
                        x_val[None, :].to(tl.float32)
                        * (gate_q.to(tl.float32) - 8.0)
                        * gate_scale
                    )
                    up_partial += (
                        x_val[None, :].to(tl.float32)
                        * (up_q.to(tl.float32) - 8.0)
                        * up_scale
                    )
                gate_acc += tl.sum(gate_partial, axis=1)
                up_acc += tl.sum(up_partial, axis=1)

            if ACT_KIND == 1:
                # GELU tanh approximation used by Gemma's gelu_pytorch_tanh.
                # Use sigmoid(2x) identity because tanh is not available everywhere.
                x3 = gate_acc * gate_acc * gate_acc
                inner = 0.7978845608028654 * (gate_acc + 0.044715 * x3)
                act = gate_acc / (1.0 + tl.exp(-2.0 * inner))
            else:
                act = gate_acc / (1.0 + tl.exp(-gate_acc))
            hidden = act * up_acc * route_weight
            down_pack_idx = i_idx // 8
            down_group_idx = i_idx // 32
            down_nibble = i_idx % 8
            down_packed = tl.load(
                down_q_ptr
                + expert_id * stride_d_e
                + offs_h[:, None] * stride_d_h
                + down_pack_idx[None, :] * stride_d_k,
                mask=mask_h[:, None] & mask_i[None, :],
                other=0,
            ).to(tl.int32)
            down_scale = tl.load(
                down_s_ptr
                + expert_id * stride_ds_e
                + offs_h[:, None] * stride_ds_h
                + down_group_idx[None, :] * stride_ds_g,
                mask=mask_h[:, None] & mask_i[None, :],
                other=0.0,
            ).to(tl.float32)
            down_q = (down_packed >> (down_nibble[None, :] * 4)) & 0xF
            acc_h += tl.sum(
                hidden[None, :] * (down_q.to(tl.float32) - 8.0) * down_scale,
                axis=1,
            )

    out = acc_h.to(tl.bfloat16) if USE_BF16_OUTPUT else acc_h.to(tl.float16)
    tl.store(out_ptr + offs_h, out, mask=mask_h)


@triton.jit
def _gemma4_moe_prefill_grouped_fused_kernel(
    x_ptr,
    route_token_ptr,
    route_expert_ptr,
    route_weight_ptr,
    gate_up_q_ptr,
    gate_up_s_ptr,
    down_q_ptr,
    down_s_ptr,
    acc_ptr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    stride_x_m: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_gu_e: tl.constexpr,
    stride_gu_n: tl.constexpr,
    stride_gu_k: tl.constexpr,
    stride_gus_e: tl.constexpr,
    stride_gus_n: tl.constexpr,
    stride_gus_g: tl.constexpr,
    stride_d_e: tl.constexpr,
    stride_d_h: tl.constexpr,
    stride_d_k: tl.constexpr,
    stride_ds_e: tl.constexpr,
    stride_ds_h: tl.constexpr,
    stride_ds_g: tl.constexpr,
    stride_acc_m: tl.constexpr,
    stride_acc_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_PACKS_H: tl.constexpr,
    ACT_KIND: tl.constexpr,
):
    """
    Expert-grouped prefill fused MoE kernel for expert-major symmetric int4.

    Layout:
      x: [M, H]
      route_token/route_expert/route_weight: [M * top_k], sorted by expert id
      gate_up_q: [E, 2 * intermediate, H/8]
      gate_up_s: [E, 2 * intermediate, H/32]
      down_q: [E, H, intermediate/8]
      down_s: [E, H, intermediate/32]
      acc: [M, H], fp32 accumulator zeroed by the caller.

    Tiling:
      Grid is [H tiles, sorted routes]. Each program computes one routed token
      contribution for BLOCK_H output rows. It streams gate/up int4 weights
      for BLOCK_I intermediate rows, applies activation in registers, consumes
      the matching packed down rows, then atomically accumulates into acc. The
      sorted route layout groups selected experts without materializing dense
      expert weights or writing a full [M, top_k, intermediate] tmp tensor.
    """
    pid_h = tl.program_id(0)
    pid_r = tl.program_id(1)
    token_id = tl.load(route_token_ptr + pid_r).to(tl.int64)
    expert_id = tl.load(route_expert_ptr + pid_r).to(tl.int64)
    route_weight = tl.load(route_weight_ptr + pid_r).to(tl.float32)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    offs_i = tl.arange(0, BLOCK_I)
    offs_hp = tl.arange(0, BLOCK_PACKS_H)
    num_h_packs = H // 8
    acc_h = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for i_base in range(0, tl.cdiv(INTERMEDIATE, BLOCK_I)):
        i_idx = i_base * BLOCK_I + offs_i
        mask_i = i_idx < INTERMEDIATE
        gate_acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_I,), dtype=tl.float32)

        for h_pack_base in range(0, tl.cdiv(num_h_packs, BLOCK_PACKS_H)):
            h_pack_idx = h_pack_base * BLOCK_PACKS_H + offs_hp
            mask_hp = h_pack_idx < num_h_packs
            h_group_idx = h_pack_idx // 4
            gate_packed = tl.load(
                gate_up_q_ptr
                + expert_id * stride_gu_e
                + i_idx[:, None] * stride_gu_n
                + h_pack_idx[None, :] * stride_gu_k,
                mask=mask_i[:, None] & mask_hp[None, :],
                other=0,
            ).to(tl.int32)
            up_packed = tl.load(
                gate_up_q_ptr
                + expert_id * stride_gu_e
                + (INTERMEDIATE + i_idx[:, None]) * stride_gu_n
                + h_pack_idx[None, :] * stride_gu_k,
                mask=mask_i[:, None] & mask_hp[None, :],
                other=0,
            ).to(tl.int32)
            gate_scale = tl.load(
                gate_up_s_ptr
                + expert_id * stride_gus_e
                + i_idx[:, None] * stride_gus_n
                + h_group_idx[None, :] * stride_gus_g,
                mask=mask_i[:, None] & mask_hp[None, :],
                other=0.0,
            ).to(tl.float32)
            up_scale = tl.load(
                gate_up_s_ptr
                + expert_id * stride_gus_e
                + (INTERMEDIATE + i_idx[:, None]) * stride_gus_n
                + h_group_idx[None, :] * stride_gus_g,
                mask=mask_i[:, None] & mask_hp[None, :],
                other=0.0,
            ).to(tl.float32)
            gate_partial = tl.zeros((BLOCK_I, BLOCK_PACKS_H), dtype=tl.float32)
            up_partial = tl.zeros((BLOCK_I, BLOCK_PACKS_H), dtype=tl.float32)
            for nibble_h in tl.static_range(0, 8):
                h_idx = h_pack_idx * 8 + nibble_h
                x_val = tl.load(
                    x_ptr + token_id * stride_x_m + h_idx * stride_x_h,
                    mask=mask_hp & (h_idx < H),
                    other=0.0,
                )
                gate_q = (gate_packed >> (nibble_h * 4)) & 0xF
                up_q = (up_packed >> (nibble_h * 4)) & 0xF
                gate_partial += (
                    x_val[None, :].to(tl.float32)
                    * (gate_q.to(tl.float32) - 8.0)
                    * gate_scale
                )
                up_partial += (
                    x_val[None, :].to(tl.float32)
                    * (up_q.to(tl.float32) - 8.0)
                    * up_scale
                )
            gate_acc += tl.sum(gate_partial, axis=1)
            up_acc += tl.sum(up_partial, axis=1)

        if ACT_KIND == 1:
            # GELU tanh approximation used by Gemma's gelu_pytorch_tanh.
            x3 = gate_acc * gate_acc * gate_acc
            inner = 0.7978845608028654 * (gate_acc + 0.044715 * x3)
            act = gate_acc / (1.0 + tl.exp(-2.0 * inner))
        else:
            act = gate_acc / (1.0 + tl.exp(-gate_acc))
        hidden = act * up_acc * route_weight
        down_pack_idx = i_idx // 8
        down_group_idx = i_idx // 32
        down_nibble = i_idx % 8
        down_packed = tl.load(
            down_q_ptr
            + expert_id * stride_d_e
            + offs_h[:, None] * stride_d_h
            + down_pack_idx[None, :] * stride_d_k,
            mask=mask_h[:, None] & mask_i[None, :],
            other=0,
        ).to(tl.int32)
        down_scale = tl.load(
            down_s_ptr
            + expert_id * stride_ds_e
            + offs_h[:, None] * stride_ds_h
            + down_group_idx[None, :] * stride_ds_g,
            mask=mask_h[:, None] & mask_i[None, :],
            other=0.0,
        ).to(tl.float32)
        down_q = (down_packed >> (down_nibble[None, :] * 4)) & 0xF
        acc_h += tl.sum(
            hidden[None, :] * (down_q.to(tl.float32) - 8.0) * down_scale,
            axis=1,
        )

    tl.atomic_add(
        acc_ptr + token_id * stride_acc_m + offs_h * stride_acc_h,
        acc_h,
        sem="relaxed",
        mask=mask_h,
    )


def _group_size_from_shapes(qweight: torch.Tensor, scales: torch.Tensor) -> int:
    packed_cols = int(qweight.shape[-1])
    scale_groups = int(scales.shape[-1])
    if packed_cols <= 0 or scale_groups <= 0:
        return 0
    return (packed_cols * 8) // scale_groups


def _validate_decode_inputs(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
) -> tuple[bool, str]:
    if x.dim() != 2 or int(x.shape[0]) != 1:
        return False, "input_not_m1_2d"
    if topk_weights.dim() != 2 or topk_ids.dim() != 2:
        return False, "routing_not_2d"
    if int(topk_weights.shape[0]) != 1 or int(topk_ids.shape[0]) != 1:
        return False, "routing_not_m1"
    if tuple(topk_weights.shape) != tuple(topk_ids.shape):
        return False, "routing_shape_mismatch"
    if not _activation_supported(activation):
        return False, "unsupported_activation"
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False, f"unsupported_dtype_{str(x.dtype)}"
    if (
        gate_up_qweight.dim() != 3
        or gate_up_scales.dim() != 3
        or down_qweight.dim() != 3
        or down_scales.dim() != 3
    ):
        return False, "weights_not_expert_major"
    if any(
        tensor.device != x.device
        for tensor in (
            topk_weights,
            topk_ids,
            gate_up_qweight,
            gate_up_scales,
            down_qweight,
            down_scales,
        )
    ):
        return False, "device_mismatch"

    hidden_dim = int(x.shape[1])
    intermediate = int(intermediate_dim)
    if hidden_dim % 32 != 0 or intermediate % 32 != 0:
        return False, "dims_not_group_aligned"
    if int(gate_up_qweight.shape[1]) < 2 * intermediate:
        return False, "gate_up_rows_mismatch"
    if int(gate_up_qweight.shape[2]) * 8 != hidden_dim:
        return False, "gate_up_k_mismatch"
    if int(down_qweight.shape[1]) < hidden_dim:
        return False, "down_rows_mismatch"
    if int(down_qweight.shape[2]) * 8 != intermediate:
        return False, "down_k_mismatch"
    if _group_size_from_shapes(gate_up_qweight, gate_up_scales) != 32:
        return False, "gate_up_group_not_32"
    if _group_size_from_shapes(down_qweight, down_scales) != 32:
        return False, "down_group_not_32"
    return True, "ok"


def _validate_batched_decode_inputs(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
) -> tuple[bool, str]:
    if x.dim() != 2:
        return False, "input_not_2d"
    if int(x.shape[0]) <= 0:
        return False, "empty_batch"
    if int(x.shape[0]) > 16:
        return False, "batch_too_large_for_decode"
    if topk_weights.dim() != 2 or topk_ids.dim() != 2:
        return False, "routing_not_2d"
    if tuple(topk_weights.shape) != tuple(topk_ids.shape):
        return False, "routing_shape_mismatch"
    if int(topk_weights.shape[0]) != int(x.shape[0]):
        return False, "routing_batch_mismatch"
    if not _activation_supported(activation):
        return False, "unsupported_activation"
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False, f"unsupported_dtype_{str(x.dtype)}"
    if (
        gate_up_qweight.dim() != 3
        or gate_up_scales.dim() != 3
        or down_qweight.dim() != 3
        or down_scales.dim() != 3
    ):
        return False, "weights_not_expert_major"
    if any(
        tensor.device != x.device
        for tensor in (
            topk_weights,
            topk_ids,
            gate_up_qweight,
            gate_up_scales,
            down_qweight,
            down_scales,
        )
    ):
        return False, "device_mismatch"

    hidden_dim = int(x.shape[1])
    intermediate = int(intermediate_dim)
    if hidden_dim % 32 != 0 or intermediate % 32 != 0:
        return False, "dims_not_group_aligned"
    if int(gate_up_qweight.shape[1]) < 2 * intermediate:
        return False, "gate_up_rows_mismatch"
    if int(gate_up_qweight.shape[2]) * 8 != hidden_dim:
        return False, "gate_up_k_mismatch"
    if int(down_qweight.shape[1]) < hidden_dim:
        return False, "down_rows_mismatch"
    if int(down_qweight.shape[2]) * 8 != intermediate:
        return False, "down_k_mismatch"
    if _group_size_from_shapes(gate_up_qweight, gate_up_scales) != 32:
        return False, "gate_up_group_not_32"
    if _group_size_from_shapes(down_qweight, down_scales) != 32:
        return False, "down_group_not_32"
    return True, "ok"


def _validate_batched_prefill_inputs(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
) -> tuple[bool, str]:
    if x.dim() != 2:
        return False, "input_not_2d"
    if int(x.shape[0]) <= 0:
        return False, "empty_batch"
    if topk_weights.dim() != 2 or topk_ids.dim() != 2:
        return False, "routing_not_2d"
    if tuple(topk_weights.shape) != tuple(topk_ids.shape):
        return False, "routing_shape_mismatch"
    if int(topk_weights.shape[0]) != int(x.shape[0]):
        return False, "routing_batch_mismatch"
    if not _activation_supported(activation):
        return False, "unsupported_activation"
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False, f"unsupported_dtype_{str(x.dtype)}"
    if (
        gate_up_qweight.dim() != 3
        or gate_up_scales.dim() != 3
        or down_qweight.dim() != 3
        or down_scales.dim() != 3
    ):
        return False, "weights_not_expert_major"
    if any(
        tensor.device != x.device
        for tensor in (
            topk_weights,
            topk_ids,
            gate_up_qweight,
            gate_up_scales,
            down_qweight,
            down_scales,
        )
    ):
        return False, "device_mismatch"

    hidden_dim = int(x.shape[1])
    intermediate = int(intermediate_dim)
    if hidden_dim % 32 != 0 or intermediate % 32 != 0:
        return False, "dims_not_group_aligned"
    if int(gate_up_qweight.shape[1]) < 2 * intermediate:
        return False, "gate_up_rows_mismatch"
    if int(gate_up_qweight.shape[2]) * 8 != hidden_dim:
        return False, "gate_up_k_mismatch"
    if int(down_qweight.shape[1]) < hidden_dim:
        return False, "down_rows_mismatch"
    if int(down_qweight.shape[2]) * 8 != intermediate:
        return False, "down_k_mismatch"
    if _group_size_from_shapes(gate_up_qweight, gate_up_scales) != 32:
        return False, "gate_up_group_not_32"
    if _group_size_from_shapes(down_qweight, down_scales) != 32:
        return False, "down_group_not_32"
    return True, "ok"


def gemma4_moe_int4_decode(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
) -> tuple[torch.Tensor, bool, str]:
    supported, reason = _validate_decode_inputs(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation=activation,
    )
    if not supported:
        return x, False, reason

    hidden_dim = int(x.shape[1])
    intermediate = int(intermediate_dim)
    top_k = int(topk_ids.shape[1])
    if top_k <= 0:
        return torch.zeros_like(x), True, "ok"
    act_kind = _activation_kind(activation)

    gate_up_q = gate_up_qweight.contiguous()
    gate_up_s = gate_up_scales.contiguous()
    down_q = down_qweight.contiguous()
    down_s = down_scales.contiguous()
    x_contig = x.contiguous()
    route_w = topk_weights.contiguous()
    route_ids = topk_ids.contiguous()

    tmp = torch.empty((top_k, intermediate), device=x.device, dtype=torch.float32)
    out = torch.empty((1, hidden_dim), device=x.device, dtype=x.dtype)
    block_i = 32 if intermediate <= 1024 else 64
    block_h = 64 if hidden_dim <= 4096 else 128
    block_packs = 8
    _gemma4_moe_gate_up_m1_kernel[(top_k, triton.cdiv(intermediate, block_i))](
        x_contig,
        route_ids,
        route_w,
        gate_up_q,
        gate_up_s,
        tmp,
        H=hidden_dim,
        INTERMEDIATE=intermediate,
        stride_gu_e=gate_up_q.stride(0),
        stride_gu_n=gate_up_q.stride(1),
        stride_gu_k=gate_up_q.stride(2),
        stride_gus_e=gate_up_s.stride(0),
        stride_gus_n=gate_up_s.stride(1),
        stride_gus_g=gate_up_s.stride(2),
        stride_tmp_k=tmp.stride(0),
        stride_tmp_i=tmp.stride(1),
        BLOCK_I=block_i,
        BLOCK_PACKS=block_packs,
        ACT_KIND=act_kind,
        num_warps=4,
        num_stages=1,
    )
    _gemma4_moe_down_reduce_m1_kernel[(triton.cdiv(hidden_dim, block_h),)](
        route_ids,
        down_q,
        down_s,
        tmp,
        out,
        H=hidden_dim,
        INTERMEDIATE=intermediate,
        TOP_K=top_k,
        stride_d_e=down_q.stride(0),
        stride_d_h=down_q.stride(1),
        stride_d_k=down_q.stride(2),
        stride_ds_e=down_s.stride(0),
        stride_ds_h=down_s.stride(1),
        stride_ds_g=down_s.stride(2),
        stride_tmp_k=tmp.stride(0),
        stride_tmp_i=tmp.stride(1),
        BLOCK_H=block_h,
        BLOCK_PACKS=block_packs,
        USE_BF16_OUTPUT=x.dtype == torch.bfloat16,
        num_warps=4,
        num_stages=1,
    )
    return out, True, "ok"


def gemma4_moe_int4_decode_batched(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
) -> tuple[torch.Tensor, bool, str]:
    supported, reason = _validate_batched_decode_inputs(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation=activation,
    )
    if not supported:
        return x, False, reason

    n_tokens = int(x.shape[0])
    hidden_dim = int(x.shape[1])
    intermediate = int(intermediate_dim)
    top_k = int(topk_ids.shape[1])
    if top_k <= 0:
        return torch.zeros_like(x), True, "ok"
    act_kind = _activation_kind(activation)

    gate_up_q = gate_up_qweight.contiguous()
    gate_up_s = gate_up_scales.contiguous()
    down_q = down_qweight.contiguous()
    down_s = down_scales.contiguous()
    x_contig = x.contiguous()
    route_w = topk_weights.contiguous()
    route_ids = topk_ids.contiguous()

    tmp = torch.empty(
        (n_tokens, top_k, intermediate), device=x.device, dtype=torch.float32
    )
    out = torch.empty((n_tokens, hidden_dim), device=x.device, dtype=x.dtype)
    block_i = 32 if intermediate <= 1024 else 64
    block_h = 64 if hidden_dim <= 4096 else 128
    block_packs = 8
    _gemma4_moe_gate_up_batched_kernel[
        (n_tokens, top_k, triton.cdiv(intermediate, block_i))
    ](
        x_contig,
        route_ids,
        route_w,
        gate_up_q,
        gate_up_s,
        tmp,
        H=hidden_dim,
        INTERMEDIATE=intermediate,
        stride_x_m=x_contig.stride(0),
        stride_ids_m=route_ids.stride(0),
        stride_ids_k=route_ids.stride(1),
        stride_w_m=route_w.stride(0),
        stride_w_k=route_w.stride(1),
        stride_gu_e=gate_up_q.stride(0),
        stride_gu_n=gate_up_q.stride(1),
        stride_gu_k=gate_up_q.stride(2),
        stride_gus_e=gate_up_s.stride(0),
        stride_gus_n=gate_up_s.stride(1),
        stride_gus_g=gate_up_s.stride(2),
        stride_tmp_m=tmp.stride(0),
        stride_tmp_k=tmp.stride(1),
        stride_tmp_i=tmp.stride(2),
        BLOCK_I=block_i,
        BLOCK_PACKS=block_packs,
        ACT_KIND=act_kind,
        num_warps=4,
        num_stages=1,
    )
    _gemma4_moe_down_reduce_batched_kernel[
        (n_tokens, triton.cdiv(hidden_dim, block_h))
    ](
        route_ids,
        down_q,
        down_s,
        tmp,
        out,
        H=hidden_dim,
        INTERMEDIATE=intermediate,
        TOP_K=top_k,
        stride_ids_m=route_ids.stride(0),
        stride_ids_k=route_ids.stride(1),
        stride_d_e=down_q.stride(0),
        stride_d_h=down_q.stride(1),
        stride_d_k=down_q.stride(2),
        stride_ds_e=down_s.stride(0),
        stride_ds_h=down_s.stride(1),
        stride_ds_g=down_s.stride(2),
        stride_tmp_m=tmp.stride(0),
        stride_tmp_k=tmp.stride(1),
        stride_tmp_i=tmp.stride(2),
        stride_out_m=out.stride(0),
        stride_out_h=out.stride(1),
        BLOCK_H=block_h,
        BLOCK_PACKS=block_packs,
        USE_BF16_OUTPUT=x.dtype == torch.bfloat16,
        num_warps=4,
        num_stages=1,
    )
    return out, True, "ok"


def gemma4_moe_int4_decode_batched_tuned(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
    block_h_override: int | None = None,
    block_packs_override: int | None = None,
) -> tuple[torch.Tensor, bool, str]:
    supported, reason = _validate_batched_decode_inputs(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation=activation,
    )
    if not supported:
        return x, False, reason

    n_tokens = int(x.shape[0])
    hidden_dim = int(x.shape[1])
    intermediate = int(intermediate_dim)
    top_k = int(topk_ids.shape[1])
    if top_k <= 0:
        return torch.zeros_like(x), True, "ok"
    act_kind = _activation_kind(activation)

    gate_up_q = gate_up_qweight.contiguous()
    gate_up_s = gate_up_scales.contiguous()
    down_q = down_qweight.contiguous()
    down_s = down_scales.contiguous()
    x_contig = x.contiguous()
    route_w = topk_weights.contiguous()
    route_ids = topk_ids.contiguous()

    tmp = torch.empty(
        (n_tokens, top_k, intermediate), device=x.device, dtype=torch.float32
    )
    out = torch.empty((n_tokens, hidden_dim), device=x.device, dtype=x.dtype)
    block_i = 32 if intermediate <= 1024 else 64
    default_block_h = (
        _TUNED_DOWN_BLOCK_H_SMALL_H
        if hidden_dim <= 4096
        else _TUNED_DOWN_BLOCK_H_LARGE_H
    )
    block_h = int(block_h_override or default_block_h)
    down_block_packs = int(block_packs_override or _TUNED_DOWN_BLOCK_PACKS)
    if not _is_power_of_two(block_h) or not _is_power_of_two(down_block_packs):
        return x, False, "invalid_tuned_tile"

    _gemma4_moe_gate_up_batched_kernel[
        (n_tokens, top_k, triton.cdiv(intermediate, block_i))
    ](
        x_contig,
        route_ids,
        route_w,
        gate_up_q,
        gate_up_s,
        tmp,
        H=hidden_dim,
        INTERMEDIATE=intermediate,
        stride_x_m=x_contig.stride(0),
        stride_ids_m=route_ids.stride(0),
        stride_ids_k=route_ids.stride(1),
        stride_w_m=route_w.stride(0),
        stride_w_k=route_w.stride(1),
        stride_gu_e=gate_up_q.stride(0),
        stride_gu_n=gate_up_q.stride(1),
        stride_gu_k=gate_up_q.stride(2),
        stride_gus_e=gate_up_s.stride(0),
        stride_gus_n=gate_up_s.stride(1),
        stride_gus_g=gate_up_s.stride(2),
        stride_tmp_m=tmp.stride(0),
        stride_tmp_k=tmp.stride(1),
        stride_tmp_i=tmp.stride(2),
        BLOCK_I=block_i,
        BLOCK_PACKS=_TUNED_GATE_UP_BLOCK_PACKS,
        ACT_KIND=act_kind,
        num_warps=4,
        num_stages=1,
    )
    _gemma4_moe_down_reduce_batched_token_major_kernel[
        (triton.cdiv(hidden_dim, block_h), n_tokens)
    ](
        route_ids,
        down_q,
        down_s,
        tmp,
        out,
        H=hidden_dim,
        INTERMEDIATE=intermediate,
        TOP_K=top_k,
        stride_ids_m=route_ids.stride(0),
        stride_ids_k=route_ids.stride(1),
        stride_d_e=down_q.stride(0),
        stride_d_h=down_q.stride(1),
        stride_d_k=down_q.stride(2),
        stride_ds_e=down_s.stride(0),
        stride_ds_h=down_s.stride(1),
        stride_ds_g=down_s.stride(2),
        stride_tmp_m=tmp.stride(0),
        stride_tmp_k=tmp.stride(1),
        stride_tmp_i=tmp.stride(2),
        stride_out_m=out.stride(0),
        stride_out_h=out.stride(1),
        BLOCK_H=block_h,
        BLOCK_PACKS=down_block_packs,
        USE_BF16_OUTPUT=x.dtype == torch.bfloat16,
        num_warps=4,
        num_stages=1,
    )
    return out, True, "ok"


def gemma4_moe_int4_decode_batched_chunked(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
    block_i_override: int | None = None,
) -> tuple[torch.Tensor, bool, str]:
    supported, reason = _validate_batched_decode_inputs(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation=activation,
    )
    if not supported:
        return x, False, reason

    n_tokens = int(x.shape[0])
    hidden_dim = int(x.shape[1])
    intermediate = int(intermediate_dim)
    top_k = int(topk_ids.shape[1])
    if top_k <= 0:
        return torch.zeros_like(x), True, "ok"
    act_kind = _activation_kind(activation)

    block_i = int(block_i_override or min(_CHUNKED_BLOCK_I, intermediate))
    if not _is_power_of_two(block_i) or block_i % 32 != 0:
        return x, False, "invalid_chunked_tile"

    gate_up_q = gate_up_qweight.contiguous()
    gate_up_s = gate_up_scales.contiguous()
    down_q = down_qweight.contiguous()
    down_s = down_scales.contiguous()
    x_contig = x.contiguous()
    route_w = topk_weights.contiguous()
    route_ids = topk_ids.contiguous()

    tmp = torch.empty((n_tokens, top_k, block_i), device=x.device, dtype=torch.float32)
    acc = torch.empty((n_tokens, hidden_dim), device=x.device, dtype=torch.float32)
    out = torch.empty((n_tokens, hidden_dim), device=x.device, dtype=x.dtype)
    for i_offset in range(0, intermediate, block_i):
        is_first_chunk = i_offset == 0
        is_last_chunk = i_offset + block_i >= intermediate
        with _profile_span("moe_int4_decode_gate_up_chunk"):
            _gemma4_moe_gate_up_batched_chunk_kernel[(n_tokens, top_k)](
                x_contig,
                route_ids,
                route_w,
                gate_up_q,
                gate_up_s,
                tmp,
                H=hidden_dim,
                INTERMEDIATE=intermediate,
                I_OFFSET=i_offset,
                TMP_OFFSET=0,
                stride_x_m=x_contig.stride(0),
                stride_ids_m=route_ids.stride(0),
                stride_ids_k=route_ids.stride(1),
                stride_w_m=route_w.stride(0),
                stride_w_k=route_w.stride(1),
                stride_gu_e=gate_up_q.stride(0),
                stride_gu_n=gate_up_q.stride(1),
                stride_gu_k=gate_up_q.stride(2),
                stride_gus_e=gate_up_s.stride(0),
                stride_gus_n=gate_up_s.stride(1),
                stride_gus_g=gate_up_s.stride(2),
                stride_tmp_m=tmp.stride(0),
                stride_tmp_k=tmp.stride(1),
                stride_tmp_i=tmp.stride(2),
                BLOCK_I=block_i,
                BLOCK_PACKS=_CHUNKED_BLOCK_PACKS_H,
                ACT_KIND=act_kind,
                num_warps=4,
                num_stages=1,
            )
        with _profile_span("moe_int4_decode_down_reduce_chunk"):
            _gemma4_moe_down_reduce_batched_chunk_kernel[
                (triton.cdiv(hidden_dim, _CHUNKED_BLOCK_H), n_tokens)
            ](
                route_ids,
                down_q,
                down_s,
                tmp,
                acc,
                out,
                H=hidden_dim,
                INTERMEDIATE=intermediate,
                TOP_K=top_k,
                I_OFFSET=i_offset,
                TMP_OFFSET=0,
                stride_ids_m=route_ids.stride(0),
                stride_ids_k=route_ids.stride(1),
                stride_d_e=down_q.stride(0),
                stride_d_h=down_q.stride(1),
                stride_d_k=down_q.stride(2),
                stride_ds_e=down_s.stride(0),
                stride_ds_h=down_s.stride(1),
                stride_ds_g=down_s.stride(2),
                stride_tmp_m=tmp.stride(0),
                stride_tmp_k=tmp.stride(1),
                stride_tmp_i=tmp.stride(2),
                stride_acc_m=acc.stride(0),
                stride_acc_h=acc.stride(1),
                stride_out_m=out.stride(0),
                stride_out_h=out.stride(1),
                BLOCK_H=_CHUNKED_BLOCK_H,
                BLOCK_I=block_i,
                BLOCK_PACKS=_CHUNKED_BLOCK_PACKS_I,
                IS_FIRST_CHUNK=is_first_chunk,
                IS_LAST_CHUNK=is_last_chunk,
                USE_BF16_OUTPUT=x.dtype == torch.bfloat16,
                num_warps=4,
                num_stages=1,
            )
    return out, True, "ok"


def gemma4_moe_int4_decode_batched_grouped(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
    block_i_override: int | None = None,
) -> tuple[torch.Tensor, bool, str]:
    supported, reason = _validate_batched_decode_inputs(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation=activation,
    )
    if not supported:
        return x, False, reason

    n_tokens = int(x.shape[0])
    hidden_dim = int(x.shape[1])
    intermediate = int(intermediate_dim)
    top_k = int(topk_ids.shape[1])
    if top_k <= 0:
        return torch.zeros_like(x), True, "ok"
    act_kind = _activation_kind(activation)

    block_i = int(block_i_override or min(_CHUNKED_BLOCK_I, intermediate))
    if not _is_power_of_two(block_i) or block_i % 32 != 0:
        return x, False, "invalid_grouped_tile"

    gate_up_q = gate_up_qweight.contiguous()
    gate_up_s = gate_up_scales.contiguous()
    down_q = down_qweight.contiguous()
    down_s = down_scales.contiguous()
    x_contig = x.contiguous()
    route_w = topk_weights.contiguous()
    route_ids = topk_ids.contiguous()

    tmp = torch.empty((n_tokens, block_i, top_k), device=x.device, dtype=x.dtype)
    acc = torch.empty((n_tokens, hidden_dim), device=x.device, dtype=torch.float32)
    out = torch.empty((n_tokens, hidden_dim), device=x.device, dtype=x.dtype)
    for i_offset in range(0, intermediate, block_i):
        is_first_chunk = i_offset == 0
        is_last_chunk = i_offset + block_i >= intermediate
        with _profile_span("moe_int4_prefill_gate_up_chunk"):
            _gemma4_moe_gate_up_batched_chunk_kernel[(n_tokens, top_k)](
                x_contig,
                route_ids,
                route_w,
                gate_up_q,
                gate_up_s,
                tmp,
                H=hidden_dim,
                INTERMEDIATE=intermediate,
                I_OFFSET=i_offset,
                TMP_OFFSET=0,
                stride_x_m=x_contig.stride(0),
                stride_ids_m=route_ids.stride(0),
                stride_ids_k=route_ids.stride(1),
                stride_w_m=route_w.stride(0),
                stride_w_k=route_w.stride(1),
                stride_gu_e=gate_up_q.stride(0),
                stride_gu_n=gate_up_q.stride(1),
                stride_gu_k=gate_up_q.stride(2),
                stride_gus_e=gate_up_s.stride(0),
                stride_gus_n=gate_up_s.stride(1),
                stride_gus_g=gate_up_s.stride(2),
                stride_tmp_m=tmp.stride(0),
                stride_tmp_k=tmp.stride(2),
                stride_tmp_i=tmp.stride(1),
                BLOCK_I=block_i,
                BLOCK_PACKS=_CHUNKED_BLOCK_PACKS_H,
                ACT_KIND=act_kind,
                num_warps=4,
                num_stages=1,
            )
        with _profile_span("moe_int4_prefill_down_reduce_chunk"):
            _gemma4_moe_down_reduce_batched_grouped_chunk_kernel[
                (triton.cdiv(hidden_dim, _CHUNKED_BLOCK_H), n_tokens)
            ](
                route_ids,
                down_q,
                down_s,
                tmp,
                acc,
                out,
                H=hidden_dim,
                INTERMEDIATE=intermediate,
                TOP_K=top_k,
                I_OFFSET=i_offset,
                stride_ids_m=route_ids.stride(0),
                stride_ids_k=route_ids.stride(1),
                stride_d_e=down_q.stride(0),
                stride_d_h=down_q.stride(1),
                stride_d_k=down_q.stride(2),
                stride_ds_e=down_s.stride(0),
                stride_ds_h=down_s.stride(1),
                stride_ds_g=down_s.stride(2),
                stride_tmp_m=tmp.stride(0),
                stride_tmp_i=tmp.stride(1),
                stride_tmp_k=tmp.stride(2),
                stride_acc_m=acc.stride(0),
                stride_acc_h=acc.stride(1),
                stride_out_m=out.stride(0),
                stride_out_h=out.stride(1),
                BLOCK_H=_CHUNKED_BLOCK_H,
                BLOCK_I=block_i,
                BLOCK_PACKS=_CHUNKED_BLOCK_PACKS_I,
                IS_FIRST_CHUNK=is_first_chunk,
                IS_LAST_CHUNK=is_last_chunk,
                USE_BF16_OUTPUT=x.dtype == torch.bfloat16,
                num_warps=4,
                num_stages=1,
            )
    return out, True, "ok"


def gemma4_moe_int4_decode_batched_grouped_streaming(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
    block_i_override: int | None = None,
) -> tuple[torch.Tensor, bool, str]:
    """
    Decode grouped/chunked strategy with compact top-k tmp layout.

    The underlying kernels keep only one intermediate chunk in global memory as
    [M, BLOCK_I, top_k], with route positions adjacent for each intermediate
    row. This is the low-risk grouped streaming candidate for reducing the
    tmp[M, top_k, I] write/read footprint without repeating gate/up per H tile.
    """
    return gemma4_moe_int4_decode_batched_grouped(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation=activation,
        block_i_override=block_i_override,
    )


def gemma4_moe_int4_prefill_grouped(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
    block_i_override: int | None = None,
) -> tuple[torch.Tensor, bool, str]:
    """
    Prefill grouped Gemma4 MoE path for expert-major symmetric int4 weights.

    Layout:
      x: [M, H]
      topk_ids/topk_weights: [M, top_k]
      gate_up_qweight: [E, 2 * intermediate, H/8]
      gate_up_scales: [E, 2 * intermediate, H/32]
      down_qweight: [E, H, intermediate/8]
      down_scales: [E, H, intermediate/32]

    Tiling:
      The kernel path streams one intermediate chunk at a time. Gate/up writes
      a compact [M, BLOCK_I, top_k] route-major chunk, and down/reduce consumes
      that chunk immediately into a fp32 [M, H] accumulator. This keeps the
      large [M, top_k, intermediate] buffer out of global memory and consumes
      packed int4 weights directly without dense expert materialization.
    """
    supported, reason = _validate_batched_prefill_inputs(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation=activation,
    )
    if not supported:
        return x, False, reason

    n_tokens = int(x.shape[0])
    hidden_dim = int(x.shape[1])
    intermediate = int(intermediate_dim)
    top_k = int(topk_ids.shape[1])
    if top_k <= 0:
        return torch.zeros_like(x), True, "ok"
    act_kind = _activation_kind(activation)

    block_i = int(block_i_override or min(_CHUNKED_BLOCK_I, intermediate))
    if not _is_power_of_two(block_i) or block_i % 32 != 0:
        return x, False, "invalid_prefill_grouped_tile"

    gate_up_q = gate_up_qweight.contiguous()
    gate_up_s = gate_up_scales.contiguous()
    down_q = down_qweight.contiguous()
    down_s = down_scales.contiguous()
    x_contig = x.contiguous()
    route_w = topk_weights.contiguous()
    route_ids = topk_ids.contiguous()

    tmp = torch.empty((n_tokens, block_i, top_k), device=x.device, dtype=x.dtype)
    acc = torch.empty((n_tokens, hidden_dim), device=x.device, dtype=torch.float32)
    out = torch.empty((n_tokens, hidden_dim), device=x.device, dtype=x.dtype)
    for i_offset in range(0, intermediate, block_i):
        is_first_chunk = i_offset == 0
        is_last_chunk = i_offset + block_i >= intermediate
        _gemma4_moe_gate_up_batched_chunk_kernel[(n_tokens, top_k)](
            x_contig,
            route_ids,
            route_w,
            gate_up_q,
            gate_up_s,
            tmp,
            H=hidden_dim,
            INTERMEDIATE=intermediate,
            I_OFFSET=i_offset,
            TMP_OFFSET=0,
            stride_x_m=x_contig.stride(0),
            stride_ids_m=route_ids.stride(0),
            stride_ids_k=route_ids.stride(1),
            stride_w_m=route_w.stride(0),
            stride_w_k=route_w.stride(1),
            stride_gu_e=gate_up_q.stride(0),
            stride_gu_n=gate_up_q.stride(1),
            stride_gu_k=gate_up_q.stride(2),
            stride_gus_e=gate_up_s.stride(0),
            stride_gus_n=gate_up_s.stride(1),
            stride_gus_g=gate_up_s.stride(2),
            stride_tmp_m=tmp.stride(0),
            stride_tmp_k=tmp.stride(2),
            stride_tmp_i=tmp.stride(1),
            BLOCK_I=block_i,
            BLOCK_PACKS=_CHUNKED_BLOCK_PACKS_H,
            ACT_KIND=act_kind,
            num_warps=4,
            num_stages=1,
        )
        _gemma4_moe_down_reduce_batched_grouped_chunk_kernel[
            (triton.cdiv(hidden_dim, _CHUNKED_BLOCK_H), n_tokens)
        ](
            route_ids,
            down_q,
            down_s,
            tmp,
            acc,
            out,
            H=hidden_dim,
            INTERMEDIATE=intermediate,
            TOP_K=top_k,
            I_OFFSET=i_offset,
            stride_ids_m=route_ids.stride(0),
            stride_ids_k=route_ids.stride(1),
            stride_d_e=down_q.stride(0),
            stride_d_h=down_q.stride(1),
            stride_d_k=down_q.stride(2),
            stride_ds_e=down_s.stride(0),
            stride_ds_h=down_s.stride(1),
            stride_ds_g=down_s.stride(2),
            stride_tmp_m=tmp.stride(0),
            stride_tmp_i=tmp.stride(1),
            stride_tmp_k=tmp.stride(2),
            stride_acc_m=acc.stride(0),
            stride_acc_h=acc.stride(1),
            stride_out_m=out.stride(0),
            stride_out_h=out.stride(1),
            BLOCK_H=_CHUNKED_BLOCK_H,
            BLOCK_I=block_i,
            BLOCK_PACKS=_CHUNKED_BLOCK_PACKS_I,
            IS_FIRST_CHUNK=is_first_chunk,
            IS_LAST_CHUNK=is_last_chunk,
            USE_BF16_OUTPUT=x.dtype == torch.bfloat16,
            num_warps=4,
            num_stages=1,
        )
    return out, True, "ok"


def gemma4_moe_int4_prefill_grouped_fused(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
    block_i_override: int | None = None,
    block_h_override: int | None = None,
) -> tuple[torch.Tensor, bool, str]:
    """
    Single-kernel grouped prefill path for expert-major symmetric int4 weights.

    The wrapper flattens top-k routes and sorts them by expert id so neighboring
    programs consume the same expert-major packed weights. The Triton kernel
    directly dequantizes gate/up and down weights in registers and accumulates
    each routed contribution into a fp32 output buffer.
    """
    supported, reason = _validate_batched_prefill_inputs(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation=activation,
    )
    if not supported:
        return x, False, reason

    n_tokens = int(x.shape[0])
    hidden_dim = int(x.shape[1])
    intermediate = int(intermediate_dim)
    top_k = int(topk_ids.shape[1])
    if top_k <= 0:
        return torch.zeros_like(x), True, "ok"
    act_kind = _activation_kind(activation)

    block_i = int(block_i_override or min(64, intermediate))
    block_h = int(block_h_override or _CHUNKED_BLOCK_H)
    if (
        not _is_power_of_two(block_i)
        or block_i % 32 != 0
        or not _is_power_of_two(block_h)
    ):
        return x, False, "invalid_prefill_grouped_fused_tile"

    gate_up_q = gate_up_qweight.contiguous()
    gate_up_s = gate_up_scales.contiguous()
    down_q = down_qweight.contiguous()
    down_s = down_scales.contiguous()
    x_contig = x.contiguous()
    route_w = topk_weights.contiguous().reshape(-1)
    route_ids = topk_ids.contiguous().reshape(-1).to(torch.long)
    route_tokens = torch.arange(
        n_tokens,
        device=x.device,
        dtype=torch.long,
    ).repeat_interleave(top_k)
    sort_idx = torch.argsort(route_ids, stable=True)
    sorted_ids = route_ids.index_select(0, sort_idx).contiguous()
    sorted_tokens = route_tokens.index_select(0, sort_idx).contiguous()
    sorted_weights = route_w.index_select(0, sort_idx).contiguous()
    route_count = int(sorted_ids.numel())
    if route_count <= 0:
        return torch.zeros_like(x), True, "ok"

    acc = torch.zeros((n_tokens, hidden_dim), device=x.device, dtype=torch.float32)
    with _profile_span("moe_int4_prefill_grouped_fused"):
        _gemma4_moe_prefill_grouped_fused_kernel[
            (triton.cdiv(hidden_dim, block_h), route_count)
        ](
            x_contig,
            sorted_tokens,
            sorted_ids,
            sorted_weights,
            gate_up_q,
            gate_up_s,
            down_q,
            down_s,
            acc,
            H=hidden_dim,
            INTERMEDIATE=intermediate,
            stride_x_m=x_contig.stride(0),
            stride_x_h=x_contig.stride(1),
            stride_gu_e=gate_up_q.stride(0),
            stride_gu_n=gate_up_q.stride(1),
            stride_gu_k=gate_up_q.stride(2),
            stride_gus_e=gate_up_s.stride(0),
            stride_gus_n=gate_up_s.stride(1),
            stride_gus_g=gate_up_s.stride(2),
            stride_d_e=down_q.stride(0),
            stride_d_h=down_q.stride(1),
            stride_d_k=down_q.stride(2),
            stride_ds_e=down_s.stride(0),
            stride_ds_h=down_s.stride(1),
            stride_ds_g=down_s.stride(2),
            stride_acc_m=acc.stride(0),
            stride_acc_h=acc.stride(1),
            BLOCK_H=block_h,
            BLOCK_I=block_i,
            BLOCK_PACKS_H=_CHUNKED_BLOCK_PACKS_H,
            ACT_KIND=act_kind,
            num_warps=4,
            num_stages=1,
        )
    return acc.to(x.dtype), True, "ok"


def gemma4_moe_int4_decode_batched_chunked_pair(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
    block_i_override: int | None = None,
) -> tuple[torch.Tensor, bool, str]:
    supported, reason = _validate_batched_decode_inputs(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation=activation,
    )
    if not supported:
        return x, False, reason

    n_tokens = int(x.shape[0])
    hidden_dim = int(x.shape[1])
    intermediate = int(intermediate_dim)
    top_k = int(topk_ids.shape[1])
    if top_k <= 0:
        return torch.zeros_like(x), True, "ok"
    act_kind = _activation_kind(activation)

    block_i = int(block_i_override or min(_CHUNKED_BLOCK_I, intermediate))
    pair_block_i = block_i * 2
    if (
        not _is_power_of_two(block_i)
        or block_i % 32 != 0
        or not _is_power_of_two(pair_block_i)
        or pair_block_i % 32 != 0
    ):
        return x, False, "invalid_chunked_tile"

    gate_up_q = gate_up_qweight.contiguous()
    gate_up_s = gate_up_scales.contiguous()
    down_q = down_qweight.contiguous()
    down_s = down_scales.contiguous()
    x_contig = x.contiguous()
    route_w = topk_weights.contiguous()
    route_ids = topk_ids.contiguous()

    tmp = torch.empty(
        (n_tokens, top_k, pair_block_i), device=x.device, dtype=torch.float32
    )
    acc = torch.empty((n_tokens, hidden_dim), device=x.device, dtype=torch.float32)
    out = torch.empty((n_tokens, hidden_dim), device=x.device, dtype=x.dtype)
    for pair_offset in range(0, intermediate, pair_block_i):
        _gemma4_moe_gate_up_batched_chunk_kernel[(n_tokens, top_k)](
            x_contig,
            route_ids,
            route_w,
            gate_up_q,
            gate_up_s,
            tmp,
            H=hidden_dim,
            INTERMEDIATE=intermediate,
            I_OFFSET=pair_offset,
            TMP_OFFSET=0,
            stride_x_m=x_contig.stride(0),
            stride_ids_m=route_ids.stride(0),
            stride_ids_k=route_ids.stride(1),
            stride_w_m=route_w.stride(0),
            stride_w_k=route_w.stride(1),
            stride_gu_e=gate_up_q.stride(0),
            stride_gu_n=gate_up_q.stride(1),
            stride_gu_k=gate_up_q.stride(2),
            stride_gus_e=gate_up_s.stride(0),
            stride_gus_n=gate_up_s.stride(1),
            stride_gus_g=gate_up_s.stride(2),
            stride_tmp_m=tmp.stride(0),
            stride_tmp_k=tmp.stride(1),
            stride_tmp_i=tmp.stride(2),
            BLOCK_I=pair_block_i,
            BLOCK_PACKS=_CHUNKED_BLOCK_PACKS_H,
            ACT_KIND=act_kind,
            num_warps=4,
            num_stages=1,
        )
        for tmp_offset in range(0, pair_block_i, block_i):
            i_offset = pair_offset + tmp_offset
            if i_offset >= intermediate:
                break
            is_first_chunk = i_offset == 0
            is_last_chunk = i_offset + block_i >= intermediate
            _gemma4_moe_down_reduce_batched_chunk_kernel[
                (triton.cdiv(hidden_dim, _CHUNKED_BLOCK_H), n_tokens)
            ](
                route_ids,
                down_q,
                down_s,
                tmp,
                acc,
                out,
                H=hidden_dim,
                INTERMEDIATE=intermediate,
                TOP_K=top_k,
                I_OFFSET=i_offset,
                TMP_OFFSET=tmp_offset,
                stride_ids_m=route_ids.stride(0),
                stride_ids_k=route_ids.stride(1),
                stride_d_e=down_q.stride(0),
                stride_d_h=down_q.stride(1),
                stride_d_k=down_q.stride(2),
                stride_ds_e=down_s.stride(0),
                stride_ds_h=down_s.stride(1),
                stride_ds_g=down_s.stride(2),
                stride_tmp_m=tmp.stride(0),
                stride_tmp_k=tmp.stride(1),
                stride_tmp_i=tmp.stride(2),
                stride_acc_m=acc.stride(0),
                stride_acc_h=acc.stride(1),
                stride_out_m=out.stride(0),
                stride_out_h=out.stride(1),
                BLOCK_H=_CHUNKED_BLOCK_H,
                BLOCK_I=block_i,
                BLOCK_PACKS=_CHUNKED_BLOCK_PACKS_I,
                IS_FIRST_CHUNK=is_first_chunk,
                IS_LAST_CHUNK=is_last_chunk,
                USE_BF16_OUTPUT=x.dtype == torch.bfloat16,
                num_warps=4,
                num_stages=1,
            )
    return out, True, "ok"


def gemma4_moe_int4_decode_batched_chunked_downpair(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
    block_i_override: int | None = None,
) -> tuple[torch.Tensor, bool, str]:
    supported, reason = _validate_batched_decode_inputs(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation=activation,
    )
    if not supported:
        return x, False, reason

    n_tokens = int(x.shape[0])
    hidden_dim = int(x.shape[1])
    intermediate = int(intermediate_dim)
    top_k = int(topk_ids.shape[1])
    if top_k <= 0:
        return torch.zeros_like(x), True, "ok"
    act_kind = _activation_kind(activation)

    block_i = int(block_i_override or min(_CHUNKED_BLOCK_I, intermediate))
    pair_block_i = block_i * 2
    if (
        not _is_power_of_two(block_i)
        or block_i % 32 != 0
        or not _is_power_of_two(pair_block_i)
        or pair_block_i % 32 != 0
    ):
        return x, False, "invalid_chunked_tile"

    gate_up_q = gate_up_qweight.contiguous()
    gate_up_s = gate_up_scales.contiguous()
    down_q = down_qweight.contiguous()
    down_s = down_scales.contiguous()
    x_contig = x.contiguous()
    route_w = topk_weights.contiguous()
    route_ids = topk_ids.contiguous()

    tmp = torch.empty(
        (n_tokens, top_k, pair_block_i), device=x.device, dtype=torch.float32
    )
    acc = torch.empty((n_tokens, hidden_dim), device=x.device, dtype=torch.float32)
    out = torch.empty((n_tokens, hidden_dim), device=x.device, dtype=x.dtype)
    for pair_offset in range(0, intermediate, pair_block_i):
        is_first_pair = pair_offset == 0
        is_last_pair = pair_offset + pair_block_i >= intermediate
        _gemma4_moe_gate_up_batched_chunk_kernel[(n_tokens, top_k)](
            x_contig,
            route_ids,
            route_w,
            gate_up_q,
            gate_up_s,
            tmp,
            H=hidden_dim,
            INTERMEDIATE=intermediate,
            I_OFFSET=pair_offset,
            TMP_OFFSET=0,
            stride_x_m=x_contig.stride(0),
            stride_ids_m=route_ids.stride(0),
            stride_ids_k=route_ids.stride(1),
            stride_w_m=route_w.stride(0),
            stride_w_k=route_w.stride(1),
            stride_gu_e=gate_up_q.stride(0),
            stride_gu_n=gate_up_q.stride(1),
            stride_gu_k=gate_up_q.stride(2),
            stride_gus_e=gate_up_s.stride(0),
            stride_gus_n=gate_up_s.stride(1),
            stride_gus_g=gate_up_s.stride(2),
            stride_tmp_m=tmp.stride(0),
            stride_tmp_k=tmp.stride(1),
            stride_tmp_i=tmp.stride(2),
            BLOCK_I=pair_block_i,
            BLOCK_PACKS=_CHUNKED_BLOCK_PACKS_H,
            ACT_KIND=act_kind,
            num_warps=4,
            num_stages=1,
        )
        _gemma4_moe_down_reduce_batched_chunk_pair_kernel[
            (triton.cdiv(hidden_dim, _CHUNKED_BLOCK_H), n_tokens)
        ](
            route_ids,
            down_q,
            down_s,
            tmp,
            acc,
            out,
            H=hidden_dim,
            INTERMEDIATE=intermediate,
            TOP_K=top_k,
            I_OFFSET=pair_offset,
            stride_ids_m=route_ids.stride(0),
            stride_ids_k=route_ids.stride(1),
            stride_d_e=down_q.stride(0),
            stride_d_h=down_q.stride(1),
            stride_d_k=down_q.stride(2),
            stride_ds_e=down_s.stride(0),
            stride_ds_h=down_s.stride(1),
            stride_ds_g=down_s.stride(2),
            stride_tmp_m=tmp.stride(0),
            stride_tmp_k=tmp.stride(1),
            stride_tmp_i=tmp.stride(2),
            stride_acc_m=acc.stride(0),
            stride_acc_h=acc.stride(1),
            stride_out_m=out.stride(0),
            stride_out_h=out.stride(1),
            BLOCK_H=_CHUNKED_BLOCK_H,
            BLOCK_I=block_i,
            BLOCK_PACKS=_CHUNKED_BLOCK_PACKS_I,
            IS_FIRST_PAIR=is_first_pair,
            IS_LAST_PAIR=is_last_pair,
            USE_BF16_OUTPUT=x.dtype == torch.bfloat16,
            num_warps=4,
            num_stages=1,
        )
    return out, True, "ok"


def gemma4_moe_int4_decode_batched_chunked_splitgate_downpair(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
    block_i_override: int | None = None,
) -> tuple[torch.Tensor, bool, str]:
    supported, reason = _validate_batched_decode_inputs(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation=activation,
    )
    if not supported:
        return x, False, reason

    n_tokens = int(x.shape[0])
    hidden_dim = int(x.shape[1])
    intermediate = int(intermediate_dim)
    top_k = int(topk_ids.shape[1])
    if top_k <= 0:
        return torch.zeros_like(x), True, "ok"
    act_kind = _activation_kind(activation)

    block_i = int(block_i_override or min(_CHUNKED_BLOCK_I, intermediate))
    pair_block_i = block_i * 2
    if (
        not _is_power_of_two(block_i)
        or block_i % 32 != 0
        or not _is_power_of_two(pair_block_i)
        or pair_block_i % 32 != 0
    ):
        return x, False, "invalid_chunked_tile"

    gate_up_q = gate_up_qweight.contiguous()
    gate_up_s = gate_up_scales.contiguous()
    down_q = down_qweight.contiguous()
    down_s = down_scales.contiguous()
    x_contig = x.contiguous()
    route_w = topk_weights.contiguous()
    route_ids = topk_ids.contiguous()

    tmp = torch.empty(
        (n_tokens, top_k, pair_block_i), device=x.device, dtype=torch.float32
    )
    acc = torch.empty((n_tokens, hidden_dim), device=x.device, dtype=torch.float32)
    out = torch.empty((n_tokens, hidden_dim), device=x.device, dtype=x.dtype)
    for pair_offset in range(0, intermediate, pair_block_i):
        is_first_pair = pair_offset == 0
        is_last_pair = pair_offset + pair_block_i >= intermediate
        for tmp_offset in range(0, pair_block_i, block_i):
            i_offset = pair_offset + tmp_offset
            if i_offset >= intermediate:
                break
            _gemma4_moe_gate_up_batched_chunk_kernel[(n_tokens, top_k)](
                x_contig,
                route_ids,
                route_w,
                gate_up_q,
                gate_up_s,
                tmp,
                H=hidden_dim,
                INTERMEDIATE=intermediate,
                I_OFFSET=i_offset,
                TMP_OFFSET=tmp_offset,
                stride_x_m=x_contig.stride(0),
                stride_ids_m=route_ids.stride(0),
                stride_ids_k=route_ids.stride(1),
                stride_w_m=route_w.stride(0),
                stride_w_k=route_w.stride(1),
                stride_gu_e=gate_up_q.stride(0),
                stride_gu_n=gate_up_q.stride(1),
                stride_gu_k=gate_up_q.stride(2),
                stride_gus_e=gate_up_s.stride(0),
                stride_gus_n=gate_up_s.stride(1),
                stride_gus_g=gate_up_s.stride(2),
                stride_tmp_m=tmp.stride(0),
                stride_tmp_k=tmp.stride(1),
                stride_tmp_i=tmp.stride(2),
                BLOCK_I=block_i,
                BLOCK_PACKS=_CHUNKED_BLOCK_PACKS_H,
                ACT_KIND=act_kind,
                num_warps=4,
                num_stages=1,
            )
        _gemma4_moe_down_reduce_batched_chunk_pair_kernel[
            (triton.cdiv(hidden_dim, _CHUNKED_BLOCK_H), n_tokens)
        ](
            route_ids,
            down_q,
            down_s,
            tmp,
            acc,
            out,
            H=hidden_dim,
            INTERMEDIATE=intermediate,
            TOP_K=top_k,
            I_OFFSET=pair_offset,
            stride_ids_m=route_ids.stride(0),
            stride_ids_k=route_ids.stride(1),
            stride_d_e=down_q.stride(0),
            stride_d_h=down_q.stride(1),
            stride_d_k=down_q.stride(2),
            stride_ds_e=down_s.stride(0),
            stride_ds_h=down_s.stride(1),
            stride_ds_g=down_s.stride(2),
            stride_tmp_m=tmp.stride(0),
            stride_tmp_k=tmp.stride(1),
            stride_tmp_i=tmp.stride(2),
            stride_acc_m=acc.stride(0),
            stride_acc_h=acc.stride(1),
            stride_out_m=out.stride(0),
            stride_out_h=out.stride(1),
            BLOCK_H=_CHUNKED_BLOCK_H,
            BLOCK_I=block_i,
            BLOCK_PACKS=_CHUNKED_BLOCK_PACKS_I,
            IS_FIRST_PAIR=is_first_pair,
            IS_LAST_PAIR=is_last_pair,
            USE_BF16_OUTPUT=x.dtype == torch.bfloat16,
            num_warps=4,
            num_stages=1,
        )
    return out, True, "ok"


def gemma4_moe_int4_decode_single_kernel(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    *,
    intermediate_dim: int,
    activation: str = "silu",
) -> tuple[torch.Tensor, bool, str]:
    supported, reason = _validate_decode_inputs(
        x,
        topk_weights,
        topk_ids,
        gate_up_qweight,
        gate_up_scales,
        down_qweight,
        down_scales,
        intermediate_dim=intermediate_dim,
        activation=activation,
    )
    if not supported:
        return x, False, reason

    hidden_dim = int(x.shape[1])
    intermediate = int(intermediate_dim)
    top_k = int(topk_ids.shape[1])
    if top_k <= 0:
        return torch.zeros_like(x), True, "ok"
    act_kind = _activation_kind(activation)

    gate_up_q = gate_up_qweight.contiguous()
    gate_up_s = gate_up_scales.contiguous()
    down_q = down_qweight.contiguous()
    down_s = down_scales.contiguous()
    x_contig = x.contiguous()
    route_w = topk_weights.contiguous()
    route_ids = topk_ids.contiguous()

    out = torch.empty((1, hidden_dim), device=x.device, dtype=x.dtype)
    block_h = 8 if hidden_dim <= 1024 else 16
    block_i = 16
    block_packs_h = 8
    _gemma4_moe_single_fused_m1_kernel[(triton.cdiv(hidden_dim, block_h),)](
        x_contig,
        route_ids,
        route_w,
        gate_up_q,
        gate_up_s,
        down_q,
        down_s,
        out,
        H=hidden_dim,
        INTERMEDIATE=intermediate,
        TOP_K=top_k,
        stride_gu_e=gate_up_q.stride(0),
        stride_gu_n=gate_up_q.stride(1),
        stride_gu_k=gate_up_q.stride(2),
        stride_gus_e=gate_up_s.stride(0),
        stride_gus_n=gate_up_s.stride(1),
        stride_gus_g=gate_up_s.stride(2),
        stride_d_e=down_q.stride(0),
        stride_d_h=down_q.stride(1),
        stride_d_k=down_q.stride(2),
        stride_ds_e=down_s.stride(0),
        stride_ds_h=down_s.stride(1),
        stride_ds_g=down_s.stride(2),
        BLOCK_H=block_h,
        BLOCK_I=block_i,
        BLOCK_PACKS_H=block_packs_h,
        USE_BF16_OUTPUT=x.dtype == torch.bfloat16,
        ACT_KIND=act_kind,
        num_warps=4,
        num_stages=1,
    )
    return out, True, "ok"
