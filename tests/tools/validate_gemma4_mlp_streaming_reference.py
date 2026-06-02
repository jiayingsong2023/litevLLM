# SPDX-License-Identifier: Apache-2.0
"""Validate Gemma-style AWQ MLP two-stage math against dense reference."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
import torch.nn.functional as F

from vllm.kernels.triton.awq_fused_gemm import (
    packed_int4_symmetric_fused_gate_up_m1_safe,
    packed_int4_symmetric_fused_gemm,
)
from vllm.model_executor.layers.quantization.tensor import (
    dequantize_symmetric_packed_int4_pytorch,
)


def _activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    act = activation.lower()
    if act in ("gelu", "gelu_pytorch_tanh"):
        return F.gelu(x, approximate="tanh")
    return F.silu(x)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    return float(((af * bf).sum() / (af.norm() * bf.norm() + 1e-8)).item())


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device is required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    group_size = 32
    torch.manual_seed(args.seed)

    x = torch.randn((1, args.hidden), device=device, dtype=dtype)
    gate_q = torch.randint(
        0, 255, (args.intermediate, args.hidden // 8), device=device, dtype=torch.int32
    )
    up_q = torch.randint(
        0, 255, (args.intermediate, args.hidden // 8), device=device, dtype=torch.int32
    )
    down_q = torch.randint(
        0, 255, (args.hidden, args.intermediate // 8), device=device, dtype=torch.int32
    )
    gate_s = (
        torch.rand(
            (args.intermediate, args.hidden // group_size),
            device=device,
            dtype=torch.float16,
        )
        * args.scale_factor
        + args.scale_floor
    )
    up_s = (
        torch.rand(
            (args.intermediate, args.hidden // group_size),
            device=device,
            dtype=torch.float16,
        )
        * args.scale_factor
        + args.scale_floor
    )
    down_s = (
        torch.rand(
            (args.hidden, args.intermediate // group_size),
            device=device,
            dtype=torch.float16,
        )
        * args.scale_factor
        + args.scale_floor
    )

    policy = {"awq_fused_gate_up": True, "awq_decode_gemv": True}
    h_fused, used, reason = packed_int4_symmetric_fused_gate_up_m1_safe(
        x,
        gate_q,
        up_q,
        gate_s,
        up_s,
        group_size,
        activation=args.activation,
        policy=policy,
    )
    if not used:
        raise RuntimeError(f"fused gate/up unavailable: {reason}")
    y_current = packed_int4_symmetric_fused_gemm(
        h_fused, down_q, down_s, group_size, policy={"awq_decode_gemv": True}
    )
    torch.cuda.synchronize()

    gate_w = dequantize_symmetric_packed_int4_pytorch(
        gate_q, gate_s, group_size=group_size
    )
    up_w = dequantize_symmetric_packed_int4_pytorch(up_q, up_s, group_size=group_size)
    down_w = dequantize_symmetric_packed_int4_pytorch(
        down_q, down_s, group_size=group_size
    )
    gate = F.linear(x.float(), gate_w.float())
    up = F.linear(x.float(), up_w.float())
    h_ref = _activation(gate, args.activation) * up
    y_ref = F.linear(h_ref, down_w.float()).to(dtype)
    torch.cuda.synchronize()

    diff = (y_current.float() - y_ref.float()).abs()
    return {
        "shape": {"m": 1, "hidden": args.hidden, "intermediate": args.intermediate},
        "dtype": args.dtype,
        "activation": args.activation,
        "gate_up_used": used,
        "gate_up_reason": reason,
        "max_diff": float(diff.max().item()),
        "mean_diff": float(diff.mean().item()),
        "cosine": _cosine(y_current, y_ref),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=5376)
    parser.add_argument("--intermediate", type=int, default=21504)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--activation", default="silu")
    parser.add_argument("--scale-factor", type=float, default=0.02)
    parser.add_argument("--scale-floor", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    payload = _run(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")


if __name__ == "__main__":
    main()
