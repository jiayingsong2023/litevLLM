# SPDX-License-Identifier: Apache-2.0
"""Benchmark Gemma4-31B M=1 AWQ MLP current path and P2 candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch

from vllm.kernels.triton.awq_fused_gemm import (
    packed_int4_symmetric_fused_gate_up_m1_safe,
    packed_int4_symmetric_fused_gemm,
    packed_int4_symmetric_mlp_streaming_m1_recompute_safe,
)


@dataclass(frozen=True)
class BenchResult:
    name: str
    status: str
    ms: float | None
    note: str = ""


def _elapsed_ms(fn, repeat: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / max(repeat, 1)


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

    gate_policy = {"awq_fused_gate_up": True, "awq_decode_gemv": True}
    down_policy = {"awq_decode_gemv": True}

    def current_two_stage() -> None:
        h, used, reason = packed_int4_symmetric_fused_gate_up_m1_safe(
            x,
            gate_q,
            up_q,
            gate_s,
            up_s,
            group_size,
            activation=args.activation,
            policy=gate_policy,
        )
        if not used:
            raise RuntimeError(reason)
        y = packed_int4_symmetric_fused_gemm(
            h, down_q, down_s, group_size, policy=down_policy
        )
        del y

    def recompute_streaming() -> None:
        y, used, reason = packed_int4_symmetric_mlp_streaming_m1_recompute_safe(
            x,
            gate_q,
            up_q,
            down_q,
            gate_s,
            up_s,
            down_s,
            group_size,
            activation=args.activation,
        )
        if not used:
            raise RuntimeError(reason)
        del y

    for _ in range(args.warmup):
        current_two_stage()
    recompute_result: BenchResult
    try:
        for _ in range(args.warmup):
            recompute_streaming()
        recompute_result = BenchResult(
            name="candidate_streaming_recompute",
            status="ok",
            ms=_elapsed_ms(recompute_streaming, args.repeat),
            note="negative-control single kernel; recomputes gate/up per down tile",
        )
    except Exception as exc:
        recompute_result = BenchResult(
            name="candidate_streaming_recompute",
            status="error",
            ms=None,
            note=type(exc).__name__ + ": " + str(exc),
        )

    results = [
        BenchResult(
            name="current_two_stage",
            status="ok",
            ms=_elapsed_ms(current_two_stage, args.repeat),
            note="fused gate/up activation + tuned down_proj GEMV",
        ),
        BenchResult(
            name="candidate_half_fusion",
            status="not_implemented",
            ms=None,
            note=(
                "requires a concrete intermediate layout candidate before benchmarking"
            ),
        ),
        recompute_result,
        BenchResult(
            name="candidate_streaming_cooperative",
            status="not_implemented",
            ms=None,
            note=(
                "requires cooperative streaming kernel that does not recompute "
                "gate/up per down tile"
            ),
        ),
    ]
    return {
        "shape": {"m": 1, "hidden": args.hidden, "intermediate": args.intermediate},
        "dtype": args.dtype,
        "activation": args.activation,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "results": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=5376)
    parser.add_argument("--intermediate", type=int, default=21504)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--activation", default="silu")
    parser.add_argument("--scale-factor", type=float, default=0.02)
    parser.add_argument("--scale-floor", type=float, default=0.001)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=64)
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
