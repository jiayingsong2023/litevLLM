# SPDX-License-Identifier: Apache-2.0
"""Benchmark experimental AWQ group32 interleaved layout for M=1 GEMV."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch

from vllm.kernels.triton.awq_fused_gemm import (
    pack_awq_group32_interleaved_qweight_scales,
    packed_int4_symmetric_fused_gemm,
    packed_int4_symmetric_group32_interleaved_gemv_m1_safe,
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

    x = torch.randn((1, args.k), device=device, dtype=dtype)
    qweight = torch.randint(
        0, 255, (args.n, args.k // 8), device=device, dtype=torch.int32
    )
    scale_dtype = torch.bfloat16 if args.scale_dtype == "bfloat16" else torch.float16
    scales = (
        torch.rand((args.n, args.k // group_size), device=device, dtype=scale_dtype)
        * args.scale_factor
        + args.scale_floor
    )
    interleaved = pack_awq_group32_interleaved_qweight_scales(qweight, scales)
    policy = {"awq_decode_gemv": True, "awq_group32_gemv_all": True}

    def current_group32() -> None:
        y = packed_int4_symmetric_fused_gemm(
            x, qweight, scales, group_size, policy=policy
        )
        del y

    def interleaved_group32() -> None:
        y, used, reason = packed_int4_symmetric_group32_interleaved_gemv_m1_safe(
            x, interleaved, group_size, scale_dtype=scales.dtype
        )
        if not used:
            raise RuntimeError(reason)
        del y

    for _ in range(args.warmup):
        current_group32()
        interleaved_group32()

    results = [
        BenchResult(
            name="current_group32",
            status="ok",
            ms=_elapsed_ms(current_group32, args.repeat),
            note="separate qweight and scale tensors",
        ),
        BenchResult(
            name="interleaved_group32",
            status="ok",
            ms=_elapsed_ms(interleaved_group32, args.repeat),
            note="packed [N,K_groups,4 qpacks + scale_bits] int32 layout",
        ),
    ]
    current = next(r for r in results if r.name == "current_group32")
    candidate = next(r for r in results if r.name == "interleaved_group32")
    speedup = None
    if current.ms and candidate.ms:
        speedup = current.ms / candidate.ms
    return {
        "shape": {"m": 1, "n": args.n, "k": args.k, "group_size": group_size},
        "dtype": args.dtype,
        "scale_dtype": args.scale_dtype,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "results": [asdict(result) for result in results],
        "speedup_current_over_interleaved": speedup,
        "interleaved_raw_bytes_per_group": 20,
        "separate_raw_bytes_per_group": 18,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5376)
    parser.add_argument("--k", type=int, default=21504)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--scale-dtype", choices=("float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--scale-factor", type=float, default=0.02)
    parser.add_argument("--scale-floor", type=float, default=0.001)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260603)
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
