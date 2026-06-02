# SPDX-License-Identifier: Apache-2.0
"""Benchmark Gemma4-31B AWQ down_proj M=1 decode configs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from vllm.kernels.triton.awq_fused_gemm import (
    packed_int4_symmetric_fused_gemm,
    set_awq_fused_tuning_config,
)


@dataclass(frozen=True)
class BenchResult:
    path: str
    block_n: int
    block_inner: int
    ms: float
    error: str = ""


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


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


def _bench_candidate(
    *,
    path: str,
    block_n: int,
    block_inner: int,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> BenchResult:
    group_size = 32
    m, n, k = 1, 5376, 21504
    torch.manual_seed(20260601 + block_n * 31 + block_inner)
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    qweight = torch.randint(0, 255, (n, k // 8), device=device, dtype=torch.int32)
    scales = (
        torch.randn((n, k // group_size), device=device, dtype=torch.float16).abs()
        + 0.01
    )

    tuning: dict[str, object] = {"FASTINFERENCE_AWQ_FUSED_GEMM_SPLIT_K": 1}
    policy: dict[str, object] = {"awq_decode_gemv": True}
    if path == "generic":
        tuning.update(
            {
                "FASTINFERENCE_AWQ_DECODE_GEMV_BLOCK_N": block_n,
                "FASTINFERENCE_AWQ_DECODE_GEMV_BLOCK_PACKS": block_inner,
            }
        )
    elif path == "group32":
        policy["awq_group32_gemv_all"] = True
        tuning.update(
            {
                "FASTINFERENCE_AWQ_GROUP32_GEMV_BLOCK_N": block_n,
                "FASTINFERENCE_AWQ_GROUP32_GEMV_BLOCK_GROUPS": block_inner,
            }
        )
    else:
        raise ValueError(f"unknown path: {path}")
    set_awq_fused_tuning_config(tuning, locked=False)

    def once() -> None:
        out = packed_int4_symmetric_fused_gemm(
            x,
            qweight,
            scales,
            group_size,
            policy=policy,
        )
        del out

    for _ in range(warmup):
        once()
    ms = _elapsed_ms(once, repeat)
    set_awq_fused_tuning_config({}, locked=False)
    return BenchResult(path=path, block_n=block_n, block_inner=block_inner, ms=ms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="generic,group32")
    parser.add_argument("--block-n", default="64,128,192,256,384")
    parser.add_argument("--generic-packs", default="8,16,32,64")
    parser.add_argument("--group32-groups", default="2,4,8,16")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=64)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device is required")
    device = torch.device("cuda")
    results: list[BenchResult] = []
    for path in [item.strip() for item in args.paths.split(",") if item.strip()]:
        inners = (
            _parse_csv_ints(args.generic_packs)
            if path == "generic"
            else _parse_csv_ints(args.group32_groups)
        )
        for block_n in _parse_csv_ints(args.block_n):
            for block_inner in inners:
                try:
                    results.append(
                        _bench_candidate(
                            path=path,
                            block_n=block_n,
                            block_inner=block_inner,
                            warmup=args.warmup,
                            repeat=args.repeat,
                            device=device,
                        )
                    )
                except Exception as exc:
                    results.append(
                        BenchResult(
                            path=path,
                            block_n=block_n,
                            block_inner=block_inner,
                            ms=math.inf,
                            error=type(exc).__name__ + ": " + str(exc),
                        )
                    )

    valid = [result for result in results if not result.error]
    payload: dict[str, Any] = {
        "results": [asdict(result) for result in results],
        "best": asdict(min(valid, key=lambda result: result.ms)) if valid else None,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")


if __name__ == "__main__":
    main()
