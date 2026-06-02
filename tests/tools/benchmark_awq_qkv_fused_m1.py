# SPDX-License-Identifier: Apache-2.0
"""Benchmark Gemma4-31B AWQ fused QKV/QK decode launch configs.

This tool sweeps the exact M=1 shapes seen in Gemma4Attention decode:

- local QKV: hidden=5376, q=8192, k=4096, v=4096
- global QK: hidden=5376, q=16384, k=2048, v is reused from k
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from vllm.kernels.triton.awq_fused_gemm import (
    packed_int4_symmetric_fused_gemm,
    packed_int4_symmetric_fused_qkv_m1_safe,
    set_awq_fused_tuning_config,
)


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    k: int
    qn: int
    kn: int
    vn: int


@dataclass(frozen=True)
class BenchResult:
    shape: str
    block_n: int
    block_groups: int
    fused_ms: float
    separate_ms: float
    speedup_vs_separate: float
    error: str = ""


SHAPES = (
    ShapeSpec("local_qkv", k=5376, qn=8192, kn=4096, vn=4096),
    ShapeSpec("global_qk", k=5376, qn=16384, kn=2048, vn=0),
)


def _rand_weight(n: int, k: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, 255, (n, k // 8), device=device, dtype=torch.int32)


def _rand_scales(n: int, k: int, group_size: int, device: torch.device) -> torch.Tensor:
    return (
        torch.randn((n, k // group_size), device=device, dtype=torch.float16).abs()
        + 0.01
    )


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


def _bench_shape(
    spec: ShapeSpec,
    *,
    block_n: int,
    block_groups: int,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> BenchResult:
    group_size = 32
    torch.manual_seed(20260601 + block_n * 17 + block_groups)
    x = torch.randn((1, spec.k), device=device, dtype=torch.bfloat16)
    qweight = _rand_weight(spec.qn, spec.k, device)
    kweight = _rand_weight(spec.kn, spec.k, device)
    vweight = _rand_weight(spec.vn, spec.k, device) if spec.vn else None
    qscales = _rand_scales(spec.qn, spec.k, group_size, device)
    kscales = _rand_scales(spec.kn, spec.k, group_size, device)
    vscales = _rand_scales(spec.vn, spec.k, group_size, device) if spec.vn else None
    policy = {"awq_decode_gemv": True}

    set_awq_fused_tuning_config(
        {
            "FASTINFERENCE_AWQ_QKV_GROUP32_GEMV_BLOCK_N": block_n,
            "FASTINFERENCE_AWQ_QKV_GROUP32_GEMV_BLOCK_GROUPS": block_groups,
        },
        locked=False,
    )

    def fused_once() -> None:
        out, used, reason = packed_int4_symmetric_fused_qkv_m1_safe(
            x,
            qweight,
            kweight,
            vweight,
            qscales,
            kscales,
            vscales,
            group_size,
            policy=policy,
        )
        if not used:
            raise RuntimeError(reason)
        del out

    def separate_once() -> None:
        yq = packed_int4_symmetric_fused_gemm(x, qweight, qscales, group_size)
        yk = packed_int4_symmetric_fused_gemm(x, kweight, kscales, group_size)
        if vweight is not None and vscales is not None:
            yv = packed_int4_symmetric_fused_gemm(x, vweight, vscales, group_size)
            del yv
        del yq, yk

    for _ in range(warmup):
        fused_once()
        separate_once()
    fused_ms = _elapsed_ms(fused_once, repeat)
    separate_ms = _elapsed_ms(separate_once, repeat)
    set_awq_fused_tuning_config({}, locked=False)
    return BenchResult(
        shape=spec.name,
        block_n=block_n,
        block_groups=block_groups,
        fused_ms=fused_ms,
        separate_ms=separate_ms,
        speedup_vs_separate=separate_ms / fused_ms if fused_ms > 0 else math.nan,
    )


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-n", default="128,192,256,384")
    parser.add_argument("--block-groups", default="2,4,8")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=64)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device is required")
    device = torch.device("cuda")
    results: list[BenchResult] = []
    for spec in SHAPES:
        for block_n in _parse_csv_ints(args.block_n):
            for block_groups in _parse_csv_ints(args.block_groups):
                try:
                    results.append(
                        _bench_shape(
                            spec,
                            block_n=block_n,
                            block_groups=block_groups,
                            warmup=args.warmup,
                            repeat=args.repeat,
                            device=device,
                        )
                    )
                except Exception as exc:
                    results.append(
                        BenchResult(
                            shape=spec.name,
                            block_n=block_n,
                            block_groups=block_groups,
                            fused_ms=math.inf,
                            separate_ms=math.inf,
                            speedup_vs_separate=math.nan,
                            error=type(exc).__name__ + ": " + str(exc),
                        )
                    )

    payload: dict[str, Any] = {
        "results": [asdict(result) for result in results],
        "best": {
            spec.name: asdict(
                min(
                    (
                        result
                        for result in results
                        if result.shape == spec.name and not result.error
                    ),
                    key=lambda result: result.fused_ms,
                )
            )
            for spec in SHAPES
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")


if __name__ == "__main__":
    main()
