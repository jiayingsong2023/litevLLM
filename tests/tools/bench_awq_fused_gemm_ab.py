# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark AWQ fused packed-int4 GEMM variants (A/B kernel tuning).

Run: uv run python tests/bench_awq_fused_gemm_ab.py
"""
from __future__ import annotations

import os
import time
from typing import Callable

import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(
    label: str,
    fn: Callable[[], None],
    *,
    warmup: int = 40,
    iters: int = 120,
) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    elapsed = time.perf_counter() - t0
    return elapsed * 1000.0 / float(iters)


def _clear_block_env() -> None:
    for k in (
        "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_M",
        "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_N",
        "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_K",
        "FASTINFERENCE_AWQ_FUSED_GEMM_NUM_WARPS",
        "FASTINFERENCE_AWQ_FUSED_GEMM_NUM_STAGES",
    ):
        os.environ.pop(k, None)


def _set_block_env(m: str, n: str, k: str, w: str, s: str) -> None:
    os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_M"] = m
    os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_N"] = n
    os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_K"] = k
    os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_NUM_WARPS"] = w
    os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_NUM_STAGES"] = s


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA/ROCm not available; skip.")
        return

    from vllm.kernels.triton.awq_fused_gemm import (
        _resolve_use_bf16_dot,
        _select_fused_gemm_blocks,
        packed_int4_symmetric_fused_gemm,
    )

    device = torch.device("cuda")
    group_size = 128
    cases = [
        ("decode_like", 1, 4096, 4096),
        ("decode_wide_n", 1, 14336, 4096),
        ("small_batch", 8, 4096, 4096),
        ("prefill_chunk", 256, 4096, 4096),
    ]

    rows: list[tuple[str, str, float, str]] = []

    for case_name, m, n, k in cases:
        _clear_block_env()
        os.environ.pop("FASTINFERENCE_AWQ_FUSED_GEMM_DOT_BF16", None)

        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        qw = torch.randint(0, 255, (n, k // 8), device=device, dtype=torch.uint8)
        sc = torch.ones(n, k // group_size, device=device, dtype=torch.float16)
        out = torch.empty(m, n, device=device, dtype=a.dtype)

        heur = _select_fused_gemm_blocks(m, n, k)
        print(f"\n=== {case_name} M={m} N={n} K={k} heuristic={heur} ===")

        for label, dot, legacy in (
            ("auto_dot_env", "auto", False),
            ("heuristic_bf16_dot", "1", False),
            ("heuristic_fp16_dot", "0", False),
            ("legacy64_bf16_dot", "1", True),
            ("legacy64_fp16_dot", "0", True),
        ):
            _clear_block_env()
            if dot == "auto":
                os.environ.pop("FASTINFERENCE_AWQ_FUSED_GEMM_DOT_BF16", None)
            else:
                os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_DOT_BF16"] = dot
            if legacy:
                _set_block_env("64", "64", "32", "4", "2")

            def run_once() -> None:
                packed_int4_symmetric_fused_gemm(
                    a, qw, sc, group_size, out=out
                )

            ms = _bench(label, run_once)
            want = _resolve_use_bf16_dot(a, m, n)
            rows.append((case_name, label, ms, f"use_bf16_dot={want}"))
            print(
                f"  {label:22s}  {ms:8.3f} ms/it  "
                f"(resolve_use_bf16_dot={want})"
            )

    print("\n" + "=" * 72)
    print("SUMMARY (ms per iteration, lower is better)")
    print("=" * 72)
    for case_name, label, ms, _ in rows:
        print(f"{case_name:16} {label:22} {ms:8.3f}")


if __name__ == "__main__":
    main()
