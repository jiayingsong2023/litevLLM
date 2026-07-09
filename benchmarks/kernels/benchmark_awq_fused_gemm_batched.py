# SPDX-License-Identifier: Apache-2.0
"""Baseline microbench for AWQ fused QKV / gate-up.

This tool measures the M=1 fast path first. Future work will add M=2/M=4
batched variants here and compare against the M=1 baseline before any
model-side wiring is changed.
"""

from __future__ import annotations

import argparse
import csv
import time

import torch

from vllm.kernels.triton.awq_fused_gemm import (
    packed_int4_symmetric_fused_gate_up_m1_safe,
    packed_int4_symmetric_fused_qkv_m1_safe,
)

# The safe wrappers gate the fast path behind policy flags that default to
# False in standalone tools. Enable them explicitly for this baseline.
_DEFAULT_POLICY: dict[str, object] = {
    "awq_decode_gemv": True,
    "awq_fused_gate_up": True,
}


def _make_qkv_weights(
    hidden: int,
    q_size: int,
    kv_size: int,
    group_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    q_qweight = torch.randint(
        0, 16, (q_size, hidden // 8), dtype=torch.int32, device=device
    )
    q_scales = torch.randn(
        q_size, hidden // group_size, dtype=torch.bfloat16, device=device
    )
    k_qweight = torch.randint(
        0, 16, (kv_size, hidden // 8), dtype=torch.int32, device=device
    )
    k_scales = torch.randn(
        kv_size, hidden // group_size, dtype=torch.bfloat16, device=device
    )
    v_qweight = torch.randint(
        0, 16, (kv_size, hidden // 8), dtype=torch.int32, device=device
    )
    v_scales = torch.randn(
        kv_size, hidden // group_size, dtype=torch.bfloat16, device=device
    )
    return q_qweight, k_qweight, v_qweight, q_scales, k_scales, v_scales


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _bench_qkv_m1(
    hidden: int,
    q_size: int,
    kv_size: int,
    group_size: int,
    device: torch.device,
) -> dict[str, float]:
    x = torch.randn(1, hidden, dtype=torch.bfloat16, device=device)
    weights = _make_qkv_weights(hidden, q_size, kv_size, group_size, device)
    out, used, reason = packed_int4_symmetric_fused_qkv_m1_safe(
        x, *weights, group_size, policy=_DEFAULT_POLICY
    )
    assert used, f"QKV M=1 not used: {reason}"

    _sync(device)
    start = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        out, _, _ = packed_int4_symmetric_fused_qkv_m1_safe(
            x, *weights, group_size, policy=_DEFAULT_POLICY
        )
    _sync(device)
    return {
        "kernel": "qkv_m1",
        "m": 1,
        "time_ms": (time.perf_counter() - start) * 1000 / iterations,
    }


def _bench_gate_up_m1(
    hidden: int,
    intermediate: int,
    group_size: int,
    device: torch.device,
) -> dict[str, float]:
    x = torch.randn(1, hidden, dtype=torch.bfloat16, device=device)
    qwg = torch.randint(
        0, 16, (intermediate, hidden // 8), dtype=torch.int32, device=device
    )
    gate_scales = torch.randn(
        intermediate, hidden // group_size, dtype=torch.bfloat16, device=device
    )
    qwu = torch.randint(
        0, 16, (intermediate, hidden // 8), dtype=torch.int32, device=device
    )
    up_scales = torch.randn(
        intermediate, hidden // group_size, dtype=torch.bfloat16, device=device
    )
    out, used, reason = packed_int4_symmetric_fused_gate_up_m1_safe(
        x,
        qwg,
        qwu,
        gate_scales,
        up_scales,
        group_size,
        activation="silu",
        policy=_DEFAULT_POLICY,
    )
    assert used, f"Gate-up M=1 not used: {reason}"

    _sync(device)
    start = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        out, _, _ = packed_int4_symmetric_fused_gate_up_m1_safe(
            x,
            qwg,
            qwu,
            gate_scales,
            up_scales,
            group_size,
            activation="silu",
            policy=_DEFAULT_POLICY,
        )
    _sync(device)
    return {
        "kernel": "gate_up_m1",
        "m": 1,
        "time_ms": (time.perf_counter() - start) * 1000 / iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2816)
    parser.add_argument("--q-size", type=int, default=2816)
    parser.add_argument("--kv-size", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=704)
    # The M=1 AWQ fast path requires group_size == 32 and K % 32 == 0.
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--out", default="/tmp/gemma26b_awq_baseline_microbench.csv")
    args = parser.parse_args()

    if args.group_size != 32:
        raise ValueError(
            f"M=1 AWQ fast path requires group_size == 32, got {args.group_size}"
        )
    for name in ("hidden", "q_size", "kv_size", "intermediate"):
        value = getattr(args, name)
        if value % 32 != 0:
            raise ValueError(
                f"{name}={value} is not a multiple of 32; the packed-int4 "
                "M=1 path requires K and N dimensions to be group-aligned."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [
        _bench_qkv_m1(args.hidden, args.q_size, args.kv_size, args.group_size, device),
        _bench_gate_up_m1(args.hidden, args.intermediate, args.group_size, device),
    ]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kernel", "m", "time_ms"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
