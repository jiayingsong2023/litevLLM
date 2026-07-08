# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark Gemma4-26B MoE int4 decode kernels."""

from __future__ import annotations

import argparse
import csv
import time

import torch

from vllm.kernels.triton.gemma4_moe_int4 import (
    gemma4_moe_int4_decode,
    gemma4_moe_int4_decode_batched,
    gemma4_moe_int4_decode_batched_chunked,
    gemma4_moe_int4_decode_batched_grouped,
    gemma4_moe_int4_decode_batched_tuned,
)


def _reference_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    qweight_gu: torch.Tensor,
    scales_gu: torch.Tensor,
    qweight_d: torch.Tensor,
    scales_d: torch.Tensor,
    intermediate_dim: int,
) -> torch.Tensor:
    """PyTorch reference: dequantize int4 -> matmul per expert -> reduce."""
    from vllm.model_executor.layers.quantization.tensor import (
        dequantize_symmetric_packed_int4_pytorch,
    )

    m, hidden = x.shape
    top_k = topk_ids.shape[1]
    out = torch.zeros(m, hidden, dtype=torch.float32, device=x.device)
    for tok in range(m):
        for k in range(top_k):
            eid = int(topk_ids[tok, k])
            w1e = dequantize_symmetric_packed_int4_pytorch(
                qweight_gu[eid : eid + 1].to(torch.int32),
                scales_gu[eid : eid + 1],
                group_size=hidden // scales_gu.shape[2],
            )[0, : 2 * intermediate_dim, :hidden].to(torch.float32)
            w2e = dequantize_symmetric_packed_int4_pytorch(
                qweight_d[eid : eid + 1].to(torch.int32),
                scales_d[eid : eid + 1],
                group_size=intermediate_dim // scales_d.shape[2],
            )[0, :hidden, :intermediate_dim].to(torch.float32)

            h = x[tok : tok + 1].to(torch.float32)
            gu = torch.matmul(h, w1e.t())
            g, u = gu.chunk(2, dim=-1)
            h = torch.nn.functional.silu(g) * u
            y = torch.matmul(h, w2e.t()) * topk_weights[tok, k].to(torch.float32)
            out[tok] += y.squeeze(0)
    return out


def benchmark_strategy(
    strategy: str,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    batch_sizes: list[int],
    device: torch.device,
    check_correctness: bool = False,
) -> list[dict[str, float]]:
    kernel_map = {
        "two_stage": gemma4_moe_int4_decode,
        "batched": gemma4_moe_int4_decode_batched,
        "batched_chunked": gemma4_moe_int4_decode_batched_chunked,
        "batched_grouped": gemma4_moe_int4_decode_batched_grouped,
        "batched_tuned": gemma4_moe_int4_decode_batched_tuned,
    }
    kernel = kernel_map[strategy]
    results: list[dict[str, float]] = []

    qweight_gu = torch.randint(
        0,
        16,
        (num_experts, 2 * intermediate_dim, hidden_dim // 8),
        dtype=torch.int32,
        device=device,
    )
    scales_gu = torch.randn(
        num_experts,
        2 * intermediate_dim,
        hidden_dim // 32,
        dtype=torch.bfloat16,
        device=device,
    )
    qweight_d = torch.randint(
        0,
        16,
        (num_experts, hidden_dim, intermediate_dim // 8),
        dtype=torch.int32,
        device=device,
    )
    scales_d = torch.randn(
        num_experts,
        hidden_dim,
        intermediate_dim // 32,
        dtype=torch.bfloat16,
        device=device,
    )

    for m in batch_sizes:
        x = torch.randn(m, hidden_dim, dtype=torch.bfloat16, device=device)
        topk_ids = torch.randint(0, num_experts, (m, top_k), device=device)
        topk_weights = torch.randn(m, top_k, dtype=torch.bfloat16, device=device)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out, used, _ = kernel(
            x,
            topk_weights,
            topk_ids,
            qweight_gu,
            scales_gu,
            qweight_d,
            scales_d,
            intermediate_dim=intermediate_dim,
            activation="silu",
        )
        if not used:
            continue

        max_err = float("nan")
        if check_correctness:
            ref = _reference_moe(
                x,
                topk_ids,
                topk_weights,
                qweight_gu,
                scales_gu,
                qweight_d,
                scales_d,
                intermediate_dim,
            )
            max_err = (out.to(torch.float32) - ref).abs().max().item()
            try:
                torch.testing.assert_close(
                    out.to(torch.float32), ref, rtol=3e-2, atol=3e-2
                )
            except AssertionError as exc:
                print(
                    f"WARN {strategy} M={m} numerical check failed "
                    f"(max_err={max_err}): {exc}"
                )

        torch.cuda.synchronize()
        start = time.perf_counter()
        iterations = 100 if m == 1 else 50
        for _ in range(iterations):
            out, used, _ = kernel(
                x,
                topk_weights,
                topk_ids,
                qweight_gu,
                scales_gu,
                qweight_d,
                scales_d,
                intermediate_dim=intermediate_dim,
                activation="silu",
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        results.append(
            {
                "strategy": strategy,
                "m": m,
                "time_ms": elapsed * 1000 / iterations,
                "used": int(used),
                "max_err": max_err,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2816)
    parser.add_argument("--intermediate", type=int, default=704)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--out", default="/tmp/gemma26b_moe_microbench.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    strategies = [
        "two_stage",
        "batched",
        "batched_chunked",
        "batched_grouped",
        "batched_tuned",
    ]

    all_results: list[dict[str, float]] = []
    for strategy in strategies:
        all_results.extend(
            benchmark_strategy(
                strategy,
                args.hidden,
                args.intermediate,
                args.num_experts,
                args.top_k,
                batch_sizes,
                device,
                check_correctness=args.check_correctness,
            )
        )

    out_path = args.out
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["strategy", "m", "time_ms", "used", "max_err"]
        )
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
