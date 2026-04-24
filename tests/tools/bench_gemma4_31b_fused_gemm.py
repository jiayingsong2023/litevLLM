#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Micro-benchmark packed INT4 fused GEMM on Gemma4-31B's real projection shapes.

This tool focuses on the dense Gemma4-31B decoder path and benchmarks:
- self-attn q/k/v/o projections for both local and global layers
- MLP gate/up/down projections

It is designed for Sprint 1 fused-GEMM tuning. The output is a JSON summary plus
human-readable best-per-case rows that can be used to solidify kernel defaults.

Example:
  uv run python tests/tools/bench_gemma4_31b_fused_gemm.py --quick
  uv run python tests/tools/bench_gemma4_31b_fused_gemm.py --decode-ms 1,2,4 --prefill-ms 128
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(max(0, warmup)):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(max(1, iters)):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / float(max(1, iters))


def _clear_block_env() -> None:
    for k in (
        "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_M",
        "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_N",
        "FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_K",
        "FASTINFERENCE_AWQ_FUSED_GEMM_NUM_WARPS",
        "FASTINFERENCE_AWQ_FUSED_GEMM_NUM_STAGES",
        "FASTINFERENCE_AWQ_FUSED_AUTOTUNE",
    ):
        os.environ.pop(k, None)


def _set_block_env(block_m: int, block_n: int, block_k: int, num_warps: int, num_stages: int) -> None:
    os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_M"] = str(block_m)
    os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_N"] = str(block_n)
    os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_BLOCK_K"] = str(block_k)
    os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_NUM_WARPS"] = str(num_warps)
    os.environ["FASTINFERENCE_AWQ_FUSED_GEMM_NUM_STAGES"] = str(num_stages)
    os.environ["FASTINFERENCE_AWQ_FUSED_AUTOTUNE"] = "0"


@dataclass(frozen=True)
class GemmShape:
    label: str
    m: int
    n: int
    k: int
    group_size: int


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_gemma4_31b_shapes(model_path: str) -> list[GemmShape]:
    cfg = json.loads(Path(model_path, "config.json").read_text(encoding="utf-8"))
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    hidden = int(text_cfg["hidden_size"])
    intermediate = int(text_cfg["intermediate_size"])
    num_heads = int(text_cfg["num_attention_heads"])
    num_kv_heads = int(text_cfg["num_key_value_heads"])
    local_head_dim = int(text_cfg["head_dim"])
    global_head_dim = int(text_cfg["global_head_dim"])
    num_global_kv_heads = int(text_cfg["num_global_key_value_heads"])
    compression = cfg.get("compression_config") or cfg.get("quantization_config") or {}
    group_size = int(compression.get("group_size", 128))
    groups = compression.get("config_groups")
    if isinstance(groups, dict):
        for g in groups.values():
            if not isinstance(g, dict):
                continue
            w = g.get("weights")
            if isinstance(w, dict) and w.get("group_size") is not None:
                group_size = int(w["group_size"])
                break

    local_q = num_heads * local_head_dim
    local_kv = num_kv_heads * local_head_dim
    global_q = num_heads * global_head_dim
    global_kv = num_global_kv_heads * global_head_dim

    return [
        GemmShape("attn_local_q_proj", 0, local_q, hidden, group_size),
        GemmShape("attn_local_k_proj", 0, local_kv, hidden, group_size),
        GemmShape("attn_local_v_proj", 0, local_kv, hidden, group_size),
        GemmShape("attn_local_o_proj", 0, hidden, local_q, group_size),
        GemmShape("attn_global_q_proj", 0, global_q, hidden, group_size),
        GemmShape("attn_global_k_proj", 0, global_kv, hidden, group_size),
        GemmShape("attn_global_v_proj", 0, global_kv, hidden, group_size),
        GemmShape("attn_global_o_proj", 0, hidden, global_q, group_size),
        GemmShape("mlp_gate_proj", 0, intermediate, hidden, group_size),
        GemmShape("mlp_up_proj", 0, intermediate, hidden, group_size),
        GemmShape("mlp_down_proj", 0, hidden, intermediate, group_size),
    ]


def expand_shapes_by_m(
    base_shapes: list[GemmShape],
    *,
    decode_ms: list[int],
    prefill_ms: list[int],
) -> list[GemmShape]:
    out: list[GemmShape] = []
    for shape in base_shapes:
        for m in decode_ms:
            out.append(GemmShape(f"{shape.label}/decode_m{m}", m, shape.n, shape.k, shape.group_size))
        for m in prefill_ms:
            out.append(GemmShape(f"{shape.label}/prefill_m{m}", m, shape.n, shape.k, shape.group_size))
    return out


def candidate_configs() -> list[tuple[str, tuple[int, int, int, int, int] | None]]:
    return [
        ("autotune_default", None),
        ("bm16_bn128_bk64_w8_s2", (16, 128, 64, 8, 2)),
        ("bm16_bn256_bk64_w8_s2", (16, 256, 64, 8, 2)),
        ("bm32_bn128_bk64_w8_s2", (32, 128, 64, 8, 2)),
        ("bm32_bn256_bk64_w8_s2", (32, 256, 64, 8, 2)),
        ("bm64_bn64_bk64_w8_s2", (64, 64, 64, 8, 2)),
        ("bm64_bn128_bk64_w8_s2", (64, 128, 64, 8, 2)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Gemma4-31B packed INT4 fused GEMM")
    parser.add_argument("--model", type=str, default="models/gemma-4-31B-it-AWQ-4bit")
    parser.add_argument("--decode-ms", type=str, default="1,2,4")
    parser.add_argument("--prefill-ms", type=str, default="128,256")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--quick", action="store_true", help="Reduce shapes and iterations for a fast sanity run.")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device required.")
    if not os.path.isdir(args.model):
        raise SystemExit(f"Model path not found: {args.model}")

    from vllm.kernels.triton.awq_fused_gemm import (
        _resolve_use_bf16_dot,
        _select_fused_gemm_blocks,
        packed_int4_symmetric_fused_gemm,
    )

    decode_ms = _parse_int_list(args.decode_ms)
    prefill_ms = _parse_int_list(args.prefill_ms)
    if args.quick:
        decode_ms = [1, 4]
        prefill_ms = [128]
        args.warmup = min(args.warmup, 8)
        args.iters = min(args.iters, 20)

    base_shapes = load_gemma4_31b_shapes(args.model)
    if args.quick:
        base_shapes = [
            s for s in base_shapes
            if s.label in ("attn_local_q_proj", "attn_global_q_proj", "mlp_gate_proj", "mlp_down_proj")
        ]
    shapes = expand_shapes_by_m(base_shapes, decode_ms=decode_ms, prefill_ms=prefill_ms)

    device = torch.device("cuda")
    results: list[dict[str, object]] = []

    for shape in shapes:
        _clear_block_env()
        a = torch.randn(shape.m, shape.k, device=device, dtype=torch.bfloat16)
        qweight = torch.randint(0, 255, (shape.n, shape.k // 8), device=device, dtype=torch.uint8)
        scales = torch.ones(shape.n, shape.k // shape.group_size, device=device, dtype=torch.float16)
        out = torch.empty(shape.m, shape.n, device=device, dtype=a.dtype)
        heur = _select_fused_gemm_blocks(shape.m, shape.n, shape.k)
        use_bf16_dot = _resolve_use_bf16_dot(a, shape.m, shape.n)

        case_rows: list[dict[str, object]] = []
        for cfg_name, cfg in [("heuristic_env", heur)] + candidate_configs():
            _clear_block_env()
            if cfg is not None:
                _set_block_env(*cfg)
            else:
                os.environ["FASTINFERENCE_AWQ_FUSED_AUTOTUNE"] = "1"

            def run_once() -> None:
                packed_int4_symmetric_fused_gemm(
                    a,
                    qweight,
                    scales,
                    shape.group_size,
                    out=out,
                )

            ms = _bench(run_once, warmup=args.warmup, iters=args.iters)
            row = {
                "config": cfg_name,
                "ms": ms,
                "tflops_est": (2.0 * shape.m * shape.n * shape.k) / (ms / 1000.0) / 1e12,
            }
            case_rows.append(row)

        best = min(case_rows, key=lambda r: float(r["ms"]))
        results.append(
            {
                "shape": asdict(shape),
                "heuristic_blocks": heur,
                "use_bf16_dot": use_bf16_dot,
                "best": best,
                "rows": case_rows,
            }
        )
        print(
            f"{shape.label:24s} M={shape.m:4d} N={shape.n:5d} K={shape.k:5d} "
            f"heuristic={heur} best={best['config']} {float(best['ms']):8.3f} ms"
        )

    summary = {
        "model": args.model,
        "decode_ms": decode_ms,
        "prefill_ms": prefill_ms,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }

    print("[Gemma4FusedGemmBench] " + json.dumps(summary, ensure_ascii=True))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
