#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Build persistent AWQ fused profile JSON from micro-benchmark output.

Usage:
  uv run python tests/tools/build_awq_fused_profile.py \
    --bench-json tests/reports/gemma31b_fused_bench_for_profile.json \
    --out vllm/kernels/triton/awq_fused_profile.generated.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build AWQ fused persistent profile from bench JSON")
    p.add_argument("--bench-json", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument(
        "--only-label-substr",
        type=str,
        default="",
        help="Optional substring filter on shape label (e.g. decode_m).",
    )
    return p.parse_args()


def _pick_best_non_autotune(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for row in rows:
        cfg = str(row.get("config", ""))
        if cfg == "autotune_default":
            continue
        ms = float(row.get("ms", float("inf")))
        if best is None or ms < float(best.get("ms", float("inf"))):
            best = row
    return best


def _cfg_tuple(row: dict[str, Any], heur: list[int]) -> tuple[int, int, int, int, int]:
    cfg = str(row.get("config", ""))
    if cfg == "heuristic_env":
        return int(heur[0]), int(heur[1]), int(heur[2]), int(heur[3]), int(heur[4])
    parts = cfg.split("_")
    # expected e.g. bm16_bn256_bk64_w8_s2
    vals = {p[:2]: int(p[2:]) for p in parts if len(p) >= 3 and p[:2] in ("bm", "bn", "bk")}
    w = next((int(p[1:]) for p in parts if p.startswith("w")), int(heur[3]))
    s = next((int(p[1:]) for p in parts if p.startswith("s")), int(heur[4]))
    return int(vals.get("bm", heur[0])), int(vals.get("bn", heur[1])), int(vals.get("bk", heur[2])), w, s


def main() -> int:
    args = _parse_args()
    payload = json.loads(Path(args.bench_json).read_text(encoding="utf-8"))
    out_rows: list[dict[str, int]] = []
    for item in payload.get("results", []):
        shape = item.get("shape") or {}
        label = str(shape.get("label", ""))
        if args.only_label_substr and args.only_label_substr not in label:
            continue
        rows = item.get("rows") or []
        heur = item.get("heuristic_blocks") or []
        if not isinstance(rows, list) or len(heur) < 5:
            continue
        best = _pick_best_non_autotune(rows)
        if best is None:
            continue
        bm, bn, bk, nw, ns = _cfg_tuple(best, heur)
        out_rows.append(
            {
                "m_min": int(shape.get("m", 1)),
                "m_max": int(shape.get("m", 1)),
                "n": int(shape.get("n", 1)),
                "k": int(shape.get("k", 1)),
                "group_size": int(args.group_size),
                "block_m": bm,
                "block_n": bn,
                "block_k": bk,
                "num_warps": nw,
                "num_stages": ns,
            }
        )

    out = {"version": 1, "packed_int4_symmetric": out_rows}
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[build_awq_fused_profile] wrote {out_path} with {len(out_rows)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

