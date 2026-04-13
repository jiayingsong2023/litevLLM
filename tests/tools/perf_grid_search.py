#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Day-2 performance grid search automation:
- batch grid benchmark runs (coarse + optional fine)
- automatic ranking/selection
- artifact archival (per-run JSON/stdout/stderr + leaderboard + best config)

Default target is Gemma4-26B A4B text workload with TurboQuant INT4 KV.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GridConfig:
    name: str
    prefill_chunk: int
    prefill_microbatch: int
    prefill_reserved_tokens: int
    prefill_reserve_backlog: int
    prefill_catchup_ratio: float
    decode_priority: int = 1
    kv_type: str = "turbo_int4"
    fusion_level: int = 2
    kv_max_active_requests: int | None = None
    kv_max_model_len: int | None = None


def _default_baseline_config() -> GridConfig:
    return GridConfig(
        name="baseline_c256_m2_r0_b2",
        prefill_chunk=256,
        prefill_microbatch=2,
        prefill_reserved_tokens=0,
        prefill_reserve_backlog=2,
        prefill_catchup_ratio=0.25,
    )


def build_coarse_grid() -> list[GridConfig]:
    out: list[GridConfig] = []
    for c in (128, 256, 384, 512):
        for m in (1, 2):
            out.append(
                GridConfig(
                    name=f"coarse_c{c}_m{m}",
                    prefill_chunk=c,
                    prefill_microbatch=m,
                    prefill_reserved_tokens=0,
                    prefill_reserve_backlog=2,
                    prefill_catchup_ratio=0.25,
                )
            )
    # Ensure preferred baseline appears first for stable comparisons.
    baseline = _default_baseline_config()
    return [baseline] + [x for x in out if x.prefill_chunk != 256 or x.prefill_microbatch != 2]


def build_fine_grid(top: list[GridConfig]) -> list[GridConfig]:
    out: list[GridConfig] = []
    seen: set[tuple[int, int, int, int]] = set()
    for base in top:
        for reserved in (0, 128):
            for backlog in (1, 2):
                key = (
                    base.prefill_chunk,
                    base.prefill_microbatch,
                    reserved,
                    backlog,
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    GridConfig(
                        name=f"fine_c{base.prefill_chunk}_m{base.prefill_microbatch}_r{reserved}_b{backlog}",
                        prefill_chunk=base.prefill_chunk,
                        prefill_microbatch=base.prefill_microbatch,
                        prefill_reserved_tokens=reserved,
                        prefill_reserve_backlog=backlog,
                        prefill_catchup_ratio=0.25,
                    )
                )
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day-2 perf grid search automation")
    p.add_argument(
        "--model-key",
        type=str,
        default="gemma4_26b_a4b",
        choices=("gemma4_26b_a4b", "gemma4_31b_q4", "qwen35_9b_awq", "tinyllama"),
        help="Model key for tests/e2e_full_benchmark.py --models",
    )
    p.add_argument("--work-dir", type=str, default="tests/reports/day2_grid")
    p.add_argument(
        "--phase",
        type=str,
        default="full",
        choices=("coarse", "full"),
        help="coarse: only 8 coarse runs; full: coarse + fine(top2)",
    )
    p.add_argument("--top-k-fine-seed", type=int, default=2)
    p.add_argument(
        "--ttft-degrade-limit",
        type=float,
        default=0.08,
        help="Reject configs whose ttft_p50 exceeds baseline by this ratio.",
    )
    p.add_argument(
        "--run-correctness-on-best",
        action="store_true",
        help="Run correctness regression once on selected best config.",
    )
    p.add_argument(
        "--correctness-skip-a-tier",
        action="store_true",
        help="When running correctness, set SKIP_A_TIER=1.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _run_cmd(
    *,
    cmd: list[str],
    env: dict[str, str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    return int(proc.returncode)


def _score_row(row: dict[str, Any], baseline_ttft: float, ttft_degrade_limit: float) -> tuple[bool, str]:
    if row.get("ok") is not True:
        return False, "run_failed"
    if row.get("timed_out"):
        return False, "timed_out"
    if row.get("skipped"):
        return False, "skipped"
    ttft = _safe_float(row.get("ttft_p50_ms"))
    if baseline_ttft == baseline_ttft and ttft == ttft:
        if ttft > baseline_ttft * (1.0 + ttft_degrade_limit):
            return False, "ttft_regressed"
    return True, "pass"


def _run_one(
    *,
    root: Path,
    run_dir: Path,
    model_key: str,
    cfg: GridConfig,
    dry_run: bool,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_json = run_dir / "e2e_summary.json"
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"

    env = os.environ.copy()
    env["FASTINFERENCE_KV_TYPE"] = cfg.kv_type
    env["FASTINFERENCE_FUSION_LEVEL"] = str(cfg.fusion_level)
    if cfg.kv_max_active_requests is None:
        env.pop("FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS", None)
    else:
        env["FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS"] = str(cfg.kv_max_active_requests)
    if cfg.kv_max_model_len is None:
        env.pop("FASTINFERENCE_KV_MAX_MODEL_LEN", None)
    else:
        env["FASTINFERENCE_KV_MAX_MODEL_LEN"] = str(cfg.kv_max_model_len)
    env["FASTINFERENCE_LITE_PREFILL_CHUNK"] = str(cfg.prefill_chunk)
    env["FASTINFERENCE_LITE_PREFILL_MICROBATCH"] = str(cfg.prefill_microbatch)
    env["FASTINFERENCE_LITE_PREFILL_RESERVED_TOKENS"] = str(cfg.prefill_reserved_tokens)
    env["FASTINFERENCE_LITE_PREFILL_RESERVE_BACKLOG"] = str(cfg.prefill_reserve_backlog)
    env["FASTINFERENCE_LITE_PREFILL_CATCHUP_RATIO"] = str(cfg.prefill_catchup_ratio)
    env["FASTINFERENCE_LITE_DECODE_PRIORITY"] = str(cfg.decode_priority)

    cmd = [
        sys.executable,
        "tests/e2e_full_benchmark.py",
        "--models",
        model_key,
        "--json-out",
        str(out_json),
    ]
    t0 = time.perf_counter()
    if dry_run:
        stdout_log.write_text("DRY_RUN\n" + " ".join(cmd), encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        rc = 0
    else:
        rc = _run_cmd(
            cmd=cmd,
            env=env,
            cwd=root,
            stdout_path=stdout_log,
            stderr_path=stderr_log,
        )
    elapsed_s = time.perf_counter() - t0

    row: dict[str, Any] = {
        "name": cfg.name,
        "config": asdict(cfg),
        "ok": rc == 0 and out_json.exists(),
        "returncode": rc,
        "elapsed_s": elapsed_s,
        "summary_path": str(out_json),
        "stdout_path": str(stdout_log),
        "stderr_path": str(stderr_log),
    }

    if row["ok"]:
        try:
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            s = (payload.get("summary") or {}).get(model_key) or {}
            row["skipped"] = bool(s.get("skipped", 0.0) == 1.0)
            row["timed_out"] = bool(s.get("timed_out", 0.0) == 1.0)
            row["aggregate_tps"] = _safe_float(s.get("aggregate_tps"))
            row["decode_tps_aggregate"] = _safe_float(s.get("decode_tps_aggregate"))
            row["decode_tps_p50"] = _safe_float(s.get("decode_tps_p50"))
            row["ttft_p50_ms"] = _safe_float(s.get("ttft_p50_ms"))
            row["ttft_p95_ms"] = _safe_float(s.get("ttft_p95_ms"))
            row["e2e_p50_ms"] = _safe_float(s.get("e2e_p50_ms"))
            row["e2e_p95_ms"] = _safe_float(s.get("e2e_p95_ms"))
        except Exception as exc:
            row["ok"] = False
            row["parse_error"] = f"{type(exc).__name__}: {exc}"
    return row


def _select_top(rows: list[dict[str, Any]], topn: int) -> list[dict[str, Any]]:
    valid = [r for r in rows if r.get("status") == "pass"]

    def _effective_tps(r: dict[str, Any]) -> float:
        d = _safe_float(r.get("decode_tps_aggregate"), -1.0)
        if d == d and d > 0:
            return d
        return _safe_float(r.get("aggregate_tps"), -1.0)

    valid.sort(
        key=lambda r: (
            _effective_tps(r),
            -_safe_float(r.get("ttft_p50_ms"), 1e18),
        ),
        reverse=True,
    )
    return valid[: max(0, topn)]


def _run_correctness_for_best(root: Path, best: dict[str, Any], skip_a_tier: bool) -> dict[str, Any]:
    cfg = best["config"]
    env = os.environ.copy()
    env["FASTINFERENCE_KV_TYPE"] = str(cfg["kv_type"])
    if cfg.get("kv_max_active_requests") is not None:
        env["FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS"] = str(cfg["kv_max_active_requests"])
    else:
        env.pop("FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS", None)
    if cfg.get("kv_max_model_len") is not None:
        env["FASTINFERENCE_KV_MAX_MODEL_LEN"] = str(cfg["kv_max_model_len"])
    else:
        env.pop("FASTINFERENCE_KV_MAX_MODEL_LEN", None)
    env["FASTINFERENCE_LITE_PREFILL_CHUNK"] = str(cfg["prefill_chunk"])
    env["FASTINFERENCE_LITE_PREFILL_MICROBATCH"] = str(cfg["prefill_microbatch"])
    env["FASTINFERENCE_LITE_PREFILL_RESERVED_TOKENS"] = str(cfg["prefill_reserved_tokens"])
    env["FASTINFERENCE_LITE_PREFILL_RESERVE_BACKLOG"] = str(cfg["prefill_reserve_backlog"])
    env["FASTINFERENCE_LITE_PREFILL_CATCHUP_RATIO"] = str(cfg["prefill_catchup_ratio"])
    env["FASTINFERENCE_LITE_DECODE_PRIORITY"] = str(cfg["decode_priority"])
    env["RUN_GEMMA4_31B"] = "0"
    env["RUN_GEMMA4_26B"] = "1"
    if skip_a_tier:
        env["SKIP_A_TIER"] = "1"

    proc = subprocess.run(
        ["bash", "tests/run_inference_correctness_regression.sh"],
        cwd=str(root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "returncode": int(proc.returncode),
        "ok": int(proc.returncode) == 0,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-80:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-80:]),
    }


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[2]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = (root / args.work_dir / ts).resolve()
    runs_dir = out_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    coarse_cfgs = build_coarse_grid()
    all_rows: list[dict[str, Any]] = []

    print(f"[Day2Grid] output_dir={out_root}")
    print(f"[Day2Grid] phase={args.phase} model_key={args.model_key} runs={len(coarse_cfgs)}")
    for idx, cfg in enumerate(coarse_cfgs, start=1):
        print(f"[Day2Grid][coarse {idx}/{len(coarse_cfgs)}] {cfg.name}")
        row = _run_one(
            root=root,
            run_dir=runs_dir / cfg.name,
            model_key=args.model_key,
            cfg=cfg,
            dry_run=args.dry_run,
        )
        all_rows.append(row)

    baseline_row = all_rows[0] if all_rows else {}
    baseline_ttft = _safe_float(baseline_row.get("ttft_p50_ms"))
    for row in all_rows:
        if args.dry_run:
            row["status"] = "dry_run"
            continue
        ok, reason = _score_row(row, baseline_ttft, args.ttft_degrade_limit)
        row["status"] = reason if not ok else "pass"

    if args.phase == "full":
        seed = _select_top(all_rows, args.top_k_fine_seed)
        fine_cfgs = build_fine_grid([GridConfig(**r["config"]) for r in seed])
        print(f"[Day2Grid] fine_seed={len(seed)} fine_runs={len(fine_cfgs)}")
        for idx, cfg in enumerate(fine_cfgs, start=1):
            print(f"[Day2Grid][fine {idx}/{len(fine_cfgs)}] {cfg.name}")
            row = _run_one(
                root=root,
                run_dir=runs_dir / cfg.name,
                model_key=args.model_key,
                cfg=cfg,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                row["status"] = "dry_run"
            else:
                ok, reason = _score_row(row, baseline_ttft, args.ttft_degrade_limit)
                row["status"] = reason if not ok else "pass"
            all_rows.append(row)

    ranked = _select_top(all_rows, topn=max(1, len(all_rows)))
    best = ranked[0] if ranked else None

    correctness = None
    if best is not None and args.run_correctness_on_best and not args.dry_run:
        print("[Day2Grid] running correctness on best config...")
        correctness = _run_correctness_for_best(
            root=root,
            best=best,
            skip_a_tier=args.correctness_skip_a_tier,
        )

    leaderboard = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_key": args.model_key,
        "phase": args.phase,
        "ttft_degrade_limit": args.ttft_degrade_limit,
        "baseline": baseline_row,
        "rows": all_rows,
        "ranked": ranked,
        "best": best,
        "correctness_on_best": correctness,
    }
    (out_root / "leaderboard.json").write_text(
        json.dumps(leaderboard, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    if best is not None:
        (out_root / "best_config.json").write_text(
            json.dumps(best, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        print(
            "[Day2Grid][BEST]",
            f"name={best['name']}",
            f"decode_tps_aggregate={best.get('decode_tps_aggregate')}",
            f"ttft_p50_ms={best.get('ttft_p50_ms')}",
        )
    else:
        if args.dry_run:
            print("[Day2Grid] dry-run completed.")
        else:
            print("[Day2Grid][WARN] no passing config found.")
    print(f"[Day2Grid] artifacts={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
