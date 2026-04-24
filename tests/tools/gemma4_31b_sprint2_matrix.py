#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Sprint 2 throughput matrix for Gemma4-31B:
- sweep prompt/decode workload shapes
- compare scheduler + KV runtime profiles
- emit per-scenario winner and global leaderboard
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
class Scenario:
    name: str
    prompt_tokens: int
    max_new_tokens: int
    max_model_len: int


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    prefill_chunk: int
    prefill_microbatch: int
    prefill_reserved_tokens: int
    prefill_reserve_backlog: int
    prefill_catchup_ratio: float
    decode_priority: int
    kv_type: str = "turbo_int4"
    kv_max_active_requests: int = 1
    kv_max_model_len: int = 512
    fusion_level: int = 2
    awq_policy_matrix: str = "balanced"


@dataclass(frozen=True)
class LocalDecodeGateScenario:
    name: str
    prompt_tokens: int
    max_new_tokens: int
    max_model_len: int


def _parse_int_csv(raw: str, *, minimum: int) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value < minimum:
            raise ValueError(f"value {value} is below minimum {minimum}")
        out.append(value)
    if not out:
        raise ValueError("empty integer list")
    uniq = sorted(set(out))
    return uniq


def _parse_prompt_decode_pairs(raw: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in raw.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise ValueError(
                f"invalid pair '{part}'. expected format like '256x32,384x64'"
            )
        left, right = token.split("x", 1)
        prompt = int(left.strip())
        decode = int(right.strip())
        if prompt < 8:
            raise ValueError(f"prompt token value {prompt} is below minimum 8")
        if decode < 1:
            raise ValueError(f"decode token value {decode} is below minimum 1")
        out.append((prompt, decode))
    if not out:
        raise ValueError("empty prompt/decode pair list")
    return sorted(set(out))


def _default_profiles() -> list[RuntimeProfile]:
    return [
        RuntimeProfile(
            name="baseline",
            prefill_chunk=256,
            prefill_microbatch=2,
            prefill_reserved_tokens=0,
            prefill_reserve_backlog=2,
            prefill_catchup_ratio=0.25,
            decode_priority=1,
        ),
        RuntimeProfile(
            name="decode_bias",
            prefill_chunk=192,
            prefill_microbatch=1,
            prefill_reserved_tokens=0,
            prefill_reserve_backlog=3,
            prefill_catchup_ratio=0.20,
            decode_priority=1,
        ),
        RuntimeProfile(
            name="catchup_prefill",
            prefill_chunk=384,
            prefill_microbatch=2,
            prefill_reserved_tokens=128,
            prefill_reserve_backlog=2,
            prefill_catchup_ratio=0.35,
            decode_priority=1,
        ),
    ]


def _build_scenarios(
    prompt_tokens: list[int],
    max_new_tokens: list[int],
    max_model_len_override: int | None,
) -> list[Scenario]:
    out: list[Scenario] = []
    for p in prompt_tokens:
        for n in max_new_tokens:
            mlen = max_model_len_override
            if mlen is None:
                mlen = max(512, p + n + 64)
            out.append(
                Scenario(
                    name=f"p{p}_d{n}_m{mlen}",
                    prompt_tokens=p,
                    max_new_tokens=n,
                    max_model_len=mlen,
                )
            )
    return out


def _build_local_decode_gate_scenarios(
    *,
    prompt_decode_pairs: list[tuple[int, int]],
    max_model_len_override: int | None,
) -> list[LocalDecodeGateScenario]:
    out: list[LocalDecodeGateScenario] = []
    for prompt_tokens, max_new_tokens in prompt_decode_pairs:
        mlen = max_model_len_override
        if mlen is None:
            mlen = max(512, prompt_tokens + max_new_tokens + 64)
        out.append(
            LocalDecodeGateScenario(
                name=f"p{prompt_tokens}_d{max_new_tokens}_m{mlen}",
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                max_model_len=mlen,
            )
        )
    return out


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _queue_timeout_env(timeout_s: float) -> dict[str, str]:
    if float(timeout_s) <= 0.0:
        return {}
    return {"FASTINFERENCE_LITE_QUEUE_TIMEOUT_SECONDS": str(float(timeout_s))}


def _run_one(
    *,
    root: Path,
    out_dir: Path,
    scenario: Scenario,
    profile: RuntimeProfile,
    concurrent: int,
    warmup_prefill_rounds: int,
    warmup_decode_rounds: int,
    warmup_decode_tokens: int,
    dry_run: bool,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    case_dir = out_dir / scenario.name / profile.name
    case_dir.mkdir(parents=True, exist_ok=True)
    summary_path = case_dir / "e2e_summary.json"
    stdout_path = case_dir / "stdout.log"
    stderr_path = case_dir / "stderr.log"

    env = os.environ.copy()
    env["FASTINFERENCE_KV_TYPE"] = profile.kv_type
    env["FASTINFERENCE_FUSION_LEVEL"] = str(profile.fusion_level)
    env["FASTINFERENCE_AWQ_POLICY_MATRIX"] = str(profile.awq_policy_matrix)
    env["FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS"] = str(profile.kv_max_active_requests)
    env["FASTINFERENCE_KV_MAX_MODEL_LEN"] = str(profile.kv_max_model_len)
    env["FASTINFERENCE_LITE_PREFILL_CHUNK"] = str(profile.prefill_chunk)
    env["FASTINFERENCE_LITE_PREFILL_MICROBATCH"] = str(profile.prefill_microbatch)
    env["FASTINFERENCE_LITE_PREFILL_RESERVED_TOKENS"] = str(profile.prefill_reserved_tokens)
    env["FASTINFERENCE_LITE_PREFILL_RESERVE_BACKLOG"] = str(profile.prefill_reserve_backlog)
    env["FASTINFERENCE_LITE_PREFILL_CATCHUP_RATIO"] = str(profile.prefill_catchup_ratio)
    env["FASTINFERENCE_LITE_DECODE_PRIORITY"] = str(profile.decode_priority)
    for k, v in (extra_env or {}).items():
        env[str(k)] = str(v)

    cmd = [
        sys.executable,
        "tests/e2e_full_benchmark.py",
        "--models",
        "gemma4_31b_q4",
        "--json-out",
        str(summary_path),
        "--gemma31b-concurrent",
        str(concurrent),
        "--gemma31b-prompt-tokens",
        str(scenario.prompt_tokens),
        "--gemma31b-max-new-tokens",
        str(scenario.max_new_tokens),
        "--gemma31b-max-model-len",
        str(scenario.max_model_len),
        "--warmup-prefill-rounds",
        str(max(0, warmup_prefill_rounds)),
        "--warmup-decode-rounds",
        str(max(0, warmup_decode_rounds)),
        "--warmup-decode-tokens",
        str(max(1, warmup_decode_tokens)),
    ]
    t0 = time.perf_counter()
    if dry_run:
        stdout_path.write_text("DRY_RUN\n" + " ".join(cmd), encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        rc = 0
    else:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        rc = int(proc.returncode)
    elapsed_s = time.perf_counter() - t0

    row: dict[str, Any] = {
        "scenario": asdict(scenario),
        "profile": asdict(profile),
        "ok": bool(rc == 0 and (dry_run or summary_path.exists())),
        "returncode": rc,
        "elapsed_s": elapsed_s,
        "summary_path": str(summary_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    if dry_run or not row["ok"]:
        return row
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        s = (payload.get("summary") or {}).get("gemma4_31b_q4") or {}
        row["timed_out"] = bool(_safe_float(s.get("timed_out"), 0.0) == 1.0)
        row["skipped"] = bool(_safe_float(s.get("skipped"), 0.0) == 1.0)
        row["aggregate_tps"] = _safe_float(s.get("aggregate_tps"))
        row["decode_tps_aggregate"] = _safe_float(s.get("decode_tps_aggregate"))
        row["decode_tps_p50"] = _safe_float(s.get("decode_tps_p50"))
        row["ttft_p50_ms"] = _safe_float(s.get("ttft_p50_ms"))
        row["e2e_p50_ms"] = _safe_float(s.get("e2e_p50_ms"))
        row["decode_p50_ms"] = _safe_float(s.get("decode_p50_ms"))
    except Exception as exc:
        row["ok"] = False
        row["parse_error"] = f"{type(exc).__name__}: {exc}"
    return row


def _decode_tps(row: dict[str, Any]) -> float:
    value = _safe_float(row.get("decode_tps_aggregate"), float("nan"))
    return value if value == value else -1.0


def _select_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        r
        for r in rows
        if r.get("ok") is True and not r.get("timed_out") and not r.get("skipped")
    ]
    if not valid:
        return None
    valid.sort(
        key=lambda r: (
            _decode_tps(r),
            _safe_float(r.get("aggregate_tps"), -1.0),
            -_safe_float(r.get("ttft_p50_ms"), 1e18),
        ),
        reverse=True,
    )
    return valid[0]


def _is_row_valid(row: dict[str, Any]) -> bool:
    return bool(
        row.get("ok") is True and not row.get("timed_out") and not row.get("skipped")
    )


def _row_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        scenario_name = str((row.get("scenario") or {}).get("name") or "")
        profile_name = str((row.get("profile") or {}).get("name") or "")
        if not scenario_name or not profile_name:
            continue
        out[(scenario_name, profile_name)] = row
    return out


def _derive_bucket_cutoff(
    *,
    rows: list[dict[str, Any]],
    scenarios: list[Scenario],
    short_profile: str,
    long_profile: str,
) -> dict[str, Any]:
    prompts = sorted({int(s.prompt_tokens) for s in scenarios})
    if not prompts:
        return {"cutoff": 0, "decode_tps_mean": float("nan"), "samples": 0}
    idx = _row_index(rows)
    best: dict[str, Any] | None = None
    for cutoff in prompts:
        tps_values: list[float] = []
        for scenario in scenarios:
            profile_name = short_profile if scenario.prompt_tokens <= cutoff else long_profile
            row = idx.get((scenario.name, profile_name))
            if row is None or not _is_row_valid(row):
                continue
            tps = _decode_tps(row)
            if tps >= 0:
                tps_values.append(tps)
        mean_tps = (sum(tps_values) / len(tps_values)) if tps_values else float("nan")
        score = {
            "cutoff": cutoff,
            "decode_tps_mean": mean_tps,
            "samples": len(tps_values),
        }
        if best is None:
            best = score
            continue
        lhs = _safe_float(score["decode_tps_mean"], -1.0)
        rhs = _safe_float(best["decode_tps_mean"], -1.0)
        if lhs > rhs or (lhs == rhs and int(score["cutoff"]) < int(best["cutoff"])):
            best = score
    return best or {"cutoff": prompts[0], "decode_tps_mean": float("nan"), "samples": 0}


def _profile_for_prompt_bucket(
    *,
    prompt_tokens: int,
    cutoff: int,
    short_profile: RuntimeProfile,
    long_profile: RuntimeProfile,
) -> RuntimeProfile:
    return short_profile if int(prompt_tokens) <= int(cutoff) else long_profile


def _summarize_valid_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if _is_row_valid(r)]
    decode_values = [_decode_tps(r) for r in valid if _decode_tps(r) >= 0]
    agg_values = [
        _safe_float(r.get("aggregate_tps"), float("nan"))
        for r in valid
        if _safe_float(r.get("aggregate_tps"), float("nan"))
        == _safe_float(r.get("aggregate_tps"), float("nan"))
    ]
    ttft_values = [
        _safe_float(r.get("ttft_p50_ms"), float("nan"))
        for r in valid
        if _safe_float(r.get("ttft_p50_ms"), float("nan"))
        == _safe_float(r.get("ttft_p50_ms"), float("nan"))
    ]
    return {
        "valid_samples": len(valid),
        "decode_tps_mean": (
            sum(decode_values) / len(decode_values) if decode_values else float("nan")
        ),
        "aggregate_tps_mean": (
            sum(agg_values) / len(agg_values) if agg_values else float("nan")
        ),
        "ttft_p50_ms_mean": (
            sum(ttft_values) / len(ttft_values) if ttft_values else float("nan")
        ),
    }


def _evaluate_bucket_gate(
    *,
    bucket_summary: dict[str, Any] | None,
    baseline_summary: dict[str, Any] | None,
    min_valid_samples: int,
    min_decode_tps_mean: float,
    min_vs_baseline_gain_pct: float | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if bucket_summary is None:
        checks.append(
            {
                "name": "bucket_summary_present",
                "ok": False,
                "reason": "bucket_summary missing; run --run-bucket-regression first",
            }
        )
        return {"ok": False, "checks": checks}
    valid_samples = int(bucket_summary.get("valid_samples", 0) or 0)
    decode_tps_mean = _safe_float(bucket_summary.get("decode_tps_mean"), float("nan"))
    checks.append(
        {
            "name": "min_valid_samples",
            "ok": valid_samples >= int(min_valid_samples),
            "actual": valid_samples,
            "threshold": int(min_valid_samples),
        }
    )
    if float(min_decode_tps_mean) > 0.0:
        checks.append(
            {
                "name": "min_decode_tps_mean",
                "ok": decode_tps_mean == decode_tps_mean
                and decode_tps_mean >= float(min_decode_tps_mean),
                "actual": decode_tps_mean,
                "threshold": float(min_decode_tps_mean),
            }
        )
    if min_vs_baseline_gain_pct is not None:
        base_tps = _safe_float(
            (baseline_summary or {}).get("decode_tps_mean"),
            float("nan"),
        )
        if not (decode_tps_mean == decode_tps_mean and base_tps == base_tps and base_tps > 0.0):
            checks.append(
                {
                    "name": "min_vs_baseline_gain_pct",
                    "ok": False,
                    "actual": float("nan"),
                    "threshold": float(min_vs_baseline_gain_pct),
                    "reason": "missing finite baseline/bucket decode_tps_mean",
                }
            )
        else:
            gain_pct = ((decode_tps_mean - base_tps) / base_tps) * 100.0
            checks.append(
                {
                    "name": "min_vs_baseline_gain_pct",
                    "ok": gain_pct >= float(min_vs_baseline_gain_pct),
                    "actual": gain_pct,
                    "threshold": float(min_vs_baseline_gain_pct),
                }
            )
    ok = all(bool(item.get("ok")) for item in checks)
    return {"ok": ok, "checks": checks}


def _evaluate_local_decode_gate(
    *,
    on_rows: list[dict[str, Any]],
    off_rows: list[dict[str, Any]],
    min_valid_pairs: int,
    min_decode_gain_pct: float,
    max_ttft_regression_pct: float,
    max_aggregate_tps_regression_pct: float,
) -> dict[str, Any]:
    idx_on = _row_index(on_rows)
    idx_off = _row_index(off_rows)
    scenario_names = sorted(
        {k[0] for k in idx_on.keys()}.intersection({k[0] for k in idx_off.keys()})
    )
    paired: list[dict[str, Any]] = []
    for scenario_name in scenario_names:
        on_row = idx_on.get((scenario_name, "local_decode_triton_on"))
        off_row = idx_off.get((scenario_name, "local_decode_triton_off"))
        if on_row is None or off_row is None:
            continue
        if not _is_row_valid(on_row) or not _is_row_valid(off_row):
            continue
        on_decode = _decode_tps(on_row)
        off_decode = _decode_tps(off_row)
        on_agg = _safe_float(on_row.get("aggregate_tps"), float("nan"))
        off_agg = _safe_float(off_row.get("aggregate_tps"), float("nan"))
        on_ttft = _safe_float(on_row.get("ttft_p50_ms"), float("nan"))
        off_ttft = _safe_float(off_row.get("ttft_p50_ms"), float("nan"))
        decode_gain_pct = (
            ((on_decode - off_decode) / off_decode) * 100.0
            if off_decode > 0.0
            else float("nan")
        )
        aggregate_tps_gain_pct = (
            ((on_agg - off_agg) / off_agg) * 100.0
            if off_agg == off_agg and off_agg > 0.0 and on_agg == on_agg
            else float("nan")
        )
        ttft_regression_pct = (
            ((on_ttft - off_ttft) / off_ttft) * 100.0
            if off_ttft == off_ttft and off_ttft > 0.0 and on_ttft == on_ttft
            else float("nan")
        )
        paired.append(
            {
                "scenario": scenario_name,
                "decode_gain_pct": decode_gain_pct,
                "aggregate_tps_gain_pct": aggregate_tps_gain_pct,
                "ttft_regression_pct": ttft_regression_pct,
                "on_decode_tps": on_decode,
                "off_decode_tps": off_decode,
                "on_aggregate_tps": on_agg,
                "off_aggregate_tps": off_agg,
                "on_ttft_p50_ms": on_ttft,
                "off_ttft_p50_ms": off_ttft,
            }
        )
    decode_gain_values = [
        _safe_float(x.get("decode_gain_pct"), float("nan"))
        for x in paired
        if _safe_float(x.get("decode_gain_pct"), float("nan"))
        == _safe_float(x.get("decode_gain_pct"), float("nan"))
    ]
    agg_gain_values = [
        _safe_float(x.get("aggregate_tps_gain_pct"), float("nan"))
        for x in paired
        if _safe_float(x.get("aggregate_tps_gain_pct"), float("nan"))
        == _safe_float(x.get("aggregate_tps_gain_pct"), float("nan"))
    ]
    ttft_reg_values = [
        _safe_float(x.get("ttft_regression_pct"), float("nan"))
        for x in paired
        if _safe_float(x.get("ttft_regression_pct"), float("nan"))
        == _safe_float(x.get("ttft_regression_pct"), float("nan"))
    ]
    summary = {
        "valid_pairs": len(paired),
        "decode_gain_pct_mean": (
            sum(decode_gain_values) / len(decode_gain_values)
            if decode_gain_values
            else float("nan")
        ),
        "aggregate_tps_gain_pct_mean": (
            sum(agg_gain_values) / len(agg_gain_values) if agg_gain_values else float("nan")
        ),
        "ttft_regression_pct_mean": (
            sum(ttft_reg_values) / len(ttft_reg_values) if ttft_reg_values else float("nan")
        ),
    }
    checks: list[dict[str, Any]] = [
        {
            "name": "min_valid_pairs",
            "ok": int(summary["valid_pairs"]) >= int(min_valid_pairs),
            "actual": int(summary["valid_pairs"]),
            "threshold": int(min_valid_pairs),
        }
    ]
    if float(min_decode_gain_pct) > 0.0:
        decode_gain_mean = _safe_float(summary["decode_gain_pct_mean"], float("nan"))
        checks.append(
            {
                "name": "min_decode_gain_pct_mean",
                "ok": decode_gain_mean == decode_gain_mean
                and decode_gain_mean >= float(min_decode_gain_pct),
                "actual": decode_gain_mean,
                "threshold": float(min_decode_gain_pct),
            }
        )
    if float(max_ttft_regression_pct) >= 0.0:
        ttft_reg_mean = _safe_float(summary["ttft_regression_pct_mean"], float("nan"))
        checks.append(
            {
                "name": "max_ttft_regression_pct_mean",
                "ok": ttft_reg_mean == ttft_reg_mean
                and ttft_reg_mean <= float(max_ttft_regression_pct),
                "actual": ttft_reg_mean,
                "threshold": float(max_ttft_regression_pct),
            }
        )
    if float(max_aggregate_tps_regression_pct) >= 0.0:
        agg_gain_mean = _safe_float(summary["aggregate_tps_gain_pct_mean"], float("nan"))
        checks.append(
            {
                "name": "max_aggregate_tps_regression_pct_mean",
                "ok": agg_gain_mean == agg_gain_mean
                and ((-agg_gain_mean) <= float(max_aggregate_tps_regression_pct)),
                "actual": (-agg_gain_mean if agg_gain_mean == agg_gain_mean else float("nan")),
                "threshold": float(max_aggregate_tps_regression_pct),
            }
        )
    return {
        "ok": all(bool(x.get("ok")) for x in checks),
        "checks": checks,
        "summary": summary,
        "pairs": paired,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gemma4-31B Sprint2 throughput matrix")
    p.add_argument("--work-dir", type=str, default="tests/reports/sprint2_matrix")
    p.add_argument("--prompt-tokens", type=str, default="64,128,256,384")
    p.add_argument("--max-new-tokens", type=str, default="16,32,64")
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Fixed max_model_len for all scenarios. If omitted, auto=max(512,prompt+decode+64).",
    )
    p.add_argument(
        "--profiles",
        type=str,
        default="baseline,decode_bias,catchup_prefill",
        help="Comma-separated profile names from {baseline,decode_bias,catchup_prefill}.",
    )
    p.add_argument("--concurrent", type=int, default=1)
    p.add_argument("--warmup-prefill-rounds", type=int, default=1)
    p.add_argument("--warmup-decode-rounds", type=int, default=1)
    p.add_argument("--warmup-decode-tokens", type=int, default=8)
    p.add_argument(
        "--matrix-queue-timeout-s",
        type=float,
        default=30.0,
        help=(
            "Queue timeout (seconds) applied to main matrix runs via "
            "FASTINFERENCE_LITE_QUEUE_TIMEOUT_SECONDS. <=0 means keep engine default."
        ),
    )
    p.add_argument(
        "--run-local-decode-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run local decode Triton ON/OFF gate matrix by default "
            "(p256/p384 × d32/d64). Use --no-run-local-decode-gate to disable."
        ),
    )
    p.add_argument(
        "--local-decode-gate-profile",
        type=str,
        default="baseline",
        help="Runtime profile name used for local decode gate ON/OFF A/B runs.",
    )
    p.add_argument(
        "--local-decode-gate-queue-timeout-s",
        type=float,
        default=180.0,
        help=(
            "Queue timeout (seconds) applied only to local decode gate ON/OFF runs via "
            "FASTINFERENCE_LITE_QUEUE_TIMEOUT_SECONDS."
        ),
    )
    p.add_argument(
        "--local-decode-gate-shapes",
        type=str,
        default="256x32,256x64,384x32,384x64",
        help="Prompt/decode shape pairs for local decode gate.",
    )
    p.add_argument(
        "--local-decode-gate-min-valid-pairs",
        type=int,
        default=4,
        help="Local decode gate: minimum number of valid ON/OFF pairs.",
    )
    p.add_argument(
        "--local-decode-gate-min-decode-gain-pct",
        type=float,
        default=5.0,
        help="Local decode gate: minimum decode_tps mean gain (%) for ON vs OFF.",
    )
    p.add_argument(
        "--local-decode-gate-max-ttft-regression-pct",
        type=float,
        default=5.0,
        help="Local decode gate: maximum allowed TTFT regression mean (%). Negative disables this check.",
    )
    p.add_argument(
        "--local-decode-gate-max-aggregate-tps-regression-pct",
        type=float,
        default=2.0,
        help=(
            "Local decode gate: maximum allowed aggregate_tps mean regression (%). "
            "Negative disables this check."
        ),
    )
    p.add_argument(
        "--run-bucket-regression",
        action="store_true",
        help="Run an additional E2E regression pass with prompt-bucket profile routing.",
    )
    p.add_argument(
        "--bucket-short-profile",
        type=str,
        default="decode_bias",
        help="Profile name for short prompts (prompt_tokens <= cutoff).",
    )
    p.add_argument(
        "--bucket-long-profile",
        type=str,
        default="baseline",
        help="Profile name for long prompts (prompt_tokens > cutoff).",
    )
    p.add_argument(
        "--bucket-cutoff",
        type=str,
        default="auto",
        help="Prompt token cutoff for bucket policy. Use integer or 'auto'.",
    )
    p.add_argument(
        "--bucket-gate-min-valid-samples",
        type=int,
        default=0,
        help="Bucket regression gate: minimum valid sample count. 0 disables this check.",
    )
    p.add_argument(
        "--bucket-gate-min-decode-tps-mean",
        type=float,
        default=0.0,
        help="Bucket regression gate: minimum decode_tps_mean. 0 disables this check.",
    )
    p.add_argument(
        "--bucket-gate-min-vs-baseline-gain-pct",
        type=float,
        default=None,
        help=(
            "Bucket regression gate: minimum decode_tps_mean gain (%) vs baseline profile "
            "on the same scenario set."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[2]
    run_dir = (root / args.work_dir / time.strftime("%Y%m%d_%H%M%S")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts = _parse_int_csv(args.prompt_tokens, minimum=8)
    decodes = _parse_int_csv(args.max_new_tokens, minimum=1)
    if args.max_model_len is not None and int(args.max_model_len) < 64:
        raise ValueError("--max-model-len must be >= 64")
    scenarios = _build_scenarios(prompts, decodes, args.max_model_len)

    profile_map = {p.name: p for p in _default_profiles()}
    profile_names = [x.strip() for x in args.profiles.split(",") if x.strip()]
    profiles = [profile_map[name] for name in profile_names]
    local_decode_gate_scenarios: list[LocalDecodeGateScenario] = []
    if bool(args.run_local_decode_gate):
        local_pairs = _parse_prompt_decode_pairs(args.local_decode_gate_shapes)
        local_decode_gate_scenarios = _build_local_decode_gate_scenarios(
            prompt_decode_pairs=local_pairs,
            max_model_len_override=args.max_model_len,
        )

    print(
        f"[Sprint2Matrix] scenarios={len(scenarios)} profiles={len(profiles)} "
        f"out={run_dir}"
    )
    rows: list[dict[str, Any]] = []
    total = len(scenarios) * len(profiles)
    seq = 0
    for scenario in scenarios:
        for profile in profiles:
            seq += 1
            print(
                f"[Sprint2Matrix][{seq}/{total}] {scenario.name} :: {profile.name}"
            )
            row = _run_one(
                root=root,
                out_dir=run_dir / "runs",
                scenario=scenario,
                profile=profile,
                concurrent=max(1, int(args.concurrent)),
                warmup_prefill_rounds=max(0, int(args.warmup_prefill_rounds)),
                warmup_decode_rounds=max(0, int(args.warmup_decode_rounds)),
                warmup_decode_tokens=max(1, int(args.warmup_decode_tokens)),
                dry_run=bool(args.dry_run),
                extra_env=_queue_timeout_env(float(args.matrix_queue_timeout_s)),
            )
            rows.append(row)

    best_by_scenario: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        sid = scenario.name
        best = _select_best([r for r in rows if r["scenario"]["name"] == sid])
        if best is not None:
            best_by_scenario[sid] = best

    profile_scores: dict[str, list[float]] = {}
    for row in rows:
        if not row.get("ok") or row.get("timed_out") or row.get("skipped"):
            continue
        profile_name = str((row.get("profile") or {}).get("name"))
        profile_scores.setdefault(profile_name, []).append(_decode_tps(row))
    profile_ranked = sorted(
        (
            {
                "profile": k,
                "decode_tps_mean": (sum(v) / len(v)) if v else float("nan"),
                "samples": len(v),
            }
            for k, v in profile_scores.items()
        ),
        key=lambda x: _safe_float(x["decode_tps_mean"], -1.0),
        reverse=True,
    )

    local_decode_gate: dict[str, Any] | None = None
    local_decode_gate_rows_on: list[dict[str, Any]] = []
    local_decode_gate_rows_off: list[dict[str, Any]] = []
    if bool(args.run_local_decode_gate):
        gate_profile_name = str(args.local_decode_gate_profile).strip()
        if gate_profile_name not in profile_map:
            raise ValueError(
                f"unknown --local-decode-gate-profile: {gate_profile_name}"
            )
        gate_profile = profile_map[gate_profile_name]
        total_local = len(local_decode_gate_scenarios) * 2
        seq_local = 0
        print(
            "[Sprint2Matrix][LocalDecodeGate] "
            f"profile={gate_profile_name} scenarios={len(local_decode_gate_scenarios)}"
        )
        for scenario in local_decode_gate_scenarios:
            seq_local += 1
            print(
                f"[Sprint2Matrix][LocalDecodeGate][{seq_local}/{total_local}] "
                f"{scenario.name} :: OFF"
            )
            off_row = _run_one(
                root=root,
                out_dir=run_dir / "local_decode_gate_runs",
                scenario=Scenario(
                    name=scenario.name,
                    prompt_tokens=scenario.prompt_tokens,
                    max_new_tokens=scenario.max_new_tokens,
                    max_model_len=scenario.max_model_len,
                ),
                profile=RuntimeProfile(
                    **{
                        **asdict(gate_profile),
                        "name": "local_decode_triton_off",
                    }
                ),
                concurrent=max(1, int(args.concurrent)),
                warmup_prefill_rounds=max(0, int(args.warmup_prefill_rounds)),
                warmup_decode_rounds=max(0, int(args.warmup_decode_rounds)),
                warmup_decode_tokens=max(1, int(args.warmup_decode_tokens)),
                dry_run=bool(args.dry_run),
                extra_env={
                    "FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON": "0",
                    "FASTINFERENCE_LITE_QUEUE_TIMEOUT_SECONDS": str(
                        float(args.local_decode_gate_queue_timeout_s)
                    ),
                },
            )
            local_decode_gate_rows_off.append(off_row)
            seq_local += 1
            print(
                f"[Sprint2Matrix][LocalDecodeGate][{seq_local}/{total_local}] "
                f"{scenario.name} :: ON"
            )
            on_row = _run_one(
                root=root,
                out_dir=run_dir / "local_decode_gate_runs",
                scenario=Scenario(
                    name=scenario.name,
                    prompt_tokens=scenario.prompt_tokens,
                    max_new_tokens=scenario.max_new_tokens,
                    max_model_len=scenario.max_model_len,
                ),
                profile=RuntimeProfile(
                    **{
                        **asdict(gate_profile),
                        "name": "local_decode_triton_on",
                    }
                ),
                concurrent=max(1, int(args.concurrent)),
                warmup_prefill_rounds=max(0, int(args.warmup_prefill_rounds)),
                warmup_decode_rounds=max(0, int(args.warmup_decode_rounds)),
                warmup_decode_tokens=max(1, int(args.warmup_decode_tokens)),
                dry_run=bool(args.dry_run),
                extra_env={
                    "FASTINFERENCE_GEMMA4_LOCAL_DECODE_TRITON": "1",
                    "FASTINFERENCE_LITE_QUEUE_TIMEOUT_SECONDS": str(
                        float(args.local_decode_gate_queue_timeout_s)
                    ),
                },
            )
            local_decode_gate_rows_on.append(on_row)
        local_decode_min_decode_gain_pct = (
            0.0
            if bool(args.dry_run)
            else float(args.local_decode_gate_min_decode_gain_pct)
        )
        local_decode_max_ttft_regression_pct = (
            -1.0
            if bool(args.dry_run)
            else float(args.local_decode_gate_max_ttft_regression_pct)
        )
        local_decode_max_aggregate_tps_regression_pct = (
            -1.0
            if bool(args.dry_run)
            else float(args.local_decode_gate_max_aggregate_tps_regression_pct)
        )
        local_decode_gate = _evaluate_local_decode_gate(
            on_rows=local_decode_gate_rows_on,
            off_rows=local_decode_gate_rows_off,
            min_valid_pairs=max(0, int(args.local_decode_gate_min_valid_pairs)),
            min_decode_gain_pct=local_decode_min_decode_gain_pct,
            max_ttft_regression_pct=local_decode_max_ttft_regression_pct,
            max_aggregate_tps_regression_pct=local_decode_max_aggregate_tps_regression_pct,
        )

    bucket_policy: dict[str, Any] | None = None
    bucket_rows: list[dict[str, Any]] = []
    bucket_summary: dict[str, Any] | None = None
    bucket_gate: dict[str, Any] | None = None
    if args.run_bucket_regression:
        short_name = str(args.bucket_short_profile).strip()
        long_name = str(args.bucket_long_profile).strip()
        if short_name not in profile_map:
            raise ValueError(f"unknown --bucket-short-profile: {short_name}")
        if long_name not in profile_map:
            raise ValueError(f"unknown --bucket-long-profile: {long_name}")
        short_profile = profile_map[short_name]
        long_profile = profile_map[long_name]
        if str(args.bucket_cutoff).strip().lower() == "auto":
            cutoff_meta = _derive_bucket_cutoff(
                rows=rows,
                scenarios=scenarios,
                short_profile=short_name,
                long_profile=long_name,
            )
            cutoff = int(cutoff_meta["cutoff"])
            cutoff_source = "auto"
        else:
            cutoff = int(args.bucket_cutoff)
            cutoff_meta = {
                "cutoff": cutoff,
                "decode_tps_mean": float("nan"),
                "samples": 0,
            }
            cutoff_source = "manual"
        bucket_policy = {
            "short_profile": short_name,
            "long_profile": long_name,
            "cutoff_prompt_tokens": cutoff,
            "cutoff_source": cutoff_source,
            "cutoff_eval": cutoff_meta,
        }
        print(
            "[Sprint2Matrix][BucketPolicy] "
            f"short={short_name} long={long_name} cutoff={cutoff} source={cutoff_source}"
        )
        for idx_s, scenario in enumerate(scenarios, start=1):
            chosen = _profile_for_prompt_bucket(
                prompt_tokens=scenario.prompt_tokens,
                cutoff=cutoff,
                short_profile=short_profile,
                long_profile=long_profile,
            )
            print(
                f"[Sprint2Matrix][bucket {idx_s}/{len(scenarios)}] "
                f"{scenario.name} -> {chosen.name}"
            )
            row = _run_one(
                root=root,
                out_dir=run_dir / "bucket_regression_runs",
                scenario=scenario,
                profile=chosen,
                concurrent=max(1, int(args.concurrent)),
                warmup_prefill_rounds=max(0, int(args.warmup_prefill_rounds)),
                warmup_decode_rounds=max(0, int(args.warmup_decode_rounds)),
                warmup_decode_tokens=max(1, int(args.warmup_decode_tokens)),
                dry_run=bool(args.dry_run),
                extra_env=_queue_timeout_env(float(args.matrix_queue_timeout_s)),
            )
            row["bucket_selected_profile"] = chosen.name
            bucket_rows.append(row)
        bucket_summary = _summarize_valid_rows(bucket_rows)
        baseline_rows = [
            row
            for row in rows
            if str((row.get("profile") or {}).get("name")) == str(long_name)
        ]
        baseline_summary = _summarize_valid_rows(baseline_rows)
        bucket_gate = _evaluate_bucket_gate(
            bucket_summary=bucket_summary,
            baseline_summary=baseline_summary,
            min_valid_samples=max(0, int(args.bucket_gate_min_valid_samples)),
            min_decode_tps_mean=max(0.0, float(args.bucket_gate_min_decode_tps_mean)),
            min_vs_baseline_gain_pct=(
                float(args.bucket_gate_min_vs_baseline_gain_pct)
                if args.bucket_gate_min_vs_baseline_gain_pct is not None
                else None
            ),
        )
        bucket_gate["baseline_summary"] = baseline_summary

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "matrix_queue_timeout_s": float(args.matrix_queue_timeout_s),
        "scenarios": [asdict(x) for x in scenarios],
        "profiles": [asdict(x) for x in profiles],
        "rows": rows,
        "best_by_scenario": best_by_scenario,
        "profile_ranked": profile_ranked,
        "local_decode_gate_config": {
            "enabled": bool(args.run_local_decode_gate),
            "profile": str(args.local_decode_gate_profile),
            "queue_timeout_s": float(args.local_decode_gate_queue_timeout_s),
            "shapes": str(args.local_decode_gate_shapes),
            "min_valid_pairs": int(args.local_decode_gate_min_valid_pairs),
            "min_decode_gain_pct": float(args.local_decode_gate_min_decode_gain_pct),
            "max_ttft_regression_pct": float(
                args.local_decode_gate_max_ttft_regression_pct
            ),
            "max_aggregate_tps_regression_pct": float(
                args.local_decode_gate_max_aggregate_tps_regression_pct
            ),
        },
        "local_decode_gate_rows_on": local_decode_gate_rows_on,
        "local_decode_gate_rows_off": local_decode_gate_rows_off,
        "local_decode_gate": local_decode_gate,
        "bucket_policy": bucket_policy,
        "bucket_regression_rows": bucket_rows,
        "bucket_regression_summary": bucket_summary,
        "bucket_gate": bucket_gate,
    }
    leaderboard = run_dir / "leaderboard.json"
    leaderboard.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[Sprint2Matrix] leaderboard={leaderboard}")
    if profile_ranked:
        top = profile_ranked[0]
        print(
            "[Sprint2Matrix][BEST_PROFILE] "
            f"{top['profile']} decode_tps_mean={top['decode_tps_mean']:.3f}"
        )
    if local_decode_gate is not None:
        print(
            "[Sprint2Matrix][LOCAL_DECODE_GATE] "
            f"ok={bool(local_decode_gate.get('ok'))}"
        )
        summary = local_decode_gate.get("summary") or {}
        print(
            "  "
            f"valid_pairs={summary.get('valid_pairs')}, "
            f"decode_gain_pct_mean={_safe_float(summary.get('decode_gain_pct_mean'), float('nan')):.3f}, "
            f"aggregate_tps_gain_pct_mean={_safe_float(summary.get('aggregate_tps_gain_pct_mean'), float('nan')):.3f}, "
            f"ttft_regression_pct_mean={_safe_float(summary.get('ttft_regression_pct_mean'), float('nan')):.3f}"
        )
        for check in local_decode_gate.get("checks", []):
            print(
                "  - "
                f"{check.get('name')}: ok={bool(check.get('ok'))}, "
                f"actual={check.get('actual')}, threshold={check.get('threshold')}"
            )
    if bucket_summary is not None:
        print(
            "[Sprint2Matrix][BUCKET_REGRESSION] "
            f"valid={bucket_summary['valid_samples']} "
            f"decode_tps_mean={bucket_summary['decode_tps_mean']:.3f} "
            f"aggregate_tps_mean={bucket_summary['aggregate_tps_mean']:.3f}"
        )
    if bucket_gate is not None:
        print(
            "[Sprint2Matrix][BUCKET_GATE] "
            f"ok={bool(bucket_gate.get('ok'))}"
        )
        for check in bucket_gate.get("checks", []):
            print(
                "  - "
                f"{check.get('name')}: ok={bool(check.get('ok'))}, "
                f"actual={check.get('actual')}, threshold={check.get('threshold')}"
            )
    if local_decode_gate is not None and not bool(local_decode_gate.get("ok")):
        return 2
    if bucket_gate is not None and not bool(bucket_gate.get("ok")):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
