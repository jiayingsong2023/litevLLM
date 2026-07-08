# SPDX-License-Identifier: Apache-2.0
"""One-stop profiling harness for Gemma4-26B-A4B on Radeon 8060S."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(
    cmd: list[str],
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> str:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        text=True,
        capture_output=capture,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed with rc={result.returncode}")
    return result.stdout if capture else ""


def run_e2e_baseline(out_dir: Path, max_new_tokens: int = 24) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "baseline.json"
    env = os.environ.copy()
    env.update({
        "FASTINFERENCE_KV_TYPE": "turbo_int4",
        "FASTINFERENCE_GEMMA4_ALLOW_INT4_KV": "0",
    })
    _run([
        sys.executable, "tests/e2e_full_benchmark.py",
        "--models", "gemma4_26b_a4b",
        "--gemma26b-concurrent", "1",
        "--gemma26b-max-new-tokens", str(max_new_tokens),
        "--json-out", str(json_path),
    ], env=env)
    return json_path


def run_rocprof(out_dir: Path, max_new_tokens: int = 8) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rocprof.json"
    env = os.environ.copy()
    env.update({
        "FASTINFERENCE_KV_TYPE": "turbo_int4",
        "FASTINFERENCE_GEMMA4_ALLOW_INT4_KV": "0",
    })
    _run([
        "rocprofv3", "--kernel-trace", "--stats",
        "--output-dir", str(out_dir / "rocprof"),
        "--", sys.executable, "tests/e2e_full_benchmark.py",
        "--models", "gemma4_26b_a4b",
        "--gemma26b-max-new-tokens", str(max_new_tokens),
        "--json-out", str(json_path),
    ], env=env)
    return out_dir / "rocprof" / "kernel_stats.csv"


def run_awq_audit(out_dir: Path, max_new_tokens: int = 24) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "audit.json"
    log_path = out_dir / "awq_audit.log"
    env = os.environ.copy()
    env.update({
        "FASTINFERENCE_KV_TYPE": "turbo_int4",
        "FASTINFERENCE_GEMMA4_ALLOW_INT4_KV": "0",
    })
    stdout = _run([
        sys.executable, "tests/e2e_full_benchmark.py",
        "--models", "gemma4_26b_a4b",
        "--gemma26b-concurrent", "1",
        "--gemma26b-max-new-tokens", str(max_new_tokens),
        "--json-out", str(json_path),
    ], env=env, capture=True)
    log_path.write_text(stdout, encoding="utf-8")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/gemma26b_profile")
    parser.add_argument("--rocprof-max-new-tokens", type=int, default=8)
    parser.add_argument("--baseline-max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    baseline_json = run_e2e_baseline(out_dir, args.baseline_max_new_tokens)
    kernel_stats_csv = run_rocprof(out_dir, args.rocprof_max_new_tokens)
    audit_log = run_awq_audit(out_dir, args.baseline_max_new_tokens)

    summary = {
        "baseline_json": str(baseline_json),
        "kernel_stats_csv": str(kernel_stats_csv),
        "awq_audit_log": str(audit_log),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("Profile complete.")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
