#!/usr/bin/env python3
"""Run an alternating A/B e2e benchmark from two existing worktrees."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from statistics import median
from typing import Any


def _run(worktree: Path, benchmark_args: list[str], output: Path) -> dict[str, Any]:
    command = [
        "uv",
        "run",
        "python",
        "tests/e2e_full_benchmark.py",
        *benchmark_args,
        "--json-out",
        str(output),
    ]
    subprocess.run(command, cwd=worktree, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def _metric_values(payloads: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for payload in payloads:
        for result in payload.get("summary", {}).values():
            value = result.get(metric)
            if isinstance(value, (int, float)) and value > 0:
                values.append(float(value))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-worktree", type=Path, required=True)
    parser.add_argument("--candidate-worktree", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--metric", default="decode_tps_aggregate")
    parser.add_argument("--json-out", type=Path, required=True)
    args, benchmark_args = parser.parse_known_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3")
    for worktree in (args.baseline_worktree, args.candidate_worktree):
        if not (worktree / "tests/e2e_full_benchmark.py").is_file():
            parser.error(f"not a FastInference worktree: {worktree}")

    results: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    with tempfile.TemporaryDirectory(prefix="fastinference-ab-") as temp_dir:
        root = Path(temp_dir)
        for index in range(args.runs):
            # Alternate first runner to cancel monotonic thermal drift.
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            for name in order:
                worktree = getattr(args, f"{name}_worktree")
                payload = _run(worktree, benchmark_args, root / f"{name}-{index}.json")
                results[name].append(payload)

    fingerprints = {name: [item.get("fingerprint", {}) for item in payloads] for name, payloads in results.items()}
    if any(fingerprint != fingerprints["baseline"][0] for fingerprint in fingerprints["baseline"] + fingerprints["candidate"]):
        raise SystemExit("benchmark fingerprints differ; results are recorded but not comparable")
    baseline = _metric_values(results["baseline"], args.metric)
    candidate = _metric_values(results["candidate"], args.metric)
    if not baseline or not candidate:
        raise SystemExit(f"no positive {args.metric} values")
    payload = {
        "metric": args.metric,
        "fingerprint": fingerprints["baseline"][0],
        "baseline": results["baseline"],
        "candidate": results["candidate"],
        "baseline_median": median(baseline),
        "candidate_median": median(candidate),
        "candidate_ratio": median(candidate) / median(baseline),
    }
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("baseline_median", "candidate_median", "candidate_ratio")}, indent=2))


if __name__ == "__main__":
    main()
