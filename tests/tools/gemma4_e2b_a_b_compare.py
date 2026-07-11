#!/usr/bin/env python3
import json
import sys
from pathlib import Path

BUCKETS = [128, 512, 2048]


def main() -> None:
    results = []
    bit_exact = True
    for ctx in BUCKETS:
        b = json.loads(Path(f"/tmp/e2b_baseline_{ctx}.json").read_text())
        o = json.loads(Path(f"/tmp/e2b_optimized_{ctx}.json").read_text())
        speedup = o["perf_run"]["median_decode_tps"] / b["perf_run"]["median_decode_tps"]
        tokens_match = o["perf_run"]["token_ids"] == b["perf_run"]["token_ids"]
        bit_exact = bit_exact and tokens_match
        results.append({
            "context": ctx,
            "baseline_tps": b["perf_run"]["median_decode_tps"],
            "optimized_tps": o["perf_run"]["median_decode_tps"],
            "speedup": speedup,
            "tokens_match": tokens_match,
        })
        print(
            f"ctx={ctx:4d} baseline={b['perf_run']['median_decode_tps']:.3f} "
            f"optimized={o['perf_run']['median_decode_tps']:.3f} "
            f"speedup={speedup:.3f}x tokens_match={tokens_match}"
        )

    median_speedup = sorted(r["speedup"] for r in results)[len(results) // 2]
    min_speedup = min(r["speedup"] for r in results)
    print(f"median_speedup={median_speedup:.3f}x min_speedup={min_speedup:.3f}x bit_exact={bit_exact}")

    passed = (
        bit_exact
        and median_speedup >= 1.20
        and min_speedup >= 1.00
    )
    print(f"A/B gate PASSED={passed}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
