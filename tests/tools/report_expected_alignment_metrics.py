#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Print expected alignment metrics from automated tests (see docs/INFERENCE_ACCURACY.md §4).

Usage:
  uv run python tests/tools/report_expected_alignment_metrics.py
"""
from __future__ import annotations

import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    os.chdir(_ROOT)
    tests = [
        "tests/test_qwen35_chunk_gated_delta_rule.py",
        "tests/test_qwen35_paged_prefill_vs_torch_reference.py",
    ]
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=no"] + tests
    print("Running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=_ROOT)
    print()
    print("Expected (see docs/INFERENCE_ACCURACY.md §4):")
    print("  • Chunk gated delta (Lite vs FLA naive): max_err 0 when fla installed")
    print("  • Paged prefill vs torch reference (CUDA): allclose within test rtol/atol")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
