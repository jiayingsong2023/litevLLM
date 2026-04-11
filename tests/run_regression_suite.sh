#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fast default regression: pytest unit tests + structural smoke (no full-model loads).
# GPU-heavy / disk-heavy checks: see tests/README.md and docs/INFERENCE_ACCURACY.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"
uv run pytest \
  tests/test_quality_bar_spotcheck_heuristics.py \
  tests/test_logits_dump_stats.py \
  tests/lite_smoke_test.py \
  tests/test_gemma4_strict_audit_smoke.py \
  tests/test_run_inference_correctness_regression.py \
  -v --tb=short "$@"
