#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One-command Gemma4 diagnostics recheck (warn-only):
# - enables real strict + drift diagnostic run
# - compares against fixture baselines
# - reports drift as warnings (non-gating)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"
export RUN_GEMMA4_DIAGNOSTIC=1

uv run pytest tests/test_gemma4_diagnostics_warn_only.py -q "$@"
