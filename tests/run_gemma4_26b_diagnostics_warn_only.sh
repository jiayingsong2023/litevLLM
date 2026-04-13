#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One-command Gemma4-26B diagnostics recheck (warn-only):
# - enables real strict diagnostic run
# - compares against fixture baseline
# - reports drift as warnings (non-gating)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"
export RUN_GEMMA4_26B_DIAGNOSTIC=1

uv run pytest tests/test_gemma4_26b_strict_warn_only.py -q "$@"
