#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# CI guard: forbid new lite-engine imports from legacy directories.
# Legacy dirs: vllm/worker vllm/core vllm/distributed vllm/third_party
# Lite dirs: vllm/engine vllm/serving vllm/entrypoints vllm/adapters
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LEGACY_DIRS="vllm/worker|vllm/core|vllm/distributed|vllm/third_party"

violations=$(grep -rn "from \(${LEGACY_DIRS//./\\.}\)" \
  $(find vllm/engine vllm/serving vllm/entrypoints vllm/adapters \
    -name "*.py" -not -path "*__pycache__*") 2>/dev/null || true)

if [ -n "$violations" ]; then
  echo "ERROR: Lite engine paths MUST NOT import from legacy directories:"
  echo "$violations"
  echo ""
  echo "Legacy directories: vllm/worker, vllm/core, vllm/distributed, vllm/third_party"
  echo "If this import is intentional, update docs/DEPENDENCY_CLOSURE.md and this script."
  exit 1
fi

echo "PASS: No lite-engine imports from legacy directories."
