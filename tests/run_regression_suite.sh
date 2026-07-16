#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fast default regression: discover top-level unit tests plus smoke and LoRA
# contracts. Full-model DeepSeek tests remain in their dedicated entrypoint.
# GPU-heavy / disk-heavy checks: see tests/README.md and docs/INFERENCE_ACCURACY.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"
UV_RUN=(uv run)
if [[ "${FASTINFERENCE_UV_NO_SYNC:-0}" == "1" ]]; then
  UV_RUN=(uv run --no-sync)
fi
env -u FASTINFERENCE_UV_NO_SYNC "${UV_RUN[@]}" pytest \
  tests/smoke tests/test_*.py tests/lora \
  -v --tb=short "$@"
