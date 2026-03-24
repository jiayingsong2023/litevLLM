#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fast default regression: pytest unit tests + structural smoke (no full-model loads).
# GPU-heavy / disk-heavy checks: see tests/README.md and docs/INFERENCE_ACCURACY.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"
uv run pytest \
  tests/test_qwen35_chunk_gated_delta_rule.py \
  tests/test_qwen35_paged_prefill_vs_torch_reference.py \
  tests/test_moe_gguf_packed.py \
  tests/test_quality_bar_spotcheck_heuristics.py \
  tests/test_logits_dump_stats.py \
  tests/test_lora_registry_smoke.py \
  tests/test_multimodal_registry_smoke.py \
  tests/lite_smoke_test.py \
  -v --tb=short "$@"
