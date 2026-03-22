#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Tier-B greedy spot-check + optional A-tier (see docs/INFERENCE_ACCURACY.md §5).
#
# Usage (from repo root):
#   MODEL=models/Qwen3.5-9B-FP16 bash scripts/run_inference_quality_suite.sh
#   RUN_A_TIER=1 MODEL=... bash scripts/run_inference_quality_suite.sh
#
# Optional second argument: quant mode (default: none)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"
export FASTINFERENCE_KV_FP8="${FASTINFERENCE_KV_FP8:-0}"

MODEL="${MODEL:-models/Qwen3.5-9B-FP16}"
QUANT="${1:-none}"

echo "=== Inference quality suite (B-tier greedy) ==="
echo "  MODEL=$MODEL  QUANT=$QUANT  FASTINFERENCE_KV_FP8=$FASTINFERENCE_KV_FP8"
  echo "  See docs/INFERENCE_ACCURACY.md §5 for per-layer / GGUF commands."
echo ""

if [[ ! -d "$MODEL" ]]; then
  echo "[Warn] Model directory not found: $MODEL — skipping LiteEngine run."
  echo "Set MODEL=... to your checkpoint path."
  exit 0
fi

uv run python scripts/quality_bar_spotcheck.py \
  --model "$MODEL" \
  --quant "$QUANT" \
  --prompt-subset minimal \
  --max-new-tokens 96 \
  --temperature 0

echo ""
echo "=== Done (B-tier). Next: A-tier ==="
echo "  PYTHONPATH=. uv run python tests/verify_semantic_integrity.py \\"
echo "    --model \"$MODEL\" --preset qwen35_9b_fp16 --hf-model \"$MODEL\" --hf-device cuda"

if [[ "${RUN_A_TIER:-0}" == "1" ]]; then
  echo ""
  echo "=== A-tier (RUN_A_TIER=1): verify_semantic_integrity ==="
  uv run python tests/verify_semantic_integrity.py \
    --model "$MODEL" \
    --preset qwen35_9b_fp16 \
    --hf-model "$MODEL" \
    --hf-device cuda
fi
