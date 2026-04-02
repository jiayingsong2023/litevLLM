#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Inference accuracy regression for:
#   - TinyLlama-1.1B-Chat-v1.0
#   - Qwen3.5-9B-AWQ
#   - Qwen3.5-35B-AWQ
#
# Run from repo root. Requires local models/ paths and a working CUDA/ROCm device.
#
# Usage:
#   FASTINFERENCE_KV_FP8=0 bash tests/run_inference_accuracy_regression.sh  # force bf16/fp16 KV (more VRAM)
#   SKIP_A_TIER=1 bash tests/run_inference_accuracy_regression.sh   # B-tier only (faster)
#   SKIP_35B=1 bash tests/run_inference_accuracy_regression.sh       # skip 35B checks
#   FASTINFERENCE_AWQ_POLICY_MATRIX=throughput bash tests/run_inference_accuracy_regression.sh
#     # AWQ matrix presets: safe | balanced | throughput | strict
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"
# KV defaults are now handled per-model inside this script.

MODEL_TINYLLAMA="${MODEL_TINYLLAMA:-models/TinyLlama-1.1B-Chat-v1.0}"
MODEL_QWEN35_9B_AWQ="${MODEL_QWEN35_9B_AWQ:-models/Qwen3.5-9B-AWQ}"
MODEL_QWEN35_35B_AWQ="${MODEL_QWEN35_35B_AWQ:-models/Qwen3.5-35B-AWQ}"

HF_QWEN35_9B_FP16="${HF_QWEN35_9B_FP16:-models/Qwen3.5-9B-FP16}"
HF_QWEN35_35B_FP16="${HF_QWEN35_35B_FP16:-models/Qwen3.5-35B-FP16}"

SKIP_35B="${SKIP_35B:-0}"
RUN_PERF_DIAG="${RUN_PERF_DIAG:-0}"
RUN_AWQ_FUSED_AB="${RUN_AWQ_FUSED_AB:-0}"

require_model_dir() {
  local model_dir="$1"
  local label="$2"
  if [[ ! -d "$model_dir" ]]; then
    echo "[ERROR] Missing model directory for ${label}: ${model_dir}"
    echo "        You can override with env var or skip 35B via SKIP_35B=1."
    exit 1
  fi
}

SPOTCHECK=(uv run python tests/tools/quality_bar_spotcheck.py
  --prompt-subset minimal --max-new-tokens 96 --temperature 0 --chat-template auto --frugal)

require_model_dir "$MODEL_TINYLLAMA" "TinyLlama"
require_model_dir "$MODEL_QWEN35_9B_AWQ" "Qwen3.5-9B-AWQ"
if [[ "$SKIP_35B" != "1" ]]; then
  require_model_dir "$MODEL_QWEN35_35B_AWQ" "Qwen3.5-35B-AWQ"
fi

echo "=== Tier-B (quality_bar_spotcheck) ==="
echo "[1/3] TinyLlama"
FASTINFERENCE_KV_TYPE=fp8 "${SPOTCHECK[@]}" --model "$MODEL_TINYLLAMA" --quant none

echo "[2/3] Qwen3.5-9B AWQ"
FASTINFERENCE_KV_TYPE=turbo_int4 "${SPOTCHECK[@]}" --model "$MODEL_QWEN35_9B_AWQ" --quant awq

if [[ "$SKIP_35B" == "1" ]]; then
  echo "[3/3] Qwen3.5-35B AWQ (skipped by SKIP_35B=1)"
else
  echo "[3/3] Qwen3.5-35B AWQ (FP8-stable profile)"
  FASTINFERENCE_KV_TYPE=fp8 \
  FASTINFERENCE_QWEN35_MOE_FP8=1 \
  FASTINFERENCE_QWEN35_MOE_OFFLOAD=1 \
    uv run python tests/tools/quality_bar_spotcheck.py \
      --model "$MODEL_QWEN35_35B_AWQ" \
      --quant awq \
      --prompt-subset minimal \
      --max-new-tokens 16 \
      --temperature 0 \
      --chat-template auto \
      --frugal
fi

if [[ "${SKIP_A_TIER:-0}" == "1" ]]; then
  echo "SKIP_A_TIER=1 — done after Tier-B."
  exit 0
fi

echo ""
echo "=== Tier-A (verify_semantic_integrity, prefill-only) ==="
echo "[A1] TinyLlama — Lite vs HF same tree"
FASTINFERENCE_KV_TYPE=fp8 uv run python tests/verify_semantic_integrity.py \
  --model "$MODEL_TINYLLAMA" \
  --preset tinyllama \
  --hf-same-as-lite \
  --hf-device cuda \
  --prefill-only \
  --apply-chat-template off

echo "[A2] Qwen3.5-9B AWQ vs FP16 HF (HF may load on CPU when paths differ)"
FASTINFERENCE_KV_TYPE=turbo_int4 uv run python tests/verify_semantic_integrity.py \
  --model "$MODEL_QWEN35_9B_AWQ" \
  --preset qwen35_9b_awq \
  --hf-model "$HF_QWEN35_9B_FP16" \
  --prefill-only \
  --apply-chat-template off

if [[ "$SKIP_35B" == "1" ]]; then
  echo "[A3] Qwen3.5-35B AWQ (skipped by SKIP_35B=1)"
elif [[ ! -d "$HF_QWEN35_35B_FP16" ]]; then
  echo "[A3] Qwen3.5-35B AWQ vs FP16 HF (skipped: missing HF baseline dir '$HF_QWEN35_35B_FP16')"
else
  echo "[A3] Qwen3.5-35B AWQ vs FP16 HF (prefill-only, FP8-stable profile)"
  FASTINFERENCE_KV_TYPE=turbo_int4 \
  FASTINFERENCE_QWEN35_MOE_FP8=1 \
  FASTINFERENCE_QWEN35_MOE_OFFLOAD=1 \
    uv run python tests/verify_semantic_integrity.py \
      --model "$MODEL_QWEN35_35B_AWQ" \
      --preset qwen35_35b_moe_awq \
      --hf-model "$HF_QWEN35_35B_FP16" \
      --prefill-only \
      --apply-chat-template off
fi

if [[ "$RUN_AWQ_FUSED_AB" == "1" ]]; then
  echo ""
  echo "=== AWQ Fused A/B (RUN_AWQ_FUSED_AB=1) ==="
  echo "[AB1] Qwen3.5-9B AWQ baseline (fused disabled)"
  uv run python tests/verify_semantic_integrity.py \
    --model "$MODEL_QWEN35_9B_AWQ" \
    --preset qwen35_9b_awq \
    --hf-model "$HF_QWEN35_9B_FP16" \
    --prefill-only \
    --awq-disable-fused \
    --apply-chat-template off
  echo "[AB2] Qwen3.5-9B AWQ fused forced"
  uv run python tests/verify_semantic_integrity.py \
    --model "$MODEL_QWEN35_9B_AWQ" \
    --preset qwen35_9b_awq \
    --hf-model "$HF_QWEN35_9B_FP16" \
    --prefill-only \
    --awq-force-fused \
    --apply-chat-template off
fi

echo ""
echo "=== All requested accuracy regression steps completed OK ==="

if [[ "$RUN_PERF_DIAG" == "1" ]]; then
  echo ""
  echo "=== Optional Perf Diagnostics (RUN_PERF_DIAG=1) ==="
  PERF_MODELS="tinyllama,qwen35_9b_awq"
  if [[ "$SKIP_35B" != "1" ]]; then
    PERF_MODELS="${PERF_MODELS},qwen35_35b_awq"
  fi
  PERF_JSON="${PERF_JSON:-.tmp_perf_regression_awq_from_accuracy_suite.json}"
  echo "[P1] Running tests/e2e_full_benchmark.py --models ${PERF_MODELS}"
  uv run python tests/e2e_full_benchmark.py \
    --models "${PERF_MODELS}" \
    --json-out "${PERF_JSON}"
  echo "[P1] Perf diagnostics JSON: ${PERF_JSON}"
fi
