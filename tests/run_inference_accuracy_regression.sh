#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Inference accuracy regression for: TinyLlama, Qwen3.5-9B AWQ/GGUF, DeepSeek-V2-Lite GGUF.
# Run from repo root. Requires local models/ paths and a working CUDA/ROCm device.
#
# Usage:
#   FASTINFERENCE_KV_FP8=0 bash tests/run_inference_accuracy_regression.sh
#   SKIP_A_TIER=1 bash tests/run_inference_accuracy_regression.sh   # B-tier only (faster)
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"
export FASTINFERENCE_KV_FP8="${FASTINFERENCE_KV_FP8:-0}"

SPOTCHECK=(uv run python tests/tools/quality_bar_spotcheck.py
  --prompt-subset minimal --max-new-tokens 96 --temperature 0 --chat-template auto --frugal)

echo "=== Tier-B (quality_bar_spotcheck) ==="
echo "[1/4] TinyLlama"
"${SPOTCHECK[@]}" --model models/TinyLlama-1.1B-Chat-v1.0 --quant none

echo "[2/4] Qwen3.5-9B AWQ"
"${SPOTCHECK[@]}" --model models/Qwen3.5-9B-AWQ --quant awq

echo "[3/4] Qwen3.5-9B GGUF"
"${SPOTCHECK[@]}" --model models/Qwen3.5-9B-GGUF --quant gguf

echo "[4/4] DeepSeek-V2-Lite GGUF"
"${SPOTCHECK[@]}" --model models/DeepSeek-V2-Lite-GGUF --quant gguf

if [[ "${SKIP_A_TIER:-0}" == "1" ]]; then
  echo "SKIP_A_TIER=1 — done after Tier-B."
  exit 0
fi

echo ""
echo "=== Tier-A (verify_semantic_integrity, prefill-only) ==="
echo "[A1] TinyLlama — Lite vs HF same tree"
uv run python tests/verify_semantic_integrity.py \
  --model models/TinyLlama-1.1B-Chat-v1.0 \
  --preset tinyllama \
  --hf-same-as-lite \
  --hf-device cuda \
  --prefill-only \
  --apply-chat-template off

echo "[A2] Qwen3.5-9B AWQ vs FP16 HF (HF may load on CPU when paths differ)"
uv run python tests/verify_semantic_integrity.py \
  --model models/Qwen3.5-9B-AWQ \
  --preset qwen35_9b_awq \
  --hf-model models/Qwen3.5-9B-FP16 \
  --prefill-only \
  --apply-chat-template off

echo "[A3] Qwen3.5-9B GGUF vs FP16 HF"
uv run python tests/verify_semantic_integrity.py \
  --model models/Qwen3.5-9B-GGUF \
  --preset qwen35_9b_gguf \
  --hf-model models/Qwen3.5-9B-FP16 \
  --prefill-only \
  --apply-chat-template off

# DeepSeek A-tier (two checks):
#  - A4a: same safetensors — strict bf16 Lite vs HF (implementation parity with transformers).
#  - A4b: GGUF vs Chat bf16 — Q4 quantization drift; use --regression-gate deepseek-gguf (cosine / top-k / argmax composite).
#  Tune: FASTINFERENCE_DEEPSEEK_GGUF_REGRESSION_MIN_COS / _MIN_TOPK (see compare_hf_lite_deepseek_logits.py).
echo "[A4a] DeepSeek-V2-Lite-Chat safetensors — strict parity (CosSim>=0.998 + argmax)"
uv run python tests/tools/compare_hf_lite_deepseek_logits.py \
  --lite-model models/DeepSeek-V2-Lite-Chat \
  --hf-model models/DeepSeek-V2-Lite-Chat \
  --chat-template auto \
  --prompt "The capital of France is" \
  --greedy-steps 0 \
  --regression-gate safetensors

echo "[A4b] DeepSeek-V2-Lite-GGUF vs Chat HF — GGUF regression gate (not same as A4a)"
uv run python tests/tools/compare_hf_lite_deepseek_logits.py \
  --lite-model models/DeepSeek-V2-Lite-GGUF \
  --hf-model models/DeepSeek-V2-Lite-Chat \
  --chat-template auto \
  --prompt "The capital of France is" \
  --greedy-steps 0 \
  --regression-gate deepseek-gguf

echo ""
echo "=== All accuracy regression steps completed OK ==="
