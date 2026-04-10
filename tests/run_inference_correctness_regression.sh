#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Inference correctness regression for:
#   - TinyLlama-1.1B-Chat-v1.0
#   - Qwen3.5-9B-AWQ
#   - Gemma4-31B-it-AWQ-4bit (optional, text-only path)
#
# Run from repo root. Requires local models/ paths and a working CUDA/ROCm device.
#
# Policy:
#   - models <= 14B: A-strict + B
#   - models > 14B:  A-lite + B
#
# Usage:
#   FASTINFERENCE_KV_FP8=0 bash tests/run_inference_correctness_regression.sh  # force bf16/fp16 KV (more VRAM)
#   SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh   # B-tier only (faster)
#   FASTINFERENCE_AWQ_POLICY_MATRIX=throughput bash tests/run_inference_correctness_regression.sh
#     # AWQ matrix presets: safe | balanced | throughput | strict
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"
# KV defaults are now handled per-model inside this script.

MODEL_TINYLLAMA="${MODEL_TINYLLAMA:-models/TinyLlama-1.1B-Chat-v1.0}"
MODEL_QWEN35_9B_AWQ="${MODEL_QWEN35_9B_AWQ:-models/Qwen3.5-9B-AWQ}"
MODEL_GEMMA4_31B_Q4="${MODEL_GEMMA4_31B_Q4:-}"
GEMMA4_PROMPTS_FILE="${GEMMA4_PROMPTS_FILE:-tests/tools/fixtures/gemma4_correctness_prompts_default.json}"

HF_QWEN35_9B_FP16="${HF_QWEN35_9B_FP16:-models/Qwen3.5-9B-FP16}"
HF_GEMMA4_31B="${HF_GEMMA4_31B:-}"
RUN_PERF_DIAG="${RUN_PERF_DIAG:-0}"
RUN_AWQ_FUSED_AB="${RUN_AWQ_FUSED_AB:-0}"
RUN_GEMMA4_31B="${RUN_GEMMA4_31B:-1}"
RUN_GEMMA4_A_TIER="${RUN_GEMMA4_A_TIER:-0}"
RUN_GEMMA4_A_LITE="${RUN_GEMMA4_A_LITE:-1}"

require_model_dir() {
  local model_dir="$1"
  local label="$2"
  if [[ ! -d "$model_dir" ]]; then
    echo "[ERROR] Missing model directory for ${label}: ${model_dir}"
    echo "        You can override the path with an env var."
    exit 1
  fi
}

is_hf_repo_id() {
  local model_ref="$1"
  # Treat existing local paths as local even when relative paths contain "/".
  if [[ -e "$model_ref" ]]; then
    return 1
  fi
  # Minimal HF repo id heuristic: owner/repo with no extra path segments.
  [[ "$model_ref" == */* && "$model_ref" != /* && "$model_ref" != */*/* ]]
}

warn_if_repo_id_proxy_is_unsupported() {
  local model_ref="$1"
  if ! is_hf_repo_id "$model_ref"; then
    return 0
  fi
  local proxy="${ALL_PROXY:-${all_proxy:-}}"
  if [[ "$proxy" == socks://* ]]; then
    echo "[Warn] MODEL_GEMMA4_31B_Q4 resolves to a Hugging Face repo id: ${model_ref}"
    echo "[Warn] Current ALL_PROXY uses 'socks://' (${proxy}), which httpx/huggingface_hub rejects."
    echo "[Warn] Prefer a local model dir under models/, or switch proxy to 'socks5://'."
  fi
}

require_model_ref() {
  local model_ref="$1"
  local label="$2"
  if [[ -d "$model_ref" ]]; then
    return 0
  fi
  if is_hf_repo_id "$model_ref"; then
    echo "[Info] Using Hugging Face repo id for ${label}: ${model_ref}"
    return 0
  fi
  echo "[ERROR] Missing local model directory or invalid HF repo id for ${label}: ${model_ref}"
  echo "        Local directory expected, or HF format like owner/repo."
  exit 1
}

resolve_gemma4_model_ref() {
  local candidates=(
    "models/gemma-4-31B-it-AWQ-4bit"
    "models/Gemma-4-31B-Q4"
    "models/Gemma-4-31B-AWQ"
    "models/Gemma-4-31B-AWQ-4bit"
  )
  local candidate
  if [[ -n "${MODEL_GEMMA4_31B_Q4}" ]]; then
    if [[ -d "${MODEL_GEMMA4_31B_Q4}" ]]; then
      printf '%s\n' "${MODEL_GEMMA4_31B_Q4}"
      return 0
    fi
    if is_hf_repo_id "${MODEL_GEMMA4_31B_Q4}"; then
      for candidate in "${candidates[@]}"; do
        if [[ -d "${candidate}" ]]; then
          echo "[Info] Using local Gemma4 model dir instead of repo id override: ${candidate}" >&2
          printf '%s\n' "${candidate}"
          return 0
        fi
      done
    fi
    printf '%s\n' "${MODEL_GEMMA4_31B_Q4}"
    return 0
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  # Last resort for users who explicitly want repo-id loading. Normal correctness
  # runs should stay offline/local whenever a downloaded model folder exists.
  printf '%s\n' "cyankiwi/gemma-4-31B-it-AWQ-4bit"
  return 0
}

SPOTCHECK=(uv run python tests/tools/quality_bar_spotcheck.py
  --prompt-subset minimal --max-new-tokens 96 --temperature 0 --chat-template auto --frugal)
GEMMA4_SPOTCHECK=(uv run python tests/tools/quality_bar_spotcheck.py
  --max-new-tokens 48 --temperature 0 --chat-template auto --frugal)
GEMMA4_A_LITE_SMOKE=(uv run python tests/tools/gemma4_single_prompt_smoke.py
  --max-new-tokens 32
  --temperature 0
  --gpu-memory-utilization 0.90
  --max-model-len 512
  --min-output-chars 8
  --max-num-batched-tokens 1024)

require_model_dir "$MODEL_TINYLLAMA" "TinyLlama"
require_model_dir "$MODEL_QWEN35_9B_AWQ" "Qwen3.5-9B-AWQ"
MODEL_GEMMA4_31B_Q4="$(resolve_gemma4_model_ref)"
warn_if_repo_id_proxy_is_unsupported "$MODEL_GEMMA4_31B_Q4"

echo "=== Tier-B (quality_bar_spotcheck) ==="
echo "[1/2] TinyLlama"
FASTINFERENCE_KV_TYPE=fp8 "${SPOTCHECK[@]}" --model "$MODEL_TINYLLAMA" --quant none

echo "[2/2] Qwen3.5-9B AWQ"
FASTINFERENCE_KV_TYPE=turbo_int4 "${SPOTCHECK[@]}" --model "$MODEL_QWEN35_9B_AWQ" --quant awq

GEMMA4_AVAILABLE=0
if [[ "${RUN_GEMMA4_31B}" == "1" ]]; then
  if [[ -d "$MODEL_GEMMA4_31B_Q4" ]] || is_hf_repo_id "$MODEL_GEMMA4_31B_Q4"; then
    GEMMA4_AVAILABLE=1
    echo "[3/3] Gemma4-31B Q4 (text-only)"
    require_model_ref "$MODEL_GEMMA4_31B_Q4" "Gemma4-31B-Q4"
    GEMMA4_PROMPT_ARGS=(--prompt-subset minimal)
    if [[ -f "$GEMMA4_PROMPTS_FILE" ]]; then
      # Gemma4 uses a model-specific prompt pack to avoid short boundary prompts
      # (e.g. "short_hi") over-blocking the generic correctness suite.
      GEMMA4_PROMPT_ARGS=(--prompts-file "$GEMMA4_PROMPTS_FILE")
    else
      echo "[Warn] GEMMA4_PROMPTS_FILE not found, fallback to built-in minimal set: $GEMMA4_PROMPTS_FILE"
    fi
    FASTINFERENCE_KV_TYPE=turbo_int4 \
    FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS=1 \
    FASTINFERENCE_KV_MAX_MODEL_LEN=512 \
    "${GEMMA4_SPOTCHECK[@]}" --model "$MODEL_GEMMA4_31B_Q4" --quant awq "${GEMMA4_PROMPT_ARGS[@]}"
  else
    echo "[Warn] Gemma4 model dir not found, skipping: $MODEL_GEMMA4_31B_Q4"
  fi
fi

if [[ "${SKIP_A_TIER:-0}" == "1" ]]; then
  echo "SKIP_A_TIER=1 — done after Tier-B."
  exit 0
fi

echo ""
echo "=== Tier-A-strict (<=14B, HF parity) ==="
echo "[A1] TinyLlama — Lite vs HF same tree"
FASTINFERENCE_KV_TYPE=fp8 uv run python tests/verify_semantic_integrity.py \
  --model "$MODEL_TINYLLAMA" \
  --preset tinyllama \
  --hf-same-as-lite \
  --hf-device cuda \
  --prefill-only \
  --apply-chat-template off

echo "[A2] Qwen3.5-9B AWQ vs FP16 HF"
FASTINFERENCE_KV_TYPE=turbo_int4 uv run python tests/verify_semantic_integrity.py \
  --model "$MODEL_QWEN35_9B_AWQ" \
  --preset qwen35_9b_awq \
  --hf-model "$HF_QWEN35_9B_FP16" \
  --prefill-only \
  --apply-chat-template off

if [[ "${GEMMA4_AVAILABLE}" == "1" && "${RUN_GEMMA4_A_TIER}" == "1" ]]; then
  echo "[A3-strict] Gemma4-31B Q4 vs HF (prefill-only, opt-in)"
  GEMMA_HF_ARGS=("--hf-same-as-lite")
  if [[ -n "$HF_GEMMA4_31B" ]]; then
    GEMMA_HF_ARGS=("--hf-model" "$HF_GEMMA4_31B")
  fi
  # Gemma4-31B needs a much smaller KV pool than the generic high-end default.
  FASTINFERENCE_KV_TYPE=turbo_int4 \
  FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS=1 \
  FASTINFERENCE_KV_MAX_MODEL_LEN=512 \
  uv run python tests/verify_semantic_integrity.py \
    --model "$MODEL_GEMMA4_31B_Q4" \
    --preset gemma4_31b_q4 \
    --prefill-only \
    --apply-chat-template off \
    "${GEMMA_HF_ARGS[@]}"
fi

if [[ "${GEMMA4_AVAILABLE}" == "1" && "${RUN_GEMMA4_A_LITE}" == "1" ]]; then
  echo ""
  echo "=== Tier-A-lite (>14B, key-point audit) ==="
  echo "[A3-lite] Gemma4-31B Q4 multi-prompt text-only audit"
  FASTINFERENCE_KV_TYPE=turbo_int4 \
  FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS=1 \
  FASTINFERENCE_KV_MAX_MODEL_LEN=512 \
  "${GEMMA4_A_LITE_SMOKE[@]}" --model "$MODEL_GEMMA4_31B_Q4"
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
echo "=== All requested correctness regression steps completed OK ==="

if [[ "$RUN_PERF_DIAG" == "1" ]]; then
  echo ""
  echo "=== Optional Perf Diagnostics (RUN_PERF_DIAG=1) ==="
  PERF_MODELS="tinyllama,qwen35_9b_awq"
  if [[ "${GEMMA4_AVAILABLE}" == "1" ]]; then
    PERF_MODELS="${PERF_MODELS},gemma4_31b_q4"
  fi
  PERF_JSON="${PERF_JSON:-.tmp_perf_regression_awq_from_accuracy_suite.json}"
  echo "[P1] Running tests/e2e_full_benchmark.py --models ${PERF_MODELS}"
  uv run python tests/e2e_full_benchmark.py \
    --models "${PERF_MODELS}" \
    --json-out "${PERF_JSON}"
  echo "[P1] Perf diagnostics JSON: ${PERF_JSON}"
fi
