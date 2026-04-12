#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"

MODEL="${MODEL_GEMMA4_26B_A4B:-models/gemma-4-26B-A4B-it-AWQ-4bit}"
OUT="${GEMMA4_DECODE_AB_JSON_OUT:-tests/reports/gemma4_decode_window_ab_report.json}"
KV_TYPE="${FASTINFERENCE_KV_TYPE:-fp16}"
KV_MAX_ACTIVE_REQUESTS="${FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS:-1}"
KV_MAX_MODEL_LEN="${FASTINFERENCE_KV_MAX_MODEL_LEN:-256}"
PROMPT="${GEMMA4_DECODE_AB_PROMPT:-Hi,}"
TOKEN_START="${GEMMA4_DECODE_AB_TOKEN_START:-2}"
TOKEN_END="${GEMMA4_DECODE_AB_TOKEN_END:-16}"
COS_THRESHOLD="${GEMMA4_DECODE_AB_COS_THRESHOLD:-0.99}"
MAX_BATCHED_TOKENS="${GEMMA4_DECODE_AB_MAX_NUM_BATCHED_TOKENS:-256}"
GPU_MEM_UTIL="${GEMMA4_DECODE_AB_GPU_MEM_UTIL:-0.55}"
GUARD_SPAN="${GEMMA4_DECODE_AB_GUARD_SPAN:-3}"

FASTINFERENCE_KV_TYPE="$KV_TYPE" \
FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS="$KV_MAX_ACTIVE_REQUESTS" \
FASTINFERENCE_KV_MAX_MODEL_LEN="$KV_MAX_MODEL_LEN" \
uv run python tests/tools/gemma4_decode_window_ab_report.py \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --token-start "$TOKEN_START" \
  --token-end "$TOKEN_END" \
  --cos-threshold "$COS_THRESHOLD" \
  --kv-type "$KV_TYPE" \
  --max-model-len "$KV_MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --guard-span "$GUARD_SPAN" \
  --json-out "$OUT" \
  --pretty
