#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Inference correctness regression for:
#   - TinyLlama-1.1B-Chat-v1.0
#   - Qwen3.5-9B-AWQ
#   - Gemma4-31B-it-AWQ-4bit
#   - Gemma4-26B-A4B-it-AWQ-4bit
#
# Run from repo root. Requires local models/ paths and a working CUDA/ROCm device.
#
# Policy:
#   - models <= 14B: A-strict + B
#   - models > 14B: B + isolated A-lite by default
#   - exception: Gemma4-31B strict HF parity is disabled; Gemma4-26B keeps a prefill-only strict audit
#
# Usage:
#   FASTINFERENCE_CONFIG=/path/to/config.toml bash tests/run_inference_correctness_regression.sh
#     # config locator is supported by the runtime, but this script installs
#     # per-model temporary configs for reproducible regression behavior.
#   SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh   # B-tier only (faster)
#   FASTINFERENCE_AWQ_POLICY_MATRIX=throughput bash tests/run_inference_correctness_regression.sh
#     # AWQ matrix presets: safe | balanced | throughput | strict
#   RUN_GEMMA4_31B=0 or RUN_GEMMA4_26B=0 can disable one large-model family explicitly.
#   RUN_DEEPSEEK_V4_FLASH_GPU_SMOKE=1 enables the opt-in real GGUF DeepSeek Tier-B quality smoke.
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"
# KV defaults are now handled per-model inside this script.

FI_REGRESSION_CONFIG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fastinference-correctness-config.XXXXXX")"
cleanup_fastinference_regression_config() {
  rm -rf "$FI_REGRESSION_CONFIG_DIR"
}
trap cleanup_fastinference_regression_config EXIT

write_fastinference_config() {
  local path="$1"
  local profile="$2"
  local kv_type="$3"
  local legacy_enabled="$4"
  shift 4
  cat >"$path" <<EOF
profile = "${profile}"
kv_type = "${kv_type}"

[legacy_env]
enabled = ${legacy_enabled}
EOF
  # Append tuning_keyvals if any remain
  if [ $# -gt 0 ]; then
    printf "\n[tuning_keyvals]\n" >>"$path"
    while [ $# -gt 0 ]; do
      local key="$1"
      local value="$2"
      shift 2
      printf '%s = "%s"\n' "$key" "$value" >>"$path"
    done
  fi
}

CONFIG_TINY_FP8="${FI_REGRESSION_CONFIG_DIR}/tiny-fp8.toml"
CONFIG_QWEN_ACCURACY_TURBO="${FI_REGRESSION_CONFIG_DIR}/qwen-accuracy-turbo.toml"
CONFIG_GEMMA_31B="${FI_REGRESSION_CONFIG_DIR}/gemma31b-benchmark-turbo.toml"
CONFIG_GEMMA_26B="${FI_REGRESSION_CONFIG_DIR}/gemma26b-benchmark-turbo.toml"
CONFIG_GEMMA_31B_A_LITE="${FI_REGRESSION_CONFIG_DIR}/gemma31b-a-lite-latency.toml"
CONFIG_GEMMA_26B_A_LITE="${FI_REGRESSION_CONFIG_DIR}/gemma26b-a-lite-latency.toml"
write_fastinference_config "$CONFIG_TINY_FP8" "auto" "fp8" "false"
write_fastinference_config "$CONFIG_QWEN_ACCURACY_TURBO" "accuracy" "turbo_int4" "false"
write_fastinference_config "$CONFIG_GEMMA_31B" "benchmark" "turbo_int4" "false" \
  FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS 1 \
  FASTINFERENCE_KV_MAX_MODEL_LEN 512 \
  FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ 1 \
  FASTINFERENCE_AWQ_DECODE_GEMV 1 \
  FASTINFERENCE_AWQ_GROUP32_GEMV_ALL 1 \
  FASTINFERENCE_AWQ_FUSED_GATE_UP 1 \
  FASTINFERENCE_GPU_GREEDY_SAMPLING 1 \
  FASTINFERENCE_GPU_GREEDY_MAX_TOKENS_ONLY 1 \
  FASTINFERENCE_GPU_GREEDY_BYPASS_CPU_POLICIES 1
write_fastinference_config "$CONFIG_GEMMA_26B" "benchmark" "turbo_int4" "false" \
  FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS 1 \
  FASTINFERENCE_KV_MAX_MODEL_LEN 512 \
  FASTINFERENCE_AWQ_DECODE_GEMV 1 \
  FASTINFERENCE_AWQ_FUSED_GATE_UP 1
write_fastinference_config "$CONFIG_GEMMA_31B_A_LITE" "latency" "fp8" "false"
write_fastinference_config "$CONFIG_GEMMA_26B_A_LITE" "latency" "fp8" "false"

MODEL_TINYLLAMA="${MODEL_TINYLLAMA:-models/TinyLlama-1.1B-Chat-v1.0}"
MODEL_QWEN35_9B_AWQ="${MODEL_QWEN35_9B_AWQ:-models/Qwen3.5-9B-AWQ}"
MODEL_GEMMA4_31B_Q4="${MODEL_GEMMA4_31B_Q4:-models/gemma-4-31B-it-AWQ-4bit}"
MODEL_GEMMA4_26B_A4B="${MODEL_GEMMA4_26B_A4B:-models/gemma-4-26B-A4B-it-AWQ-4bit}"
MODEL_DEEPSEEK_V4_FLASH_GGUF="${MODEL_DEEPSEEK_V4_FLASH_GGUF:-models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}"
TINYLLAMA_PROMPTS_FILE="${TINYLLAMA_PROMPTS_FILE:-tests/tools/fixtures/tinyllama_correctness_prompts_default.json}"
GEMMA4_PROMPTS_FILE="${GEMMA4_PROMPTS_FILE:-tests/tools/fixtures/gemma4_correctness_prompts_default.json}"

HF_QWEN35_9B_FP16="${HF_QWEN35_9B_FP16:-models/Qwen3.5-9B-FP16}"
HF_GEMMA4_31B="${HF_GEMMA4_31B:-}"
HF_GEMMA4_26B="${HF_GEMMA4_26B:-}"
RUN_PERF_DIAG="${RUN_PERF_DIAG:-0}"
RUN_AWQ_FUSED_AB="${RUN_AWQ_FUSED_AB:-0}"
RUN_GEMMA4_31B="${RUN_GEMMA4_31B:-1}"
RUN_GEMMA4_26B="${RUN_GEMMA4_26B:-1}"
RUN_GEMMA4_A_TIER="${RUN_GEMMA4_A_TIER:-0}"  # compatibility no-op; Gemma4-31B uses A-lite only.
RUN_GEMMA4_A_STRICT="${RUN_GEMMA4_A_STRICT:-0}"  # compatibility no-op; Gemma4-31B strict audit is disabled.
RUN_GEMMA4_A_LITE="${RUN_GEMMA4_A_LITE:-1}"
RUN_GEMMA4_26B_A_STRICT="${RUN_GEMMA4_26B_A_STRICT:-1}"
RUN_GEMMA4_26B_A_LITE="${RUN_GEMMA4_26B_A_LITE:-1}"
RUN_DEEPSEEK_V4_FLASH_GPU_SMOKE="${RUN_DEEPSEEK_V4_FLASH_GPU_SMOKE:-0}"
FI_CORRECTNESS_STAGE_TIMEOUT="${FI_CORRECTNESS_STAGE_TIMEOUT:-45m}"
FI_CORRECTNESS_GEMMA_STAGE_TIMEOUT="${FI_CORRECTNESS_GEMMA_STAGE_TIMEOUT:-75m}"
FI_CORRECTNESS_DEEPSEEK_STAGE_TIMEOUT="${FI_CORRECTNESS_DEEPSEEK_STAGE_TIMEOUT:-45m}"
FI_CORRECTNESS_PERF_STAGE_TIMEOUT="${FI_CORRECTNESS_PERF_STAGE_TIMEOUT:-90m}"
FI_CORRECTNESS_STAGE_KILL_AFTER="${FI_CORRECTNESS_STAGE_KILL_AFTER:-60s}"
print_gemma4_profile() {
  local label="$1"
  shift
  echo "  ${label} profile: $*"
}

run_stage() {
  local label="$1"
  local stage_timeout="$2"
  shift 2

  local start_seconds="$SECONDS"
  echo "[Stage] START ${label} timeout=${stage_timeout}"

  local rc=0
  if command -v timeout >/dev/null 2>&1; then
    timeout --foreground --kill-after="$FI_CORRECTNESS_STAGE_KILL_AFTER" "$stage_timeout" "$@" || rc=$?
  else
    echo "[Warn] coreutils 'timeout' not found; running ${label} without wall-clock guard."
    "$@" || rc=$?
  fi

  local elapsed_seconds=$((SECONDS - start_seconds))
  if [[ "$rc" -eq 0 ]]; then
    echo "[Stage] OK ${label} elapsed=${elapsed_seconds}s"
    return 0
  fi
  if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
    echo "[ERROR] Stage timed out: ${label} timeout=${stage_timeout} elapsed=${elapsed_seconds}s rc=${rc}"
    echo "        This usually indicates a stuck GPU/ROCm operation, model load, or kernel launch."
  else
    echo "[ERROR] Stage failed: ${label} elapsed=${elapsed_seconds}s rc=${rc}"
  fi
  return "$rc"
}

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

cleanup_after_model_step() {
  local label="$1"
  echo "[Info] Cleanup after ${label}"
  uv run python -c 'import gc; gc.collect(); import torch;
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    ipc_collect = getattr(torch.cuda, "ipc_collect", None)
    if ipc_collect is not None:
        ipc_collect()' >/dev/null 2>&1 || true
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

resolve_gemma4_26b_model_ref() {
  local candidates=(
    "models/gemma-4-26B-A4B-it-AWQ-4bit"
    "models/Gemma-4-26B-A4B-it-AWQ-4bit"
    "models/gemma-4-26b-a4b-it-awq-4bit"
    "models/Gemma-4-26B-A4B"
  )
  local candidate
  if [[ -n "${MODEL_GEMMA4_26B_A4B}" ]]; then
    if [[ -d "${MODEL_GEMMA4_26B_A4B}" ]]; then
      printf '%s\n' "${MODEL_GEMMA4_26B_A4B}"
      return 0
    fi
    if is_hf_repo_id "${MODEL_GEMMA4_26B_A4B}"; then
      for candidate in "${candidates[@]}"; do
        if [[ -d "${candidate}" ]]; then
          echo "[Info] Using local Gemma4 26B model dir instead of repo id override: ${candidate}" >&2
          printf '%s\n' "${candidate}"
          return 0
        fi
      done
    fi
    printf '%s\n' "${MODEL_GEMMA4_26B_A4B}"
    return 0
  fi
  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' "google/gemma-4-26B-A4B-it"
  return 0
}

SPOTCHECK=(uv run python tests/tools/quality_bar_spotcheck.py
  --prompt-subset minimal --max-new-tokens 96 --temperature 0 --chat-template auto --frugal)
GEMMA4_SPOTCHECK=(uv run python tests/tools/quality_bar_spotcheck.py
  --max-new-tokens 48 --temperature 0 --chat-template auto --frugal
  --max-model-len 512)
GEMMA4_A_LITE_SMOKE=(uv run python tests/tools/gemma4_single_prompt_smoke.py
  --max-new-tokens 32
  --temperature 0
  --gpu-memory-utilization 0.90
  --max-model-len 512
  --min-output-chars 8
  --max-num-batched-tokens 1024)
GEMMA4_A_STRICT_AUDIT=(uv run python tests/tools/gemma4_prefill_strict_audit.py
  --hf-device cuda
  --max-model-len 256
  --gpu-memory-utilization 0.80
  --max-num-batched-tokens 512)
GEMMA4_26B_A_STRICT_AUDIT=(uv run python tests/tools/gemma4_prefill_strict_audit.py
  --preset gemma4_26b_a4b
  --hf-device cuda
  --max-model-len 256
  --gpu-memory-utilization 0.80
  --max-num-batched-tokens 512)

require_model_dir "$MODEL_TINYLLAMA" "TinyLlama"
require_model_dir "$MODEL_QWEN35_9B_AWQ" "Qwen3.5-9B-AWQ"
MODEL_GEMMA4_31B_Q4="$(resolve_gemma4_model_ref)"
MODEL_GEMMA4_26B_A4B="$(resolve_gemma4_26b_model_ref)"
warn_if_repo_id_proxy_is_unsupported "$MODEL_GEMMA4_31B_Q4"

echo "=== Tier-B (quality_bar_spotcheck) ==="
echo "[1/2] TinyLlama"
TINYLLAMA_PROMPT_ARGS=(--prompt-subset minimal)
if [[ -f "$TINYLLAMA_PROMPTS_FILE" ]]; then
  TINYLLAMA_PROMPT_ARGS=(--prompts-file "$TINYLLAMA_PROMPTS_FILE")
else
  echo "[Warn] TINYLLAMA_PROMPTS_FILE not found, fallback to built-in minimal set: $TINYLLAMA_PROMPTS_FILE"
fi
run_stage "Tier-B TinyLlama spotcheck" "$FI_CORRECTNESS_STAGE_TIMEOUT" \
  env FASTINFERENCE_CONFIG="$CONFIG_TINY_FP8" \
  "${SPOTCHECK[@]}" --model "$MODEL_TINYLLAMA" --quant none "${TINYLLAMA_PROMPT_ARGS[@]}"

echo "[2/2] Qwen3.5-9B AWQ"
run_stage "Tier-B Qwen3.5-9B AWQ spotcheck" "$FI_CORRECTNESS_STAGE_TIMEOUT" \
  env FASTINFERENCE_CONFIG="$CONFIG_QWEN_ACCURACY_TURBO" \
  "${SPOTCHECK[@]}" --model "$MODEL_QWEN35_9B_AWQ" --quant awq

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
    print_gemma4_profile "Gemma4-31B" "FASTINFERENCE_CONFIG=${CONFIG_GEMMA_31B}"
    run_stage "Tier-B Gemma4-31B spotcheck" "$FI_CORRECTNESS_GEMMA_STAGE_TIMEOUT" \
      env FASTINFERENCE_CONFIG="${CONFIG_GEMMA_31B}" \
      "${GEMMA4_SPOTCHECK[@]}" --model "$MODEL_GEMMA4_31B_Q4" --quant awq "${GEMMA4_PROMPT_ARGS[@]}"
    cleanup_after_model_step "Gemma4-31B Tier-B"
  else
    echo "[Warn] Gemma4 model dir not found, skipping: $MODEL_GEMMA4_31B_Q4"
  fi
fi

GEMMA4_26B_AVAILABLE=0
if [[ "${RUN_GEMMA4_26B}" == "1" ]]; then
  if [[ -d "$MODEL_GEMMA4_26B_A4B" ]] || is_hf_repo_id "$MODEL_GEMMA4_26B_A4B"; then
    GEMMA4_26B_AVAILABLE=1
    echo "[Gemma4-26B] Q4/A4B (text-only)"
    require_model_ref "$MODEL_GEMMA4_26B_A4B" "Gemma4-26B-A4B"
    GEMMA4_26B_PROMPT_ARGS=(--prompt-subset minimal)
    if [[ -f "$GEMMA4_PROMPTS_FILE" ]]; then
      GEMMA4_26B_PROMPT_ARGS=(--prompts-file "$GEMMA4_PROMPTS_FILE")
    fi
    print_gemma4_profile "Gemma4-26B" "FASTINFERENCE_CONFIG=${CONFIG_GEMMA_26B}"
    run_stage "Tier-B Gemma4-26B spotcheck" "$FI_CORRECTNESS_GEMMA_STAGE_TIMEOUT" \
      env FASTINFERENCE_CONFIG="${CONFIG_GEMMA_26B}" \
      "${GEMMA4_SPOTCHECK[@]}" --model "$MODEL_GEMMA4_26B_A4B" --quant awq "${GEMMA4_26B_PROMPT_ARGS[@]}"
    cleanup_after_model_step "Gemma4-26B Tier-B"
  else
    echo "[Warn] Gemma4-26B model dir not found, skipping: $MODEL_GEMMA4_26B_A4B"
  fi
fi

if [[ "${RUN_DEEPSEEK_V4_FLASH_GPU_SMOKE}" == "1" ]]; then
  echo ""
  echo "=== DeepSeek V4 Flash Tier-B quality smoke (opt-in) ==="
  if [[ ! -f "$MODEL_DEEPSEEK_V4_FLASH_GGUF" ]]; then
    echo "[ERROR] Missing DeepSeek V4 Flash GGUF: ${MODEL_DEEPSEEK_V4_FLASH_GGUF}"
    echo "        Override with MODEL_DEEPSEEK_V4_FLASH_GGUF=/path/to/model.gguf"
    exit 1
  fi
  run_stage "Tier-B DeepSeek V4 Flash quality smoke" "$FI_CORRECTNESS_DEEPSEEK_STAGE_TIMEOUT" \
    uv run python tests/tools/deepseek_v4_flash_quality_smoke.py \
    --model "$MODEL_DEEPSEEK_V4_FLASH_GGUF" \
    --context-length 4096 \
    --max-tokens 8 \
    --min-output-chars 8
  cleanup_after_model_step "DeepSeek V4 Flash quality smoke"
fi

if [[ "$RUN_PERF_DIAG" == "1" ]]; then
  echo ""
  echo "=== Optional Perf Diagnostics (RUN_PERF_DIAG=1) ==="
  PERF_MODELS="tinyllama,qwen35_9b_awq"
  if [[ "${GEMMA4_AVAILABLE}" == "1" ]]; then
    PERF_MODELS="${PERF_MODELS},gemma4_31b_q4"
  fi
  if [[ "${GEMMA4_26B_AVAILABLE}" == "1" ]]; then
    PERF_MODELS="${PERF_MODELS},gemma4_26b_a4b"
  fi
  if [[ "${RUN_DEEPSEEK_V4_FLASH_GPU_SMOKE}" == "1" && -f "$MODEL_DEEPSEEK_V4_FLASH_GGUF" ]]; then
    PERF_MODELS="${PERF_MODELS},deepseek_v4_flash_q2_gguf"
  fi
  PERF_JSON="${PERF_JSON:-.tmp_perf_regression_awq_from_accuracy_suite.json}"
  echo "[P1] Running tests/e2e_full_benchmark.py --models ${PERF_MODELS}"
  run_stage "Optional perf diagnostics" "$FI_CORRECTNESS_PERF_STAGE_TIMEOUT" \
    uv run python tests/e2e_full_benchmark.py \
    --models "${PERF_MODELS}" \
    --json-out "${PERF_JSON}"
  echo "[P1] Perf diagnostics JSON: ${PERF_JSON}"
fi

if [[ "${SKIP_A_TIER:-0}" == "1" ]]; then
  echo "SKIP_A_TIER=1 — done after Tier-B."
  exit 0
fi

echo ""
echo "=== Tier-A-strict (<=14B, HF parity) ==="
echo "[A1] TinyLlama — Lite vs HF same tree"
run_stage "Tier-A TinyLlama HF parity" "$FI_CORRECTNESS_STAGE_TIMEOUT" \
  env FASTINFERENCE_CONFIG="$CONFIG_TINY_FP8" \
  uv run python tests/verify_semantic_integrity.py \
  --model "$MODEL_TINYLLAMA" \
  --preset tinyllama \
  --hf-same-as-lite \
  --hf-device cuda \
  --prefill-only \
  --apply-chat-template off

echo "[A2] Qwen3.5-9B AWQ vs FP16 HF"
require_model_dir "$HF_QWEN35_9B_FP16" "Qwen3.5-9B-FP16"
run_stage "Tier-A Qwen3.5-9B HF parity" "$FI_CORRECTNESS_STAGE_TIMEOUT" \
  env FASTINFERENCE_CONFIG="$CONFIG_QWEN_ACCURACY_TURBO" \
  uv run python tests/verify_semantic_integrity.py \
  --model "$MODEL_QWEN35_9B_AWQ" \
  --preset qwen35_9b_awq \
  --hf-model "$HF_QWEN35_9B_FP16" \
  --prefill-only \
  --apply-chat-template off

if [[ "${GEMMA4_AVAILABLE}" == "1" ]]; then
  echo "[Info] Gemma4-31B A-strict prefill audit is disabled; running Tier-B + A-lite only."
fi

if [[ "${GEMMA4_26B_AVAILABLE}" == "1" && "${RUN_GEMMA4_26B_A_STRICT}" == "1" ]]; then
  echo "[A3-strict-26B] Gemma4-26B A4B prefill-only strict audit (manual)"
  GEMMA26_HF_ARGS=()
  if [[ -n "$HF_GEMMA4_26B" ]]; then
    GEMMA26_HF_ARGS=(--hf-model "$HF_GEMMA4_26B")
  fi
  run_stage "Tier-A Gemma4-26B strict audit" "$FI_CORRECTNESS_GEMMA_STAGE_TIMEOUT" \
    env FASTINFERENCE_CONFIG="${CONFIG_GEMMA_26B}" \
    "${GEMMA4_26B_A_STRICT_AUDIT[@]}" --model "$MODEL_GEMMA4_26B_A4B" "${GEMMA26_HF_ARGS[@]}"
  cleanup_after_model_step "Gemma4-26B A-strict"
fi

if [[ "${GEMMA4_AVAILABLE}" == "1" && "${RUN_GEMMA4_A_LITE}" == "1" ]]; then
  echo ""
  echo "=== Tier-A-lite (>14B, key-point audit) ==="
  echo "[A3-lite] Gemma4-31B Q4 multi-prompt text-only audit"
  print_gemma4_profile "Gemma4-31B" "FASTINFERENCE_CONFIG=${CONFIG_GEMMA_31B_A_LITE}"
  run_stage "Tier-A-lite Gemma4-31B audit" "$FI_CORRECTNESS_GEMMA_STAGE_TIMEOUT" \
    env FASTINFERENCE_CONFIG="${CONFIG_GEMMA_31B_A_LITE}" \
    FASTINFERENCE_KV_MAX_MODEL_LEN=512 \
    FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS=1 \
    FASTINFERENCE_AWQ_DECODE_GEMV=1 \
    FASTINFERENCE_AWQ_FUSED_GATE_UP=1 \
    FASTINFERENCE_AWQ_GROUP32_GEMV_ALL=1 \
    FASTINFERENCE_GEMMA4_DENSE_DOWN_PROJ=1 \
    "${GEMMA4_A_LITE_SMOKE[@]}" --model "$MODEL_GEMMA4_31B_Q4"
  cleanup_after_model_step "Gemma4-31B A-lite"
fi

if [[ "${GEMMA4_26B_AVAILABLE}" == "1" && "${RUN_GEMMA4_26B_A_LITE}" == "1" ]]; then
  echo "[A-lite-26B] Gemma4-26B A4B multi-prompt text-only audit"
  print_gemma4_profile "Gemma4-26B" "FASTINFERENCE_CONFIG=${CONFIG_GEMMA_26B_A_LITE}"
  run_stage "Tier-A-lite Gemma4-26B audit" "$FI_CORRECTNESS_GEMMA_STAGE_TIMEOUT" \
    env FASTINFERENCE_CONFIG="${CONFIG_GEMMA_26B_A_LITE}" \
    FASTINFERENCE_KV_MAX_MODEL_LEN=512 \
    FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS=1 \
    FASTINFERENCE_AWQ_DECODE_GEMV=1 \
    FASTINFERENCE_AWQ_FUSED_GATE_UP=1 \
    "${GEMMA4_A_LITE_SMOKE[@]}" --model "$MODEL_GEMMA4_26B_A4B"
  cleanup_after_model_step "Gemma4-26B A-lite"
fi

if [[ "$RUN_AWQ_FUSED_AB" == "1" ]]; then
  echo ""
  echo "=== AWQ Fused A/B (RUN_AWQ_FUSED_AB=1) ==="
  echo "[AB1] Qwen3.5-9B AWQ baseline (fused disabled)"
  run_stage "AWQ fused A/B Qwen3.5 baseline" "$FI_CORRECTNESS_STAGE_TIMEOUT" \
    uv run python tests/verify_semantic_integrity.py \
    --model "$MODEL_QWEN35_9B_AWQ" \
    --preset qwen35_9b_awq \
    --hf-model "$HF_QWEN35_9B_FP16" \
    --prefill-only \
    --awq-disable-fused \
    --apply-chat-template off
  echo "[AB2] Qwen3.5-9B AWQ fused forced"
  run_stage "AWQ fused A/B Qwen3.5 forced" "$FI_CORRECTNESS_STAGE_TIMEOUT" \
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
