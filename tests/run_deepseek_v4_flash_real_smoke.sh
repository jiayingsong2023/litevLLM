#!/usr/bin/env bash
set -euo pipefail

timeout 600 uv run --no-sync pytest \
  tests/deepseek_v4_flash/test_model_forward_real_smoke.py \
  tests/deepseek_v4_flash/test_model_smoke_no_weights.py \
  tests/deepseek_v4_flash/test_model_loader_route.py \
  -q

timeout 600 uv run --no-sync pytest \
  tests/smoke/test_deepseek_v4_flash_http_smoke.py \
  -q

MODEL=models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf

echo "===== DeepSeek V4 Flash cold-cache gate ====="
# Cold-cache uses the direct-runtime defaults.
timeout 600 uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model "$MODEL" \
  --context-length 4096 \
  --max-tokens 16 \
  --warmup-tokens 1 \
  --min-steady-decode-tps 0.4 \
  --profile-json /tmp/ds_gate_cold_run.json

echo "===== DeepSeek V4 Flash warm-cache gate ====="
# Warm-cache uses full resident. Don't pass --profile-json here; profiler sync overhead lowers the
# steady-state decode TPS below the 1.5 gate even when the actual decode path
# is healthy (~1.58 tok/s without profiling).
timeout 600 uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model "$MODEL" \
  --context-length 4096 \
  --prompt-length 32 \
  --max-tokens 16 \
  --warmup-tokens 16 \
  --repeat 3 \
  --full-resident \
  --min-steady-decode-tps 1.5
