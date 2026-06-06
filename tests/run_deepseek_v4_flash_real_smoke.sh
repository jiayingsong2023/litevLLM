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
