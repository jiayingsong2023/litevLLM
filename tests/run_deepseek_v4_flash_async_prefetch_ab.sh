#!/usr/bin/env bash
set -euo pipefail

MODEL=models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf

run_variant() {
  local variant="$1"
  local async_value="$2"
  local out_json="/tmp/ds_async_prefetch_ab_${variant}.json"

  echo "===== DeepSeek V4 Flash async prefetch A/B: ${variant} (ASYNC_PREFETCH=${async_value}) ====="
  FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH="${async_value}" \
    timeout 600 uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
      --model "$MODEL" \
      --context-length 4096 \
      --max-tokens 16 \
      --warmup-tokens 1 \
      --min-steady-decode-tps 0.0 \
      --profile-json "$out_json"
}

run_variant "off" "0"
run_variant "on" "1"

echo "===== A/B summary ====="
python3 - <<'PY'
import json, pathlib
for variant in ("off", "on"):
    path = pathlib.Path(f"/tmp/ds_async_prefetch_ab_{variant}.json")
    if not path.exists():
        print(f"{variant}: missing {path}")
        continue
    data = json.loads(path.read_text())
    runs = data.get("runs", [])
    metrics = runs[0] if runs else {}
    summary = data.get("profile_summary", {})
    print(f"{variant}: decode_tps_steady_state={metrics.get('decode_tps_steady_state', 0):.3f} "
          f"decode_tps_agg={metrics.get('decode_tps_agg', 0):.3f} "
          f"layer_moe_ms={summary.get('phase_totals_ms', {}).get('layer_moe', 0):.3f}")
PY
