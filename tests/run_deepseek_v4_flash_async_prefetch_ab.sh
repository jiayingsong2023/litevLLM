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
uv run --no-sync python - <<'PY'
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
    counters = data.get("profile", {}).get("counters", {})
    agg = data.get("profile", {}).get("aggregate_by_name", {})
    def agg_ms(name):
        return agg.get(name, {}).get("total_ms", 0.0)
    def agg_cnt(name):
        return agg.get(name, {}).get("count", 0)
    sched = counters.get("deepseek_async_prefetch_scheduled_layers", 0)
    opp = counters.get("deepseek_async_prefetch_opportunities", 0)
    print(f"{variant}: decode_tps_steady_state={metrics.get('decode_tps_steady_state', 0):.3f} "
          f"decode_tps_agg={metrics.get('decode_tps_agg', 0):.3f} "
          f"layer_moe_ms={summary.get('phase_totals_ms', {}).get('layer_moe', 0):.3f} "
          f"async_scheduled={sched}/{opp} "
          f"read_clone_ms={agg_ms('raw_payload_read_clone'):.1f}({agg_cnt('raw_payload_read_clone')}) "
          f"h2d_ms={agg_ms('h2d_copy_enqueue'):.1f}({agg_cnt('h2d_copy_enqueue')}) "
          f"cache_insert_ms={agg_ms('cache_insert'):.1f}({agg_cnt('cache_insert')})")
PY
