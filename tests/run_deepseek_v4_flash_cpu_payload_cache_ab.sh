#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

MODEL=models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf

# Capacity in bytes. 2 GiB is enough to hold the working set of selected experts
# for a 16-token cold-cache decode pass on this model while staying small on a
# 96 GB UMA machine.
CACHE_BYTES="${FASTINFERENCE_DEEPSEEK_V4_FLASH_CPU_PAYLOAD_CACHE_BYTES:-2147483648}"

# Pin the GPU staging budget to a small value so that the working set does not
# fit in the GPU cache and repeated staging misses occur.  Without this, the
# default UMA budget on a 96 GB machine is large enough to keep every payload
# resident, and the CPU cache gets no hits.
STAGING_BUDGET_GB=1

echo "CPU payload cache capacity: ${CACHE_BYTES} bytes"
echo "GPU staging budget: ${STAGING_BUDGET_GB} GB"

run_variant() {
  local variant="$1"
  local cache_value="$2"
  local out_json="/tmp/ds_cpu_payload_cache_ab_${variant}.json"

  echo "===== DeepSeek V4 Flash CPU payload cache A/B: ${variant} (CPU_CACHE_BYTES=${cache_value}) ====="
  FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH=0 \
  FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB="${STAGING_BUDGET_GB}" \
  FASTINFERENCE_DEEPSEEK_V4_FLASH_CPU_PAYLOAD_CACHE_BYTES="${cache_value}" \
    timeout 900 uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
      --model "$MODEL" \
      --context-length 4096 \
      --max-tokens 16 \
      --warmup-tokens 1 \
      --repeat 3 \
      --min-steady-decode-tps 0.0 \
      --profile-json "$out_json"
}

run_variant "off" "0"
run_variant "on" "$CACHE_BYTES"

echo "===== A/B summary ====="
uv run --no-sync python - <<'PY'
import json, pathlib, statistics

def agg(data, name):
    return data.get("profile", {}).get("aggregate_by_name", {}).get(name, {})

def fmt_ms(name, data):
    a = agg(data, name)
    return f"{a.get('total_ms', 0.0):.1f}({a.get('count', 0)})"

for variant in ("off", "on"):
    path = pathlib.Path(f"/tmp/ds_cpu_payload_cache_ab_{variant}.json")
    if not path.exists():
        print(f"{variant}: missing {path}")
        continue
    data = json.loads(path.read_text())
    runs = data.get("runs", [])
    steady = [r.get("decode_tps_steady_state", 0.0) for r in runs]
    agg_tps = [r.get("decode_tps_agg", 0.0) for r in runs]
    summary = data.get("profile_summary", {})
    counters = data.get("profile", {}).get("counters", {})
    first = steady[0] if steady else 0.0
    steady_after = steady[1:] if len(steady) > 1 else steady
    print(
        f"{variant}: "
        f"first_run(cold)_tps={first:.3f} | "
        f"steady_after_warmup_median={statistics.median(steady_after):.3f} "
        f"[{', '.join(f'{v:.3f}' for v in steady_after)}] | "
        f"agg_tps_median={statistics.median(agg_tps):.3f} | "
        f"layer_moe_ms={summary.get('phase_totals_ms', {}).get('layer_moe', 0):.1f} | "
        f"cpu_cache_hits={counters.get('cpu_payload_cache_hits', 0)} "
        f"misses={counters.get('cpu_payload_cache_misses', 0)} "
        f"evictions={counters.get('cpu_payload_cache_evictions', 0)} | "
        f"read_clone_ms={fmt_ms('raw_payload_read_clone', data)} "
        f"h2d_ms={fmt_ms('h2d_copy_enqueue', data)} "
        f"cache_insert_ms={fmt_ms('cache_insert', data)}"
    )
PY
