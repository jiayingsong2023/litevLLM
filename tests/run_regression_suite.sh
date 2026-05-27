#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fast default regression: pytest unit tests + structural smoke (no full-model loads).
# GPU-heavy / disk-heavy checks: see tests/README.md and docs/INFERENCE_ACCURACY.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"
uv run pytest \
  tests/smoke \
  tests/test_project_governance.py \
  tests/test_kv_default_policy.py \
  tests/test_compressed_tensors_high_fidelity_flag.py \
  tests/test_perf_grid_search.py \
  tests/test_e2e_warmup_config.py \
  tests/test_step_scheduler_single_request_fast_path.py \
  tests/test_gemma4_kv_helpers.py \
  tests/test_quality_bar_spotcheck_heuristics.py \
  tests/test_logits_dump_stats.py \
  tests/lite_smoke_test.py \
  tests/test_gemma4_strict_audit_smoke.py \
  tests/test_gemma4_26b_strict_warn_only.py \
  tests/test_run_gemma4_26b_diagnostics_warn_only.py \
  tests/test_model_registry_gemma4.py \
  tests/test_gemma4_reference_loader.py \
  tests/test_gemma4_diagnostics_warn_only.py \
  tests/test_run_inference_correctness_regression.py \
  tests/test_kv_selective_attention.py \
  tests/test_paged_attention_kernel_structure.py \
  -v --tb=short "$@"
