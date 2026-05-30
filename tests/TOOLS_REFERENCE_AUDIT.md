# Tests → Tools Reference Audit

Auto-generated 2026-05-30. Documents every dependency from `tests/test_*.py`
into `tests/tools/` — direct imports, `importlib` dynamic loads, and
`subprocess` invocations.

## Summary

17 of the 22 remaining `tests/tools/` scripts have **at least one** active
consumer in a `tests/test_*.py` file. Only 5 scripts are truly unreferenced.

## Referenced tools (DO NOT move or delete without updating consumers)

### `tests/tools/_gemma4_diag_utils.py` (81 lines)
- **test_gemma4_diagnostics_warn_only.py:15** — direct import `from tests.tools._gemma4_diag_utils import ...`
- **test_gemma4_26b_strict_warn_only.py:14** — direct import

### `tests/tools/gemma4_prefill_strict_audit.py` (174 lines)
- **test_gemma4_strict_audit_smoke.py:17** — `importlib.util.spec_from_file_location("gemma4_prefill_strict_audit", ...)`
- **test_gemma4_diagnostics_warn_only.py:84** — subprocess `python tests/tools/gemma4_prefill_strict_audit.py`
- **test_gemma4_26b_strict_warn_only.py:85** — subprocess
- **test_run_inference_correctness_regression.py:63** — assertion on string `tests/tools/gemma4_prefill_strict_audit.py`
- **regression script** — `GEMMA4_A_STRICT_AUDIT`

### `tests/tools/gemma4_single_prompt_smoke.py` (305 lines)
- **test_gemma4_smoke.py:18** — `importlib.util.spec_from_file_location("gemma4_single_prompt_smoke", ...)`
- **test_gemma4_diagnostics_warn_only.py:36** — `importlib` dynamic load
- **test_run_inference_correctness_regression.py:65** — assertion on string
- **regression script** — `GEMMA4_A_LITE_SMOKE`

### `tests/tools/quality_bar_spotcheck.py` (1382 lines)
- **test_quality_bar_spotcheck_heuristics.py:19** — `importlib` dynamic load
- **test_run_inference_correctness_regression.py:170** — assertion on string containing `tests/tools/quality_bar_spotcheck.py`
- **regression script** — `SPOTCHECK` and `GEMMA4_SPOTCHECK`

### `tests/tools/gemma4_layer_drift_diagnostic.py` (262 lines)
- **test_gemma4_diagnostics_warn_only.py:119** — subprocess `python tests/tools/gemma4_layer_drift_diagnostic.py`

### `tests/tools/qwen35_moe_packed_lite_logits_audit.py` (368 lines)
- **test_logits_dump_stats.py:15** — `importlib.util.spec_from_file_location("qwen35_moe_packed_lite_logits_audit", ...)`

### `tests/tools/perf_grid_search.py` (493 lines)
- **test_perf_grid_search.py:12** — `importlib.util.spec_from_file_location("perf_grid_search", ...)`

### `tests/tools/fixtures/` (JSON data files)
- **test_run_inference_correctness_regression.py:278** — assertion on string `tests/tools/fixtures/tinyllama_correctness_prompts_default.json`
- **regression script** — `TINYLLAMA_PROMPTS_FILE`, `GEMMA4_PROMPTS_FILE`

## Unreferenced tools (candidates for migration or deletion)

| File | Lines | Recommendation |
|------|-------|----------------|
| `tests/tools/bench_awq_fused_gemm_ab.py` | 127 | Move to `benchmarks/` |
| `tests/tools/bench_gemma4_31b_fused_gemm.py` | 260 | Move to `benchmarks/` |
| `tests/tools/build_awq_fused_profile.py` | 99 | Move to `benchmarks/` |
| `tests/tools/profile_kernel_registers.py` | 272 | Move to `benchmarks/` |
| `tests/tools/gemma4_31b_sprint2_matrix.py` | 1097 | Move to `diagnostics/` |
| `tests/tools/gemma4_decode_window_ab_report.py` | 362 | Move to `diagnostics/` |
| `tests/tools/profile_gemma4_layer_breakdown.py` | 584 | Move to `diagnostics/` |
| `tests/tools/profile_qwen35_layer_breakdown.py` | 621 | Move to `diagnostics/` |
| `tests/tools/qwen35_chunk_gated_delta_alignment.py` | 112 | Move to `diagnostics/` |
| `tests/tools/qwen35_gated_delta_conv_alignment.py` | 117 | Move to `diagnostics/` |
| `tests/tools/qwen35_gguf_alignment_audit.py` | 309 | Move to `diagnostics/` |
| `tests/tools/verify_qwen35_final_hidden_alignment.py` | 324 | Move to `diagnostics/` |
| `tests/tools/report_expected_alignment_metrics.py` | 35 | Move to `diagnostics/` |

## Note

Moving any referenced tool requires:
1. Updating all `importlib` paths in consuming tests
2. Updating all `subprocess` command lines in consuming tests
3. Updating string assertions in `test_run_inference_correctness_regression.py`
4. Updating `tests/run_inference_correctness_regression.sh` paths
5. Updating `CLAUDE.md` if the diagnostic-tool-location convention changes
