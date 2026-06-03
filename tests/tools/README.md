# `tests/tools/` Regression Helpers

This directory contains only tools used by the maintained regression entrypoints
or their warn-only diagnostics. One-off benchmark, profile, and historical
alignment scripts should not live here.

Run tools from the repository root with `uv run python`.

## Maintained Tools

| Tool | Used by | Purpose |
|------|---------|---------|
| `quality_bar_spotcheck.py` | `tests/run_inference_correctness_regression.sh` | Tier-B text quality smoke for supported local models. |
| `gemma4_single_prompt_smoke.py` | `tests/run_inference_correctness_regression.sh` | Gemma4 A-lite fixed-prompt generation check. |
| `gemma4_prefill_strict_audit.py` | `tests/run_inference_correctness_regression.sh`, warn-only tests | Gemma4 prefill-only strict audit with sequential HF reference. |
| `gemma4_layer_drift_diagnostic.py` | warn-only Gemma4 diagnostic tests | Optional long-decode drift diagnostics. |
| `_gemma4_diag_utils.py` | warn-only Gemma4 diagnostic tests | Shared diagnostic helpers. |
| `perf_grid_search.py` | `tests/test_perf_grid_search.py` | Small wrapper around `tests/e2e_full_benchmark.py` for guarded perf-grid behavior. |

Fixtures under `tests/tools/fixtures/` are used by the correctness regression
and Gemma4 diagnostic tests.

## Main Entrypoints

```bash
bash tests/run_regression_suite.sh
bash tests/run_inference_correctness_regression.sh
uv run python tests/e2e_full_benchmark.py
```

Do not add new scripts here unless they are referenced by one of the maintained
entrypoints or by a focused pytest test.
