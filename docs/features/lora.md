# LoRA

LoRA support exists in the lite runtime, but it is currently experimental. Use
[CAPABILITY_MATRIX.md](../CAPABILITY_MATRIX.md) as the source of truth for
support status.

## Runtime Shape

FastInference keeps LoRA integration inside the Python/Triton lite path rather
than depending on upstream C++/CUDA LoRA extension stacks. Adapter-aware
scheduling and observability counters are present in the runtime, and LoRA
stats are included in benchmark/runtime summaries.

## Current Boundary

- The runtime path and tests exist.
- Mixed-adapter scheduling and fairness counters exist.
- Production policy calibration remains workload dependent.
- New LoRA behavior must preserve the lite-only boundary and should be covered
  by smoke, correctness, and performance gates before support is claimed.

## Verification

Run the fast regression suite after LoRA-related changes:

```bash
bash tests/run_regression_suite.sh
```

For serving behavior, also run:

```bash
uv run pytest -q tests/smoke
```
