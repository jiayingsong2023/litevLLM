# Benchmarks

FastInference uses the lite runtime benchmark entrypoints under `tests/` as the
maintained performance gate. The upstream vLLM benchmark CLI is not the support
surface for this repository.

## Main E2E Gate

```bash
uv run python tests/e2e_full_benchmark.py
```

This runs the maintained single-GPU targets:

- Gemma4-26B-A4B AWQ
- Gemma4-31B AWQ
- DeepSeek V4 Flash Q2 GGUF, when the local target model exists

Use `--json-out` to keep a machine-readable summary:

```bash
uv run python tests/e2e_full_benchmark.py \
  --json-out /tmp/fastinference_e2e.json
```

DeepSeek V4 Flash uses an adapter-owned direct GGUF runtime. Its benchmark
reports decode throughput but does not emit standard per-token streaming
observer events, so `stream_visible=0%` is expected for that workload.

## Related Gates

```bash
bash tests/run_regression_suite.sh
SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh
```

Keep new benchmark scripts tied to a maintained regression entrypoint. One-off
experiments belong in `tests/tools/` only when a regression command uses them.
