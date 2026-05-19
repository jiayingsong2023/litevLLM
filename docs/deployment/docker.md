# Docker

FastInference does not currently publish a maintained Docker image as part of
the lite-only support surface. The supported local development path is Python
3.12 plus `uv sync` on a Linux host with a single CUDA or ROCm GPU.

Use the native workflow first:

```bash
uv sync
bash tests/run_regression_suite.sh
```

For serving smoke tests:

```bash
uv run python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3.5-9B-AWQ \
  --host 0.0.0.0 \
  --port 8000
```

If a container is required for deployment, build one around the same commands
and dependency lockfile. Do not base operational expectations on upstream
`vllm/vllm-openai` images; those images target upstream vLLM, not this
lite-only FastInference runtime.
