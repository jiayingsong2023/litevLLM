# Kubernetes

Kubernetes deployment is not part of the current maintained FastInference
support surface. The project is optimized and tested as a lite-only,
single-GPU runtime.

Before building a Kubernetes deployment, validate the target model and runtime
profile on a single host:

```bash
uv sync
bash tests/run_regression_suite.sh
bash tests/run_inference_correctness_regression.sh
```

For a minimal serving process inside your own container image, use the same
entrypoint as local serving:

```bash
uv run python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3.5-9B-AWQ \
  --host 0.0.0.0 \
  --port 8000
```

Do not assume upstream vLLM Helm charts or upstream `vllm/vllm-openai` images
match this repository's trimmed runtime, dependency lockfile, or supported
model surface.
