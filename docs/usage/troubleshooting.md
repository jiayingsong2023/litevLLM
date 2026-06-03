# Troubleshooting

FastInference issues usually fall into one of four areas: config resolution,
model loading, Triton kernel compilation, or GPU memory pressure.

## Config Not Taking Effect

The public config entrypoint is `FASTINFERENCE_CONFIG`:

```bash
FASTINFERENCE_CONFIG=configs/local.toml \
uv run python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3.5-9B-AWQ
```

Minimal TOML:

```toml
profile = "benchmark"
kv_type = "turbo_int4"
```

If behavior differs from expectation, check `/debug/stats` and profile metadata
first. Avoid adding direct `os.environ` reads to engine or model hot paths.

## First Request Is Slow

Triton kernels JIT-compile on first use. The first request or benchmark warmup
can take longer than steady-state decode. Re-run after warmup before comparing
performance numbers.

## GPU Out Of Memory

Reduce one or more of:

- model size
- `max_model_len`
- `max_num_seqs`
- `max_num_batched_tokens`
- KV cache active request count in the runtime profile/config

For large Gemma4 benchmarks, use the repository benchmark defaults first before
raising concurrency.

## Illegal Memory Access

On ROCm/CUDA, illegal memory accesses are often shape-specific kernel issues or
memory pressure. Capture the model, prompt length, batch size, profile, and
kernel path, then run the smallest reproducer possible. For kernel changes,
add or update a PyTorch reference correctness test.

## Model Loading Errors

For AWQ models, ensure the model directory includes compatible safetensors and
quantization metadata. For tokenizer errors, confirm `tokenizer.json` or the
expected Hugging Face tokenizer files are present in the local model directory.

## Quality Or Garbled Output

Run the quality and semantic checks described in
[INFERENCE_ACCURACY.md](../INFERENCE_ACCURACY.md). Lowering strictness is not a
substitute for fixing unreadable, off-topic, or structurally corrupted output.

## Useful Commands

```bash
bash tests/run_regression_suite.sh
SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh
uv run pytest -q tests/smoke
```
