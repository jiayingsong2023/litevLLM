# Contributing to FastInference

FastInference is a lite-only, single-GPU, pure Python + Triton fork of vLLM.
Contributions should keep that product boundary intact.

## Development Environment

Use Python 3.12 and `uv` from the repository root:

```bash
uv sync
```

Do not install dependencies with bare `pip` or run project Python entrypoints
with bare `python`. Use `uv run ...` so commands execute in the pinned
environment.

## Main Checks

Fast structural regression:

```bash
bash tests/run_regression_suite.sh
```

Full-model correctness regression:

```bash
bash tests/run_inference_correctness_regression.sh
```

Lint and formatting:

```bash
uv run ruff check .
uv run ruff format .
```

Type checking:

```bash
uv run mypy vllm
```

Pre-commit:

```bash
uv run pre-commit run --all-files
```

## Architecture Rules

- Keep the official runtime path on `LiteEngine` and the decomposed engine
  modules under `vllm/engine/`.
- Add model-specific capability and policy logic under `vllm/adapters/`.
- Do not reintroduce C++/CUDA extension source into the maintained path.
- Import Triton only through `vllm/triton_utils/`.
- In model hot paths, read runtime settings from `attn_metadata["config"]`
  rather than directly from `os.environ`.
- Prefer PyTorch tensor logic over NumPy in runtime code.

## Kernel Work

Every new Triton kernel needs:

- a PyTorch reference correctness test
- edge cases for empty or minimal token shapes where applicable
- ASCII comments documenting memory layout and block/program tiling
- a lower-ILP fallback or pressure note if register pressure is high

For Gemma4-26B MoE work, the default accepted strategy is currently
`FASTINFERENCE_GEMMA4_MOE_INT4_KERNEL_STRATEGY=two_stage`; other
strategies are experimental until they beat this path in both correctness and
end-to-end performance gates. The current grouped decode experiments include
`batched_chunked` and `batched_grouped_streaming`, which keep compact
intermediate chunks instead of the full `[M, top_k, I]` buffer. AWQ grouped
prefill is enabled only when `FASTINFERENCE_GEMMA4_MOE_PREFILL_GROUPED=1`;
the `fused` grouped prefill strategy remains experimental after the 384-token
profile showed a large regression from repeated gate/up work.

## Supported Regression Surface

The current maintained model targets are:

- `TinyLlama-1.1B`
- `Qwen3.5-9B-AWQ`
- `Gemma4-26B-A4B-it-AWQ-4bit`
- `Gemma4-31B-it-AWQ-4bit`

Use [Capability Matrix](../CAPABILITY_MATRIX.md) as the source of truth for
supported, experimental, compatibility, and unsupported areas.
