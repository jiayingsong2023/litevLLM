# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastInference (vLLM Lite) is a lite-only, single-GPU LLM inference engine. It is a fork of upstream vLLM, heavily trimmed to a **pure Python + Triton** path targeting AMD ROCm 7.2 first (CUDA compatible). The project priority is converging on stable, maintainable Python + Triton — not maintaining compatibility with upstream vLLM's full feature surface.

Supported models: TinyLlama-1.1B (BF16), Qwen3.5-9B (AWQ), Gemma4-31B-it (AWQ-4bit), Gemma4-26B-A4B-it (AWQ-4bit MoE). Weight formats: Safetensors + AWQ with Gemma4 Q4 compressed-tensor paths. The 26B variant uses a mixture-of-experts architecture with 4 active experts per token (A4B), while the 31B uses dense MLP.

## Commands

Always use `uv` — never bare `python`/`pip`. Python 3.12 only.

```bash
uv sync                                              # install deps (ROCm torch/triton from repo.radeon.com)
uv run pytest tests/test_logits_dump_stats.py         # run a single test
bash tests/run_regression_suite.sh                    # fast unit/smoke (~17 tests, no GPU model loads)
bash tests/run_inference_correctness_regression.sh    # full-model correctness (TinyLlama, Qwen3.5-9B, Gemma4-31B)
uv run ruff check . && uv run ruff format .           # lint + format
uv run mypy vllm                                      # type-check
pre-commit run --all-files                            # full CI pre-commit
```

Start the OpenAI-compatible server:
```bash
uv run python -m vllm.entrypoints.openai.api_server --model models/Qwen3.5-9B-AWQ
```

## Architecture

The main execution path (lite engine):

```
LLM / AsyncLLM / OpenAI API Server
  → vllm/serving/config_builder.py       # VllmConfig + RuntimeConfig (TOML驱动)
  → vllm/engine/lite_engine.py           # orchestration
  → vllm/engine/step_scheduler.py        # step-level scheduling
  → vllm/engine/request_scheduler.py     # request/slot lifecycle
  → vllm/engine/prefill_executor.py      # prefill (hardware SDPA)
  → vllm/engine/decode_executor.py       # decode (Triton PagedAttention)
  → vllm/engine/sampling_driver.py       # sampling
  → vllm/engine/output_pipeline.py       # output assembly
```

Key boundaries:
- `vllm/engine/` — control plane (orchestration, scheduling, sampling, output)
- `vllm/kernels/triton/` — hot-path GPU kernels (PagedAttention, fused AWQ GEMM, MoE, RMSNorm, rotary embedding, reshape_and_cache, FP8 GEMM)
- `vllm/model_executor/` — model definitions and layer implementations (`vllm/model_executor/models/` for model-specific code, `vllm/model_executor/layers/` for shared layers like `lite_linear.py`)
- `vllm/adapters/` — model-specific adapter logic (gemma4, llama, qwen3_5), policy keys with TypedDict
- `vllm/config/`, `vllm/serving/` — configuration and serving layer
- `vllm/attention/` — attention backends and ops
- `vllm/entrypoints/` — CLI, OpenAI API server, output processors
- `vllm/utils/` — utility functions (torch_utils, text_utils)

**Deleted (upstream legacy, no lite path dependency):**
- `vllm/worker/` — ~7,184 lines removed
- `vllm/core/` — ~2,871 lines removed
- `vllm/distributed/` — ~46 lines removed
- `vllm/third_party/` — ~5,127 lines removed
- `vllm/model_executor/warmup/` — legacy kernel warmup
- `vllm/model_executor/layers/mamba/` — legacy Mamba layers
- `vllm/model_executor/layers/kda.py` — legacy KDA module

**No C++ anywhere.** All kernels are hand-written Triton. Enforced by pre-commit hooks.

## Key Conventions

- **Config access**: In model layers, read runtime settings via `attn_metadata["config"]` — never use `os.environ` directly.
- **Triton imports**: Pre-commit forbids direct `import triton`. Import through `vllm/triton_utils/` instead.
- **Kernel requirements**: Every Triton kernel must include ASCII comments describing memory layout and thread/block tiling.
- **PyTorch over NumPy** for runtime tensor logic.
- **Strict PEP 484 typing** (mypy with `check_untyped_defs = true`).
- Commits use conventional-style subjects (`feat:`, `fix:`, `perf:`, `test:`). The commit-msg hook auto-appends `Signed-off-by:`.

## Testing

- All tests in `tests/` with `test_*.py` names; one-off diagnostics in `tests/tools/`.
- New Triton kernels require a PyTorch reference correctness test plus edge cases for 0-token and max-token prompts.
- Before PR: run `bash tests/run_regression_suite.sh`. For kernel/KV-cache/numerics changes, also run `bash tests/run_inference_correctness_regression.sh`.
- Correctness regression auto-detects accuracy tier by model size (`<=14B` → strict A-tier, `>14B` → lite A-tier). Skip the heavier A-tier with `SKIP_A_TIER=1`.

## Configuration

**Never use `os.environ` to change program execution paths.** All runtime tuning parameters go through TOML config file via `FASTINFERENCE_CONFIG` environment variable.

Example TOML config (`config.toml`):
```toml
profile = "benchmark"
kv_type = "turbo_int4"

[tuning_keyvals]
FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS = "1"
FASTINFERENCE_KV_MAX_MODEL_LEN = "512"
```

Non-obvious runtime controls:
- `SKIP_A_TIER=1`: skip strict-accuracy tier in correctness regression
- `PYTORCH_ALLOC_CONF` and `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`: auto-set in `vllm/__init__.py`

Legacy env vars (`FASTINFERENCE_KV_TYPE`, `FASTINFERENCE_FUSION_LEVEL`, `FASTINFERENCE_BLOCK_SIZE`, etc.) are deprecated. Use FastInferenceConfig or [tuning_keyvals] in TOML config instead.

## HPC / GPU Rules

- AMD Instinct and Radeon first (ROCm 7.2), CUDA compatibility maintained.
- PagedAttention uses 16-token physical blocks by default; reuse block-cache data rather than re-reading global memory.
- If a Triton kernel exceeds ~64 registers per thread, document the pressure and keep a lower-ILP fallback.
- Treat Triton compilation warnings as failures.
