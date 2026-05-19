# FastInference Capability Matrix

This document is the source of truth for the current FastInference lite-only
support surface. Other docs should link here instead of carrying separate model
or feature status lists.

## Status Labels

| Status | Meaning |
| :--- | :--- |
| Supported | Maintained in the lite runtime and covered by the default correctness or smoke gates. |
| Experimental | Implemented or partially optimized, but not a default support promise. |
| Compatibility | Kept to preserve imports, API shape, or migration paths; not a second runtime target. |
| Unsupported | Outside the current lite-only product boundary. |

## Runtime And Hardware

| Area | Status | Notes |
| :--- | :--- | :--- |
| Single-GPU lite runtime | Supported | Official path through `LiteEngine`, `RuntimeController`, and the lite backend. |
| AMD ROCm | Supported | Primary tuning target. Current Gemma4 baselines use the default benchmark recommended profile. |
| CUDA compatibility | Supported | Maintained where the Python + Triton path supports it. |
| Multi-GPU distributed runtime | Unsupported | Distributed shims may remain for imports, but are not an official execution path. |
| C++/CUDA extension source | Unsupported | Enforced by pre-commit outside allowed third-party code. |

## Model Support

| Model / Family | Status | Regression Basis |
| :--- | :--- | :--- |
| TinyLlama-1.1B | Supported | Tier-B quality spotcheck and A-strict semantic integrity. |
| Qwen3.5-9B-AWQ | Supported | Tier-B quality spotcheck and A-strict AWQ-vs-FP16 audit. |
| Gemma4-26B-A4B-it-AWQ-4bit | Supported | Tier-B, A-lite, and default A-strict audit unless locally disabled. |
| Gemma4-31B-it-AWQ-4bit | Supported | Tier-B and A-lite; A-strict remains manual/specialized. |
| Llama-like models outside the regression set | Experimental | Adapter fallback exists, but support should be claimed only after model-specific smoke and correctness gates. |
| LoRA runtime | Experimental | Runtime path and tests exist; production policy calibration is still workload dependent. |
| Multimodal serving | Experimental | Single-image and multi-image paths exist, but require broader real-traffic hardening. |
| Legacy upstream vLLM model surface | Unsupported | This project no longer aims to preserve full upstream model compatibility. |

## Feature Support

| Feature | Status | Notes |
| :--- | :--- | :--- |
| Safetensors + AWQ loading | Supported | Main optimized weight path. |
| FP8 KV cache | Supported | Accuracy guard default for Gemma4 when int4 KV is not explicitly allowed. |
| TurboQuant INT4 KV cache | Supported | Default runtime policy for non-guarded models; guarded by model policy where needed. |
| Gemma4 recommended adapter profile | Supported | Gemma4 installs AWQ decode GEMV and fused gate-up defaults; dense Gemma4 also installs group32 GEMV and dense down-proj defaults. |
| Gemma4-26B MoE int4 decode kernel | Supported | Default strategy is `batched_chunked`; slower experimental strategies remain opt-in through `FASTINFERENCE_GEMMA4_MOE_INT4_KERNEL_STRATEGY`. |
| PagedAttention decode | Supported | Triton path with selective-attention experiments available. |
| Prefix cache | Experimental | Minimal runtime and observability exist; defaults still need workload calibration. |
| Structured outputs | Experimental | Grammar-backed behavior is present and tested, but broad API compatibility should stay gated. |
| Speculative decoding | Unsupported | Explicitly deferred. |
| Full upstream OpenAI/vLLM compatibility | Unsupported | Only the lite-compatible serving surface should be treated as supported. |

## Test Entrypoints

| Layer | Command | Purpose |
| :--- | :--- | :--- |
| Smoke | `uv run pytest -q tests/smoke` | Import, routing, and no-model HTTP sanity checks. |
| Fast regression | `bash tests/run_regression_suite.sh` | Unit and structural smoke tests without full model loads. |
| Correctness | `bash tests/run_inference_correctness_regression.sh` | Local-model quality and semantic gates. |
| Performance | `uv run python tests/e2e_full_benchmark.py` | Throughput, TTFT, and profile-level performance baselines. |
