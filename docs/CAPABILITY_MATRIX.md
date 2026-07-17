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
| AMD ROCm | Supported | Primary tuning target. Current Gemma4 baselines use the default balanced profile. |
| CUDA compatibility | Supported | Maintained where the Python + Triton path supports it. |
| Adapter-owned custom runtime components | Experimental | Used by DeepSeek V4 Flash GGUF for model-local executors and compressed KV lifecycle; the generic engine remains model-name agnostic. |
| Multi-GPU distributed runtime | Unsupported | `vllm/distributed/` is not part of the current code tree or execution path. |
| C++/CUDA extension source | Unsupported | Enforced by pre-commit outside allowed third-party code. |

## Supported Python And Service Surface

Only the following import and service entrypoints are compatibility promises.
Everything else under `vllm/` is an internal implementation detail and may be
removed once it is outside this surface's import closure and has no regression
consumer.

| Surface | Status | Contract |
| :--- | :--- | :--- |
| `vllm.LLM`, `SamplingParams`, `PoolingParams`, `TextPrompt`, `TokensPrompt`, `clear_cache` | Supported | Public single-GPU Python inference API exported by `vllm.__all__`. |
| `vllm.engine.async_llm.AsyncLLM` | Supported | Async engine used by serving; direct use is supported for the lite runtime only. |
| `vllm.serving.config_builder.build_vllm_config` | Compatibility | Programmatic construction of the lite runtime configuration. |
| `vllm.entrypoints.openai.api_server` | Supported | OpenAI-compatible REST server: `/v1/models` and `/v1/chat/completions`. |
| `vllm.entrypoints.api_server` | Compatibility | Legacy import that re-exports the supported OpenAI server. |
| `vllm.entrypoints.serve.tokenize` | Compatibility | Router attachment only; not a standalone server contract. |

No other upstream `vllm.*` imports, CLI surfaces, model families, distributed
features, or plugin interfaces are compatibility promises.

## Model Support

| Model / Family | Status | Regression Basis |
| :--- | :--- | :--- |
| TinyLlama-1.1B | Supported | Tier-B quality spotcheck and A-strict semantic integrity. |
| Qwen3.5-9B-AWQ | Supported | Tier-B quality spotcheck and A-strict AWQ-vs-FP16 audit. |
| Gemma4-12B-it-AWQ-INT4 | Experimental | FP8 KV M=1 regression and M=2/M=4 token-ID parity diagnostics are maintained. The production scheduler advertises M=1 only: the M=4 path fails the per-agent p95 latency gate. |
| Gemma4-26B-A4B-it-AWQ-4bit | Supported | Tier-B, A-lite, default A-strict audit unless locally disabled, and Gemma4 image multimodal quality spotcheck. |
| Gemma4-31B-it-AWQ-4bit | Supported | Tier-B, A-lite, and Gemma4 image multimodal quality spotcheck. A-strict remains manual/specialized. |
| Gemma4 image multimodal | Supported | Gemma4 26B/31B image quality is covered by the default correctness regression. The path includes prompt placeholder expansion, official Gemma4 image patch preprocessing, Gemma4 vision tower embeddings, placeholder replacement in text prefill, multi-image requests, multi-request continuous batching, and Gemma4 projector LoRA. Gemma4 E4B is not in the supported regression surface. |
| Qwen2VL image multimodal | Experimental | Qwen2VL image preprocessing, `image_grid_thw`, mRoPE positions, real vision tower, and placeholder replacement are implemented and covered by focused tests. Vision-tower LoRA is not supported. |
| DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf | Experimental | Native DS4 GGUF target with adapter-owned executors and compressed KV lifecycle under the shared AsyncLLM/LiteEngine/RuntimeController path. Raw, compressed, and indexer KV buffers remain model-local rather than using the flat block pool. When the target file is present, the default correctness script runs its Tier-B smoke; e2e remains a manual benchmark. Current validation target is 4K context. The latest short-prompt isolated e2e measured 1.78 decode tok/s (single run, 2026-07-17); this is not a production-speed parity claim. |
| Llama-like models outside the regression set | Experimental | Adapter fallback exists, but support should be claimed only after model-specific smoke and correctness gates. |
| LoRA runtime | Experimental | Safetensors PEFT adapters, text-layer LoRA, `LoRAMapping`, and mixed LoRA batches are implemented and tested. Production policy calibration is still workload dependent. |
| Multimodal serving | Experimental | Gemma4 image serving is maintained; Qwen2VL image serving is implemented but remains experimental pending broader real-checkpoint smoke coverage. Audio and video are unsupported. |
| Upstream Transformers modeling backend | Unsupported | The broken `vllm/model_executor/models/transformers/` wrappers were removed; maintained models live under `vllm/model_executor/models/` with adapter-owned policy. |
| Generic upstream asset downloader | Unsupported | The `vllm/assets/` helper package was removed; tests and demos should use explicit local fixtures or model paths. |
| Generic multimodal audio/video parser helpers | Unsupported | Lite multimodal support is image-request plumbing only unless a model-specific implementation and regression gate are added. |
| Upstream elastic scaling middleware and certificate hot reload | Unsupported | `vllm.entrypoints.serve.elastic_ep` and `vllm.entrypoints.ssl` are not part of the lite server and are removed. |
| Legacy upstream vLLM model surface | Unsupported | This project no longer aims to preserve full upstream model compatibility. |

## Feature Support

| Feature | Status | Notes |
| :--- | :--- | :--- |
| Safetensors + AWQ loading | Supported | Main optimized weight path. |
| FP8 KV cache | Supported | Accuracy guard default for Gemma4 when int4 KV is not explicitly allowed. |
| TurboQuant INT4 KV cache | Supported | Default runtime policy for non-guarded models; guarded by model policy where needed. |
| Gemma4 recommended adapter profile | Supported | Gemma4 installs AWQ decode GEMV and fused gate-up defaults through runtime profile / adapter policy; dense Gemma4 also installs group32 GEMV and dense down-proj defaults. |
| Gemma4-26B MoE int4 decode kernel | Supported | Supported through the balanced / latency profile family with the current default strategy; alternate decode strategies remain benchmark-tool experiments, not production runtime switches. |
| Split-K / atomic-reduce kernels | Experimental | Atomic reduction order is non-deterministic. They require tolerance-based GPU parity tests and are not bit-exact contracts. |
| Gemma4-26B AWQ grouped prefill MoE | Experimental | Profile-guided grouped prefill is available for the validated MoE shapes; alternate grouped / fused variants remain benchmark-tool experiments rather than production runtime controls. |
| OpenAI-compatible REST serving | Supported | Lite subset through `vllm.entrypoints.openai.api_server`: `GET /v1/models` and `POST /v1/chat/completions` with streaming and non-streaming responses. |
| Tokenize/detokenize REST router | Compatibility | Maintained under `vllm.entrypoints.serve.tokenize` when attached; not part of the standalone OpenAI API server contract. |
| PagedAttention decode | Supported | Triton path with selective-attention experiments available for standard paged-KV models. DeepSeek V4 Flash uses a separate compressed-paged KV implementation. |
| DeepSeek V4 Flash direct kernels | Experimental | Current kept optimizations are semantic-preserving: direct selected Q2/IQ2 payload kernels, Q8_0 raw sign-extension trim, fused indexer-select scale, profiler breakdown, and Triton indexer QAT. Rejected experiments include graph/capture, full expert GPU tables, Q2 static unroll, batched Q8 raw matvec, and compressor dual Q8 projection. |
| Step scheduler metrics split | Supported | `StepPlan` carries execution fields; `StepPlanMetrics` carries observer/debug counters. |
| Decode batch tensor reuse | Supported | Non-fast decode batch construction reuses `InputBatchBuilder` scratch tensors for core metadata tensors. |
| Prefix cache | Experimental | Block-level prefix-cache capture/materialize is implemented in `KVBlockManager` for the standard PagedAttention path; default policy calibration remains workload dependent. |
| Structured outputs | Experimental | Grammar-backed behavior is present and tested, but broad API compatibility should stay gated. |
| Multimodal + LoRA | Experimental | Gemma4 text layers and vision projector/connector can receive LoRA in multimodal prefill. Qwen2VL text-path LoRA uses the normal `LiteLinear` path after image embedding injection; Qwen2VL visual tower LoRA is unsupported. |
| Speculative decoding | Unsupported | Explicitly deferred. |
| Full upstream OpenAI/vLLM compatibility | Unsupported | Only the lite-compatible serving surface should be treated as supported. |

## Test Entrypoints

| Layer | Command | Purpose |
| :--- | :--- | :--- |
| Smoke | `uv run pytest -q tests/smoke` | Import, routing, and no-model HTTP sanity checks. |
| Fast regression | `bash tests/run_regression_suite.sh` | Unit and structural smoke tests without full model loads. |
| Correctness | `bash tests/run_inference_correctness_regression.sh` | Local-model quality and semantic gates. |
| Performance | `uv run python tests/e2e_full_benchmark.py` | Throughput, TTFT, profile-level baselines, async runtime counters, DeepSeek GGUF smoke benchmark, and optional baseline warning reports. |
