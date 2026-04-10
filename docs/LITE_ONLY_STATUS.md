# Lite-Only Architecture Status

## Delivery Snapshot
As of 2026-04-09, within the current Lite runtime scope:

- `P0` delivered
- `P1` delivered
- `P2` delivered
- Explicit deferred item: `speculative decoding`

For an external-facing DONE/KNOWN_GAPS list, see:

- `docs/DELIVERY_STATUS.md`

## Current Project Position
FastInference is now a `lite-only` single-GPU inference project. The repository no longer treats `Qwen3.5-35B` as an officially supported model. The maintained support surface is the lite runtime path plus the current regression targets:

- `TinyLlama-1.1B`
- `Qwen3.5-9B-AWQ`

The project still reuses selected vLLM infrastructure modules, but it no longer aims to preserve a second full runtime path or a `vllm-compatible-lite` product boundary.

## Official Runtime Path
The only official runtime path is:

- Offline sync: `vllm.entrypoints.llm.LLM` -> `vllm.engine.lite_engine.LiteEngine`
- Async/server path: `vllm.engine.async_llm.AsyncLLM` -> `vllm.engine.lite_engine.LiteEngine`

This means offline and async generation share the same lite runtime core, but that core is now internally decomposed into:

- `vllm/serving/config_builder.py`
- `vllm/adapters/*`
- `vllm/engine/step_scheduler.py`
- `vllm/engine/request_scheduler.py`
- `vllm/engine/prefill_executor.py`
- `vllm/engine/decode_executor.py`
- `vllm/engine/sampling_driver.py`
- `vllm/engine/output_pipeline.py`
- `vllm/engine/runtime_observer.py`
- `vllm/engine/errors.py`

`LiteEngine` remains the orchestrator, but it is no longer the sole owner of request state, async loop management, sampling logic, output assembly, or prefill/decode execution.

## What Was Removed
- `Qwen3.5-35B` was removed from default docs, regression, and benchmark entrypoints.
- `Qwen3.5-35B`-specific output shaping, prompt guards, token biasing, and cleanup logic were removed from `output_processor.py`.
- Loader logic that forced 35B-specific AWQ high-fidelity behavior by model-path detection was removed.

## What Was Generalized
MoE infrastructure is no longer named as if it belongs only to `Qwen3.5-35B`.

Preferred runtime knobs are now:

- `FASTINFERENCE_MOE_FP8`
- `FASTINFERENCE_MOE_OFFLOAD`
- `FASTINFERENCE_MOE_PACKED_GGUF`
- `FASTINFERENCE_MOE_LRU_SIZE`

Legacy `FASTINFERENCE_QWEN35_MOE_*` names are still accepted as compatibility aliases, but new work should use the generic names.

## Compatibility Layers
- `vllm/worker/gpu_model_runner.py` is now a compatibility shim only. It should not be extended as a lite runtime entrypoint.
- Some upstream-style modules still exist for reuse, but they are not the official lite-only execution path.

## Regression Status
Validated after the current refactor set:

- `bash tests/run_regression_suite.sh`
- `SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh`
- `uv run pytest tests/test_async_runtime_contracts.py -q`
- `uv run pytest tests/test_step_scheduler.py -q`
- `uv run pytest tests/test_runtime_observer.py -q`

All passed on the current lite-only support surface.

---

# Adapter / Policy Onboarding Checklist

## Adapter Interfaces
Add a model adapter module under `vllm/adapters/`, for example `vllm/adapters/gemma4.py`, with responsibilities:

- expose head counts, KV head counts, head dim, layer count, and max model length
- declare whether the model uses dense attention, hybrid attention, or MoE routing
- define attention metadata expectations for prefill and decode
- expose model capability flags such as `supports_moe_offload`, `supports_packed_experts`, and `supports_fp8_moe`

## Loader Interfaces
Add or extend loader modules so the model can be loaded without adding Gemma-specific branches to engine core:

- checkpoint key mapping for Gemma 4 layer names
- A4B or AWQ quant metadata parsing
- expert weight materialization path for dense and MoE layers
- optional packed-expert loader path if the model benefits from packed storage
- policy hook to choose fused vs fallback AWQ/quant execution by module role, not by model-name string

## Policy Interfaces
Only add model policy code if the model truly needs it. Keep it out of `LiteEngine`.

Recommended interfaces:

- `prompt_policy`: optional prompt wrapping or chat-template normalization
- `logits_policy`: optional logit bias or stop-token shaping
- `output_policy`: optional output cleanup
- `moe_policy`: expert residency, CPU offload, FP8 enablement, packed-weight compatibility

## Test Requirements
Before claiming support, add:

- load smoke test
- one-step prefill correctness test vs reference
- Tier-B generation smoke
- MoE routing / expert weight correctness test if applicable
- benchmark entry only after correctness passes

## Design Rule
New model support should be implemented by adding adapter, loader, and policy modules. It should not reintroduce model-name-based conditionals inside `LiteEngine`, `step_scheduler.py`, executor modules, or generic loader hot paths.
