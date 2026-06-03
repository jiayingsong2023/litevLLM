# FastInference Lite Architecture

FastInference is a lite-only, single-GPU inference runtime. It keeps the
high-performance serving path from the vLLM-derived codebase while narrowing
the maintained architecture to pure Python plus Triton.

For feature and model status, use [CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md)
as the source of truth.

## Runtime Path

```text
LLM / AsyncLLM / OpenAI API Server
  -> vllm/serving/config_builder.py  # model/config assembly
  -> vllm/engine/lite_engine.py      # orchestration
  -> vllm/engine/step_scheduler.py   # step budget and batch selection
  -> vllm/engine/request_scheduler.py
  -> vllm/engine/prefill_executor.py
  -> vllm/engine/decode_executor.py
  -> vllm/engine/sampling_driver.py
  -> vllm/engine/output_pipeline.py
```

`LiteEngine` is the official execution path for both offline and async/server
use. Legacy upstream concepts such as workers, block managers, and distributed
executors are not a second supported runtime.

## Configuration Flow

`vllm/serving/config_builder.py` constructs `VllmConfig`, resolves
`FastInferenceConfig`, then creates `RuntimeConfig`.

```text
FASTINFERENCE_CONFIG / explicit config object
  -> FastInferenceConfig
  -> RuntimeProfileRegistry
  -> RuntimeConfig
  -> engine, scheduler, backend, model metadata
```

Production code should receive policy through `RuntimeConfig` or
`attn_metadata["config"]`. Direct `os.environ` reads in model layers or engine
hot paths are not part of the maintained configuration model.

## Major Boundaries

| Area | Responsibility |
| :--- | :--- |
| `vllm/engine/` | Control plane, scheduling, request lifecycle, runtime stats, errors. |
| `vllm/serving/` | Config assembly for offline and server entrypoints. |
| `vllm/adapters/` | Model capability and policy decisions. |
| `vllm/model_executor/models/` | Maintained lite model implementations, including the split Gemma4 package. |
| `vllm/kernels/triton/` | Maintained hand-written Triton kernels. |
| `vllm/triton_utils/` | Approved Triton import and utility layer. |
| `vllm/entrypoints/openai/` | Maintained HTTP server surface. |

## Model Layer Shape

Gemma4 is split into focused modules under
`vllm/model_executor/models/gemma4/`:

- `model.py`
- `attention.py`
- `mlp.py`
- `moe.py`
- `layer.py`
- `kv_utils.py`
- `rope.py`
- `config.py`
- `profiling.py`
- `policy_utils.py`

Model-specific policy belongs in `vllm/adapters/` and must not grow new
model-name branches inside the generic engine.

The upstream Transformers modeling backend wrappers are not part of the lite
runtime. New model support should add a focused lite model module plus adapter
policy and loader coverage instead of restoring generic upstream wrappers.

## Attention And Kernels

The maintained decode path uses Triton PagedAttention. Prefill uses the
current hardware-backed SDPA path where appropriate. AWQ decode and Gemma4
paths include specialized Triton kernels for fused QKV, fused gate/up, M=1
GEMV, and selected MoE decode shapes.

Every new Triton kernel must document memory layout and program tiling in ASCII
comments, use `vllm/triton_utils/` for Triton imports, and include correctness
coverage against a PyTorch reference.

## Compatibility Code

Some upstream-derived packages remain for imports, migration, or experimental
surface. Their existence does not make them official runtime targets:

- `vllm/model_executor/warmup/` may contain compatibility artifacts.
- Multimodal and LoRA runtime hooks are maintained only to the status listed in
  the capability matrix.

The removed upstream runtime directories are `vllm/worker/`, `vllm/core/`, and
`vllm/distributed/`; upstream CLI, pooling, gRPC, executor, spec decode, and
vendored third-party Triton kernel paths have also been removed. P2 cleanup also
removed broken upstream asset helpers, generic multimodal audio/parser helpers,
and Transformers modeling backend wrappers that were outside the lite closure.

## Observability And Errors

- `vllm/engine/runtime_observer.py` records runtime counters for prefix cache,
  fairness, preemption, LoRA, multimodal, and async-driver behavior.
- `vllm/engine/errors.py` centralizes runtime error semantics.
- `GET /debug/stats` exposes a compact API server summary.

## Validation

Use the following gates before claiming broad runtime changes are ready:

```bash
bash tests/run_regression_suite.sh
bash tests/run_inference_correctness_regression.sh
```

For kernel, KV-cache, and numerics changes, also run the relevant model
correctness and performance tools described in `tests/README.md`.
