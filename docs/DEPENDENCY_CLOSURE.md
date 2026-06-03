# Lite Runtime Dependency Boundary

This document summarizes the maintained import boundary around
`vllm/engine/lite_engine.py`. It is a living guide, not a historical deletion
report.

## Required For The Maintained Path

- `vllm/engine/*` - lite orchestration, scheduling, request lifecycle,
  observer, errors, backend policy, and async runtime contracts.
- `vllm/serving/config_builder.py` - config assembly.
- `vllm/adapters/*` - model capability and runtime policy.
- `vllm/model_executor/models/*` - maintained model implementations.
- `vllm/model_executor/model_loader/*` - local model loading.
- `vllm/model_executor/layers/*` - runtime layers and quantization helpers,
  excluding removed upstream pooling layers.
- `vllm/kernels/triton/*` - maintained Triton kernels.
- `vllm/triton_utils/*` - approved Triton import and utility surface.
- `vllm/config/*` - config dataclasses.
- `vllm/entrypoints/openai/*` - maintained HTTP serving entrypoint.
- `vllm/engine/sampling_driver.py`, `vllm/inputs/*`,
  `vllm/transformers_utils/*`, and `vllm/utils/*` - sampling, input, HF,
  and utility support.

## Removed Upstream Runtime Directories

These directories are not present and should not be reintroduced as maintained
runtime dependencies:

- `vllm/worker/`
- `vllm/core/`
- `vllm/distributed/`
- `vllm/executor/`
- `vllm/grpc/`
- `vllm/spec_decode/`
- `vllm/entrypoints/cli/`
- `vllm/entrypoints/pooling/`
- `vllm/model_executor/layers/pooler/`
- `vllm/sample/`
- `vllm/structured_output/`
- `vllm/third_party/triton_kernels/`

## Present Compatibility Or Vendored Code

The following paths exist today but are not independent product targets:

- `vllm/model_executor/warmup/` - compatibility or artifact path, not a
  worker-runtime warmup contract.
- Multimodal and LoRA runtime hooks - support varies by capability matrix
  status.

## Import Rules

- Engine and model hot paths must not import `vllm.worker`, `vllm.core`, or
  `vllm.distributed`.
- Triton imports go through `vllm/triton_utils/`.
- Runtime settings flow through `RuntimeConfig` and `attn_metadata["config"]`.
- Model-specific capability logic belongs in `vllm/adapters/`.

## Verification

Useful quick checks:

```bash
rg -n "vllm\\.(worker|core|distributed)" vllm tests docs -g '*.py' -g '*.md'
bash tests/run_regression_suite.sh
```
