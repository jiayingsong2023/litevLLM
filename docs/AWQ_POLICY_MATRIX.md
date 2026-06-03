# AWQ Policy

AWQ safetensors are the main optimized weight path for FastInference. Current
runtime policy should be installed through profiles, adapter policy, and
`RuntimeConfig`; older `FASTINFERENCE_AWQ_*` names remain registered for
compatibility and benchmark tools.

## Maintained Models

| Model | AWQ Status | Notes |
| :--- | :--- | :--- |
| `Qwen3.5-9B-AWQ` | Supported | Covered by correctness regression. |
| `Gemma4-26B-A4B-it-AWQ-4bit` | Supported | MoE path, covered by Tier-B/A-lite and default A-strict unless skipped locally. |
| `Gemma4-31B-it-AWQ-4bit` | Supported | Dense path, covered by Tier-B/A-lite. |

## Policy Flow

```text
FastInferenceConfig / RuntimeProfile
  -> RuntimeConfig
  -> adapter policy
  -> quantization layer and Triton kernel launch policy
```

## Kernel Notes

The AWQ path includes specialized Triton kernels for decode GEMV, fused QKV,
fused gate/up, and selected Gemma4 dense/MoE shapes. Alternate kernel
strategies should stay tool-only until they pass correctness and end-to-end
performance gates.

## Verification

```bash
bash tests/run_regression_suite.sh
bash tests/run_inference_correctness_regression.sh
uv run python tests/e2e_full_benchmark.py
```
