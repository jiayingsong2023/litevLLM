# FP8

FastInference uses FP8 primarily as a runtime KV-cache precision option in the
lite path. Full upstream FP8 W8A8 model-quantization workflows are not part of
the maintained documentation surface here.

## Runtime KV Policy

Select FP8 KV cache through `FASTINFERENCE_CONFIG`:

```toml
profile = "accuracy"
kv_type = "fp8"
```

`kv_type` accepts:

- `auto`
- `fp16`
- `fp8`
- `turbo_int4`

The active runtime profile may also choose a KV dtype. `accuracy` currently
uses a conservative FP8 KV policy.

## Support Boundary

- FP8 KV cache is supported where the lite runtime and current model policy
  allow it.
- Dynamic FP8 weight quantization recipes from upstream vLLM are not documented
  as a FastInference support promise.
- New FP8 behavior must include correctness coverage for the affected model and
  cache path.

## Verification

```bash
bash tests/run_regression_suite.sh
bash tests/run_inference_correctness_regression.sh
```
