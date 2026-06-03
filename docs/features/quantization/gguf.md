# GGUF

GGUF loading is kept as an experimental/compatibility quantization path. The
main optimized and supported weight path is Safetensors + AWQ. Check
[CAPABILITY_MATRIX.md](../../CAPABILITY_MATRIX.md) before claiming support for a
specific GGUF model.

## Detection

`vllm/serving/config_builder.py` detects `.gguf` files in a local model
directory and installs `GGUFConfig`.

```python
from vllm import LLM

llm = LLM(model="models/local-gguf-model")
```

## Boundary

- Validate each GGUF model with load, smoke, and correctness gates.
- Do not assume upstream GGUF model coverage.
- Prefer AWQ safetensors for the maintained performance path.

## Verification

```bash
bash tests/run_regression_suite.sh
uv run python tests/tools/quality_bar_spotcheck.py --model <path> --quant gguf
```
