# Structured Outputs

Structured output support is present in the lite runtime and is currently
experimental. See [CAPABILITY_MATRIX.md](../CAPABILITY_MATRIX.md) for the
support status.

## Request Forms

The OpenAI-compatible chat endpoint accepts:

- native `structured_outputs`
- OpenAI-style `response_format.type = "json_object"`
- OpenAI-style `response_format.type = "json_schema"`

Example:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/Qwen3.5-9B-AWQ",
    "messages": [{"role": "user", "content": "Return {\"name\": \"Ada\"}."}],
    "response_format": {"type": "json_object"},
    "max_tokens": 32
  }'
```

Native structured-output example:

```json
{
  "structured_outputs": {
    "regex": "[A-Z]{3}-[0-9]{4}"
  }
}
```

## Maintainer Notes

- Keep grammar/request parsing at the serving boundary.
- Keep logits filtering and sampling behavior covered by focused tests.
- Do not claim broad upstream OpenAI compatibility without smoke and behavior
  coverage for the exact request shape.

## Verification

```bash
uv run pytest -q tests/smoke
bash tests/run_regression_suite.sh
```
