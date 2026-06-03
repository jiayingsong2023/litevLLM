# FastInference API Reference

This page documents the maintained HTTP surface in
`vllm.entrypoints.openai.api_server`.

## Start The Server

```bash
uv run python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3.5-9B-AWQ \
  --host 0.0.0.0 \
  --port 8000
```

The server uses `vllm/serving/config_builder.py` to build `VllmConfig` and
`RuntimeConfig`, then serves requests through `AsyncLLM` and `LiteEngine`.

Optional load-time policy mode:

```bash
uv run python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3.5-9B-AWQ \
  --policy-mode auto
```

`--policy-mode` accepts `auto`, `aggressive`, or `stable`. Runtime profiles and
KV policy should be configured with `FASTINFERENCE_CONFIG` and TOML.

## Runtime Config File

```toml
profile = "latency"
kv_type = "fp8"

[tuning_keyvals]
FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS = "1"
FASTINFERENCE_KV_MAX_MODEL_LEN = "512"
```

```bash
FASTINFERENCE_CONFIG=configs/latency.toml \
uv run python -m vllm.entrypoints.openai.api_server \
  --model models/gemma-4-26B-A4B-it-AWQ-4bit
```

## GET `/v1/models`

Returns the currently loaded model in OpenAI-style list form.

```bash
curl -s http://127.0.0.1:8000/v1/models
```

## POST `/v1/chat/completions`

Minimum request:

```json
{
  "model": "models/Qwen3.5-9B-AWQ",
  "messages": [{"role": "user", "content": "Hello"}]
}
```

Common fields:

| Field | Type | Notes |
| :--- | :--- | :--- |
| `stream` | boolean | Server-sent events when `true`. |
| `max_tokens` | integer | Defaults to the server sampling default. |
| `temperature` | number | Passed into `SamplingParams`. |
| `top_p` | number | Passed into `SamplingParams`. |
| `structured_outputs` | object | Native structured output request. |
| `response_format` | object | OpenAI-compatible JSON object/schema mapping. |

Non-streaming example:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/Qwen3.5-9B-AWQ",
    "messages": [{"role": "user", "content": "Give me a three-step plan."}],
    "stream": false,
    "max_tokens": 64
  }'
```

Streaming example:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/Qwen3.5-9B-AWQ",
    "messages": [{"role": "user", "content": "Count to five."}],
    "stream": true,
    "max_tokens": 32
  }'
```

JSON response-format example:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/Qwen3.5-9B-AWQ",
    "messages": [{"role": "user", "content": "Return {\"ok\": true}."}],
    "response_format": {"type": "json_object"},
    "max_tokens": 32
  }'
```

## Multimodal Message Content

The chat endpoint accepts OpenAI-style content blocks with text and `image_url`
items. Multimodal support is experimental; validate model-specific behavior
before treating it as production-ready.

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {"type": "image_url", "image_url": {"url": "file:///path/to/image.png"}}
    ]
  }]
}
```

## Runtime Stats

`GET /debug/stats` returns a compact summary of runtime observer counters,
including prefix cache, preemption, fairness, LoRA, async driver, and
multimodal counters.

```bash
curl -s http://127.0.0.1:8000/debug/stats
```

## Error Shape

Errors use a compact OpenAI-style envelope:

```json
{
  "error": {
    "message": "error detail",
    "type": "BadRequestError",
    "param": null,
    "code": 400
  }
}
```

## API Verification

```bash
uv run pytest -q tests/smoke
```
