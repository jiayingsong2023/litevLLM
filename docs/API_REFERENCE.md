# FastInference API Reference

This page documents the maintained HTTP surface in
`vllm.entrypoints.openai.api_server`.

The server is OpenAI-compatible for the lite-supported chat subset. It is not a
drop-in implementation of every OpenAI or upstream vLLM route.

It binds to `127.0.0.1` by default. Public binding is explicit and should sit
behind an authenticated, rate-limited reverse proxy.

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

`GET /healthz` reports process liveness. `GET /readyz` returns 503 until the
engine is initialized or after it enters fatal state.

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
| `max_completion_tokens` | integer | Alias for `max_tokens`. |
| `temperature`, `top_p`, `top_k`, `min_p` | number | Passed into `SamplingParams`. |
| `frequency_penalty`, `presence_penalty`, `repetition_penalty` | number | Passed into `SamplingParams`. |
| `structured_outputs` | object | Native structured output request. |
| `response_format` | object | OpenAI-compatible JSON object/schema mapping. |

The loaded model name must exactly match `model`. All message history is passed
through the loaded tokenizer's chat template. The server supports `n=1`; it
rejects stop strings, logprobs, and seed rather than silently ignoring them.
Non-streaming responses include `usage`; terminal choices report `stop` or
`length`.

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
items. Multimodal support is experimental; only bounded `data:image/...;base64`
URLs are accepted by default. Local paths and network URLs are rejected.

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

## Runtime Stats

Debug routes are disabled by default. Start the server with
`--enable-debug-endpoints` only on a trusted listener.

`GET /debug/runtime_stats` returns a compact summary of runtime observer
counters, including prefix cache, preemption, fairness, LoRA, async driver, and
multimodal counters.

```bash
curl -s http://127.0.0.1:8000/debug/runtime_stats
```

`POST /debug/runtime_stats/reset` resets runtime counters. Set
`clear_prefix_cache=true` to clear prefix-cache state as part of the reset.

```bash
curl -s -X POST \
  'http://127.0.0.1:8000/debug/runtime_stats/reset?clear_prefix_cache=true'
```

## Compatibility Boundary

Supported OpenAI-compatible routes in the standalone server are
`GET /v1/models` and `POST /v1/chat/completions`. Unsupported routes include
`/v1/responses`, `/v1/completions`, `/v1/embeddings`, pooling, score, and
rerank APIs.

## Errors

Input and unsupported-parameter errors return HTTP 400. Admission pressure
returns 429; a fatal or uninitialized runtime returns 503. Responses use the
standard FastAPI `{"detail": "..."}` error body.

## API Verification

```bash
uv run pytest -q tests/smoke
```
