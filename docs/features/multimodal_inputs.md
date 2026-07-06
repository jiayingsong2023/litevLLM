# Multimodal Inputs

Gemma4 single-image multimodal serving is supported in the lite runtime. The
supported path is Phase 2A: one request with one image, real `<image>`
placeholder expansion, Gemma4 vision tower embeddings, and placeholder
replacement during text prefill.

Broader multimodal serving remains experimental. Multi-image requests,
multi-request continuous batching, Qwen2VL, audio, and video require
model-specific implementation and validation before they should be treated as
supported.

## HTTP Request Shape

The chat endpoint accepts content blocks with text and `image_url` items:

```json
{
  "model": "models/<multimodal-model>",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {"type": "image_url", "image_url": {"url": "file:///path/to/image.png"}}
    ]
  }],
  "max_tokens": 64
}
```

The server converts text blocks into the prompt and image blocks into
`multi_modal_data` for the engine path.

## Current Boundary

- Gemma4 supports one image per request.
- Multimodal scheduling and observer counters exist.
- Multi-image and mixed multimodal batching remain outside the current support
  claim.
- Audio and video are unsupported unless model-specific code and tests are
  added. The upstream generic audio/parser helpers were removed during lite
  cleanup.

## Verification

```bash
uv run pytest -q tests/smoke
bash tests/run_regression_suite.sh
```
