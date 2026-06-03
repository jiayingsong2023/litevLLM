# Multimodal Inputs

Multimodal serving exists in the lite runtime and is currently experimental.
The maintained path is image request plumbing for model-specific validation;
generic upstream audio/video parser helpers are not part of the current code
boundary. Validate model-specific behavior before treating it as
production-supported.

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

- Image request plumbing is present.
- Multimodal scheduling and observer counters exist.
- Multi-image behavior has test coverage, but broad real-traffic hardening is
  still required.
- Audio and video are unsupported unless model-specific code and tests are
  added. The upstream generic audio/parser helpers were removed during lite
  cleanup.

## Verification

```bash
uv run pytest -q tests/smoke
bash tests/run_regression_suite.sh
```
