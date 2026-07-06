# Multimodal Inputs

Gemma4 image multimodal serving is supported in the lite runtime. The path uses
real `<image>` placeholder expansion, Gemma4 vision tower embeddings, and
placeholder replacement during text prefill. Multi-image requests and
multi-request continuous batching are covered by the lite multimodal path.

Qwen2VL image serving is implemented but remains experimental. It includes
Qwen2VL image preprocessing, `image_grid_thw`, mRoPE positions, real vision
tower embeddings, and placeholder replacement. Audio and video remain
unsupported.

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

- Gemma4 supports image requests, including multiple images per request and
  multiple image requests in the same prefill batch.
- Gemma4 multimodal LoRA supports text layers and the vision projector /
  connector.
- Qwen2VL supports image requests experimentally. Qwen2VL visual-tower LoRA is
  not supported; text-path LoRA remains the normal `LiteLinear` LoRA path after
  image embedding injection.
- Multimodal scheduling and observer counters exist.
- Audio and video are unsupported unless model-specific code and tests are
  added. The upstream generic audio/parser helpers were removed during lite
  cleanup.

## Verification

```bash
uv run --no-sync pytest tests/test_multimodal_processor.py tests/test_gemma4_multimodal.py tests/test_qwen2_vl_multimodal.py -q
bash tests/run_regression_suite.sh
```
