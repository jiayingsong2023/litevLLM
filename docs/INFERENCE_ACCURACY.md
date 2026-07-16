# Inference Accuracy

This document defines the current FastInference quality gates. It intentionally
tracks only the maintained lite runtime support surface.

## Regression Targets

- `TinyLlama-1.1B`
- `Qwen3.5-9B-AWQ`
- `Gemma4-26B-A4B-it-AWQ-4bit`
- `Gemma4-31B-it-AWQ-4bit`
- `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
  when the target GGUF file is present

`Qwen3.5-35B` and generic GGUF alignment audits are not part of the default
regression surface. DeepSeek is a model-specific experimental Tier-B smoke, not
a generic GGUF correctness claim.

## Accuracy Tiers

| Tier | Goal | Entrypoint |
|------|------|------------|
| Tier-B | Output is readable, on-topic, and not structurally corrupted. | `tests/tools/quality_bar_spotcheck.py` |
| Gemma4 image quality | 26B/31B image requests produce expected color answers on synthetic fixtures. | `tests/tools/gemma4_multimodal_quality_spotcheck.py` |
| A-lite | Large-model smoke: model loads, generates, terminates, and produces non-empty text on fixed prompts. | `tests/tools/gemma4_single_prompt_smoke.py` |
| A-strict | Prefill logits / greedy behavior align with a Hugging Face reference where feasible. | `tests/verify_semantic_integrity.py` and `tests/tools/gemma4_prefill_strict_audit.py` |

Default policy:

- `<=14B`: Tier-B + A-strict.
- `>14B`: Tier-B + A-lite.
- Gemma4-26B keeps a default prefill-only A-strict audit unless locally disabled.
- Gemma4-26B and Gemma4-31B run the image multimodal quality spotcheck by
  default when their model directories are present. Gemma4 E4B is not part of
  this supported regression surface.

## Main Command

```bash
bash tests/run_inference_correctness_regression.sh
```

Useful options:

```bash
SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh
RUN_GEMMA4_31B=0 bash tests/run_inference_correctness_regression.sh
RUN_GEMMA4_26B=0 bash tests/run_inference_correctness_regression.sh
RUN_GEMMA4_26B_A_STRICT=0 bash tests/run_inference_correctness_regression.sh
RUN_DEEPSEEK_V4_FLASH_GPU_SMOKE=1 SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh
```

DeepSeek V4 Flash runs automatically when the target GGUF exists and
`RUN_DEEPSEEK_V4_FLASH_GPU_SMOKE` is `auto` or `1`; it skips with a warning when
the file is missing. The target GGUF is roughly 80.7GiB and the path is still
experimental. Override the path with `MODEL_DEEPSEEK_V4_FLASH_GGUF` when the
model lives outside `models/DeepSeek-V4-Flash-ds4/`.

DeepSeek V4 Flash uses adapter-owned executors and compressed KV lifecycle
components under the standard scheduler/output pipeline. Its standalone GPU
smoke reports decode throughput, but `stream_visible=0%` is expected because
that tool does not emit standard per-token streaming observer events.

The script suppresses tool debug logs by default, prints a compact
prompt/output summary for Tier-B spotchecks, and prints full captured output
only on failure. Set `FI_CORRECTNESS_VERBOSE=1` to restore full stage output.

## Tier-B Spotcheck

```bash
uv run python tests/tools/quality_bar_spotcheck.py \
  --model models/Qwen3.5-9B-AWQ \
  --quant awq \
  --prompt-subset minimal \
  --max-new-tokens 96 \
  --temperature 0
```

Tier-B is a coarse gate. A PASS means the output avoided known structural
failure patterns for the fixed prompts; it is not a full semantic proof.

## A-strict Semantic Integrity

```bash
uv run python tests/verify_semantic_integrity.py \
  --model models/Qwen3.5-9B-AWQ \
  --preset qwen35_9b_awq \
  --hf-model models/Qwen3.5-9B-FP16
```

For TinyLlama, use the local TinyLlama path and the `tinyllama` preset. For
Gemma4 large models, prefer the dedicated prefill strict audit because it can
load Lite and HF reference sequentially on one GPU.

## Gemma4 Large-Model Gates

A-lite:

```bash
uv run python tests/tools/gemma4_single_prompt_smoke.py \
  --model models/gemma-4-31B-it-AWQ-4bit \
  --prompt "Summarize what a binary search tree is in one short paragraph." \
  --max-new-tokens 32 \
  --temperature 0 \
  --max-model-len 512
```

A-strict prefill audit:

```bash
uv run python tests/tools/gemma4_prefill_strict_audit.py \
  --model models/gemma-4-26B-A4B-it-AWQ-4bit \
  --hf-device cuda \
  --max-model-len 256
```

Gemma4 prompt and baseline fixtures live under `tests/tools/fixtures/`.

Gemma4 image multimodal quality:

```bash
uv run python tests/tools/gemma4_multimodal_quality_spotcheck.py \
  --model models/gemma-4-26B-A4B-it-AWQ-4bit

uv run python tests/tools/gemma4_multimodal_quality_spotcheck.py \
  --model models/gemma-4-31B-it-AWQ-4bit
```

The default correctness regression runs those two checks after the corresponding
Gemma4 Tier-B text spotcheck. A passing result requires both synthetic image
cases to answer the expected color.

## Performance Is Separate

Accuracy gates do not replace performance regression. Use:

```bash
uv run python tests/e2e_full_benchmark.py
```

The benchmark records throughput, TTFT, runtime counters, warmup behavior, and
optional baseline warnings.

## Maintenance Rules

- Keep `tests/tools/` limited to tools used by maintained regression entrypoints
  or focused warn-only diagnostics.
- Do not add model families to this document until they are represented in the
  capability matrix and have an executable gate.
- If a correctness script path changes, update this file, `tests/README.md`,
  and `tests/run_inference_correctness_regression.sh` together.
