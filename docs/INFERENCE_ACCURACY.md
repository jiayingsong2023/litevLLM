# Inference Accuracy

This document defines the current FastInference quality gates. It intentionally
tracks only the maintained lite runtime support surface.

## Regression Targets

- `TinyLlama-1.1B`
- `Qwen3.5-9B-AWQ`
- `Gemma4-26B-A4B-it-AWQ-4bit`
- `Gemma4-31B-it-AWQ-4bit`

`Qwen3.5-35B`, GGUF-specific alignment audits, and DeepSeek diagnostics are not
part of the default regression surface.

## Accuracy Tiers

| Tier | Goal | Entrypoint |
|------|------|------------|
| Tier-B | Output is readable, on-topic, and not structurally corrupted. | `tests/tools/quality_bar_spotcheck.py` |
| A-lite | Large-model smoke: model loads, generates, terminates, and produces non-empty text on fixed prompts. | `tests/tools/gemma4_single_prompt_smoke.py` |
| A-strict | Prefill logits / greedy behavior align with a Hugging Face reference where feasible. | `tests/verify_semantic_integrity.py` and `tests/tools/gemma4_prefill_strict_audit.py` |

Default policy:

- `<=14B`: Tier-B + A-strict.
- `>14B`: Tier-B + A-lite.
- Gemma4-26B keeps a default prefill-only A-strict audit unless locally disabled.

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
```

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
