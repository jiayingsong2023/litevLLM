# Gemma4-26B/31B Speculative Decoding Design

## Status

Revised: Phase 1 is now an **offline prototype tool**, not engine integration.

## Background

Previous work (`docs/superpowers/reports/2026-07-08-gemma4-26b-performance-retrospective.md`) concluded that Gemma4-26B BS=1 decode is near its engineering ceiling at ~11 tok/s. The bottleneck is kernel-launch overhead and small-kernel fragmentation, not a single operator. The remaining architecture-level options are batch decode and speculative decoding.

Before changing the engine, we must first prove that a draft source can produce enough accepted tokens per step to offset its own latency. This spec therefore starts with offline prototypes and defers engine integration until a draft strategy shows ≥20% theoretical speedup.

## Goals

- **P0**: build an offline n-gram speculative-decoding prototype for greedy (`temperature=0`) decoding.
  - No engine changes, no KV-cache sharing, no scheduler changes.
  - Outputs: `acceptance_rate`, `projected_tps`, `bit-exact` vs. baseline.
- **P1**: replace n-gram with a tokenizer-compatible small Gemma draft model (E2B first, E4B fallback).
  - Still offline, still greedy, still no engine changes.
  - Gate: `effective_tps >= baseline * 1.2` on a representative prompt set.
- **P2** (future, only if P1 passes): integrate the winning draft strategy into `LiteEngine` / `AsyncLLM` with proper KV-cache draft/target sharing and rollback.

## Non-Goals

- Engine integration in P0/P1.
- KV-cache rollback in P0/P1.
- Non-greedy sampling in P0/P1.
- Batched speculative decoding in P0/P1.
- OpenAI API server support in P0/P1.
- Custom trained draft heads (Eagle/Medusa/etc.).

## Constraints

- Python 3.12, `uv run`.
- Lite-engine target model only for baseline measurements.
- No runtime `os.environ` reads in `vllm/`.
- Do not add permanent diagnostic hooks to the hot path.
- P0/P1 code lives in `tests/tools/` as offline experiments.

## Terminology

- **K**: maximum number of draft tokens proposed per step.
- **Draft source**: produces candidate token ids without running the target model.
- **Verify**: run the target model on a sequence that includes the draft tokens and obtain target logits at the positions immediately preceding each draft token.
- **Accept**: compare draft token `d_i` with the token sampled/argmaxed from target logits at position `len(prefix) + i - 1`; accept while they match.
- **Rollback**: discard KV-cache blocks for rejected draft tokens and revert sequence length (P2 only).

## Phase 0: Offline N-gram Prototype

### Purpose

Validate the speculative-decoding acceptance loop and measurement methodology without touching the engine.

### Approach

The prototype must obtain per-position target logits, which `LLM.generate()` does not expose. Use a tool-only helper that calls the model forward directly:

```python
def run_target_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return logits for every input position (shape [1, seq_len, vocab_size])."""
    ...
```

1. **Baseline run**: target model greedy-decodes the prompt to completion. Save the reference token sequence.
2. **Speculative run**: for each step:
   - Let `prefix = prompt + accepted_tokens`.
   - Use n-gram lookup on `prefix` to propose up to K draft tokens.
   - Run `run_target_logits(target_model, prefix + d_1 + ... + d_K)`.
   - For i = 1..K, compare `d_i` with `argmax(target_logits[0, len(prefix) + i - 1])`.
   - Accept while equal; on first mismatch take the target argmax token as the next token and continue.

Because this is a prototype, the target forward may use the full prefix each step (slower than a real engine, but correctness-valid). No KV cache is reused between steps.

### N-gram lookup

```python
def propose_ngram(
    prefix_token_ids: list[int],
    generated_token_ids: list[int],
    k: int,
    ngram_min: int = 2,
    ngram_max: int = 4,
) -> list[int]:
    needle = generated_token_ids[-n:] for n from ngram_max down to ngram_min
    find the most recent earlier occurrence of needle in prefix_token_ids + generated_token_ids
    return the following min(k, available) tokens, or empty list if no match
```

### CLI / tool

- Location: `tests/tools/gemma4_speculative_prototype.py`
- Arguments:
  - `--target-model` (default `models/gemma-4-26B-A4B-it-AWQ-4bit`)
  - `--prompt` or `--prompt-file`
  - `--max-new-tokens`
  - `--num-draft-tokens` (K)
  - `--ngram-min`, `--ngram-max`
  - `--json-out`
- Outputs JSON:
  - `baseline_tokens`, `speculative_tokens`, `bit_exact`
  - `accepted_total`, `proposed_total`, `acceptance_rate`
  - `baseline_tps`, `speculative_tps`, `projected_tps` (theoretical, ignoring prototype overhead)

### Success criteria

- `bit_exact == true` for `temperature=0`.
- Tool runs end-to-end without engine modifications.
- Acceptance rate is measured and reported honestly (may be low for non-repetitive prompts).

## Phase 1: Offline E2B/E4B Draft Prototype

### Purpose

Determine whether a small Gemma draft model can beat the n-gram baseline and cross the 20% speedup gate.

### Draft model candidates

1. **Primary**: `gemma-4-E2B-it` (smaller, faster draft; lower acceptance but potentially better net speedup).
2. **Fallback**: `gemma-4-E4B-it` (larger, higher acceptance; draft latency may eat gains).

Tokenizer compatibility between the draft and target models is **not assumed**; it is a P1 gate. Before measuring speedup, verify that the draft and target tokenizers have identical vocab sizes, special token IDs, and encode/decode round-trips. If any check fails, stop and do not proceed to P2.

### Approach

Same offline loop as P0, but the draft source becomes a small `LLM` instance:

1. Load target model and draft model as two separate `LLM` instances.
2. For each step:
   - Run draft model greedy-decode for K tokens from current prefix.
   - Run target model once on `prefix + d_1 + ... + d_K`.
   - Accept/reject as in P0.

No KV sharing between draft and target in P1; each forward is independent. This is slower than the eventual engine integration but isolates the question: **does the draft model produce enough accepted tokens to theoretically justify the engine work?**

### Measurement

- `effective_tps` is computed from real per-step measurements, including the verifier cost:
  - `draft_k_time`: wall time for the draft model to produce K tokens.
  - `target_verify_k_time`: wall time for one target forward on `prefix + d_1 + ... + d_K`.
  - `accepted_or_recovered_tokens = accepted_count + 1` (the guaranteed target token, which may be a recovered token at the first rejected position).
  - `effective_tps = accepted_or_recovered_tokens / (draft_k_time + target_verify_k_time)`.
- Compare to measured baseline TPS.

### Gate

- `effective_tps >= baseline_tps * 1.2` on a representative prompt set.
- If neither E2B nor E4B passes, **stop**. Do not proceed to P2 engine integration.

## Phase 2: Engine Integration

Only enter P2 if P1 proves ≥20% theoretical speedup.

### Scope

Integrate the winning draft source into `LiteEngine` for BS=1 greedy decode:

- Add speculative configuration to `VllmConfig` (reusing existing upstream-like `speculative_config` semantics if present; otherwise extend cleanly).
- Support multi-token decode inputs in `InputBatchBuilder`.
- Add speculative fast path to `LiteSingleGpuBackend` for single running requests.
- Keep a separate draft KV cache; implement target tentative KV-cache allocation and rollback in the block manager.
- Preserve `AsyncLLM` streaming semantics.

### Verify alignment (off-by-one)

For draft tokens `d_1 .. d_K` appended after prefix `p`:

- Target forward input: `p, d_1, d_2, ..., d_K`.
- Logits index for validating `d_i` is `len(p) + i - 1` (the logit produced *before* seeing `d_i`).
- If `argmax(logits[len(p) + i - 1]) == d_i`, accept `d_i`; otherwise reject and use `argmax(logits[len(p) + i - 1])` as the next token.

### KV rollback

- Before verify: tentatively allocate KV blocks for K draft positions.
- After accepting N tokens: keep first N tentative blocks, free the rest.
- After reject at position i: keep first i tentative blocks, free i+1..K, revert sequence length.

### Batching behavior

In P2, speculative decoding is only enabled for exactly one running decode request. Batch > 1 falls back to normal single-token decode.

## API (P2 only)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="models/gemma-4-26B-A4B-it-AWQ-4bit",
    speculative_config={
        "method": "ngram",          # or "draft_model"
        "num_draft_tokens": 5,
        "draft_model": "models/gemma-4-E2B-it",
    },
)

outputs = llm.generate("...", SamplingParams(temperature=0))
```

P0/P1 use CLI arguments only; no public API changes.

## Testing plan

| Phase | Test | Pass Criteria |
|---|---|---|
| P0 | `tests/tools/gemma4_speculative_prototype.py` | `bit_exact == true` for `temperature=0`; acceptance rate reported. |
| P0 | Run on repetitive and non-repetitive prompts | Tool does not crash; results are recorded. |
| P1 | `--draft-model models/gemma-4-E2B-it` | Effective TPS ≥ baseline * 1.2 on representative prompts. |
| P1 | `--draft-model models/gemma-4-E4B-it` | Compared against E2B; best model chosen. |
| P2 | Greedy bit-exact vs. baseline | Identical token sequence. |
| P2 | `tests/e2e_full_benchmark.py` | 26B median decode TPS ≥ 13.0. |
| P2 | KV rollback stress test | Long prompt + forced mismatches: no OOM, no crash. |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| N-gram acceptance is near zero on chat prompts | Expected; P0 is a measurement/baseline phase, not a performance target. |
| E2B draft too slow to justify engine work | Gate is `effective_tps >= 1.2x`; if not met, stop. |
| Tokenizer mismatch between draft and target | P1 gate: identical vocab size, special token IDs, and encode/decode round-trip; stop if any mismatch. |
| Off-by-one in verify logic | P0 prototype bit-exact check catches any alignment bug. |
| Engine integration breaks streaming / batching | P2 limited to BS=1 greedy; batch path unchanged. |

## Success criteria

- P0: offline n-gram prototype merged to `tests/tools/`, bit-exact, acceptance rate measurable.
- P1: E2B/E4B prototype proves ≥20% theoretical speedup, or we stop and document why.
- P2: only if P1 passes; engine integration with bit-exact greedy correctness and ≥13 tok/s on 26B.
