# Gemma4-26B/31B Speculative Decoding Design

## Status

Revised after fourth review. Phase boundaries:

- **P0**: offline n-gram prototype — validate algorithm only.
- **P1**: tokenizer / memory / acceptance sweep with real draft models.
- **P1.5**: cached multi-token verifier microbenchmark — measures a **multi-step speculative run** with both models resident, real KV/block-table mutation, draft catch-up, and bit-exact verification. This is the only performance gate.
- **P2**: engine integration — only if P1.5 predicts ≥1.2x decode speedup across all context buckets.
- **P3**: broader sampling, batching, API surface.

## Background

Previous work (`docs/superpowers/reports/2026-07-08-gemma4-26b-performance-retrospective.md`) concluded that Gemma4-26B BS=1 decode is near its engineering ceiling at ~11 tok/s. The bottleneck is kernel-launch overhead and small-kernel fragmentation, not a single operator. The remaining architecture-level options are batch decode and speculative decoding.

Before changing the engine, we must prove that a draft source can produce enough accepted tokens per step to offset its own latency, **and** that a cached K+1 verifier is actually faster than K separate single-token decodes. This spec therefore keeps engine integration at P2 and uses P1.5 to measure a multi-step speculative run in a realistic cached-KV configuration.

## Goals

- **P0**: build an offline n-gram speculative-decoding prototype for greedy (`temperature=0`) decoding.
  - No engine changes, no KV-cache sharing, no scheduler changes.
  - Outputs: `acceptance_rate`, `bit-exact` vs. baseline.
  - Does **not** output decision-making `projected_tps`.
- **P1**: replace n-gram with a tokenizer-compatible small Gemma draft model (E2B first, E4B fallback).
  - Still offline, still greedy, no engine changes.
  - Verify tokenizer ID-level compatibility and memory feasibility.
  - Run a fixed prompt set and report acceptance statistics.
- **P1.5**: measure a multi-step speculative run end-to-end.
  - Both target and draft models are resident.
  - Reuse KV from a real prefill for both models.
  - Actually mutate seq_len and block tables during the run (commit + truncate + draft catch-up).
  - Run enough steps to observe acceptance distribution, partial rejects, and all-accepted catch-up.
  - Verify bit-exact against target baseline for every prompt, K, and context bucket.
  - Predict decode TPS and gate on aggregate `predicted_decode_tps >= baseline_decode_tps * 1.2` across context buckets.
  - Requires a **minimal verifier primitive** in `vllm/model_executor/models/gemma4/` (model forward + input metadata only); no scheduler/backend/API changes.
- **P2** (future, only if P1.5 passes): integrate into `LiteEngine` / `AsyncLLM` for BS=1 plain-greedy decode.

## Non-Goals

- Engine integration before P1.5 gate passes.
- KV-cache rollback before P2.
- Non-greedy sampling, repetition/frequency penalty, structured output, LoRA, or multimodal in P2 first cut.
- Batched speculative decoding in P2.
- OpenAI API server support in P2.
- Custom trained draft heads (Eagle/Medusa/etc.).

## Constraints

- Python 3.12, `uv run`.
- Lite-engine target model only for baseline measurements.
- No runtime `os.environ` reads in `vllm/`.
- Do not add permanent diagnostic hooks to the hot path.
- P0/P1 code lives in `tests/tools/` as offline experiments.
- P1.5 may add a minimal, gated verifier primitive in `vllm/model_executor/models/gemma4/` but must not change scheduler, backend, or public API.

## Terminology

- **K**: maximum number of draft tokens proposed per step.
- **Draft source**: produces candidate token ids without running the target model.
- **Verifier**: a single target forward that consumes the cached prefix plus K+1 new tokens and returns K+1 logits.
- **Accept**: compare draft token `d_i` with the token sampled/argmaxed from target logits at the position immediately preceding `d_i`.
- **Commit**: finalize accepted tokens into the target sequence and KV cache, truncate rejected draft tokens from the draft KV cache, run a draft catch-up forward when all drafts are accepted, and ensure target and draft caches end the step with identical contents and seq_len.
- **Rollback / truncate**: discard KV blocks beyond the committed length.

## Phase 0: Offline N-gram Prototype

### Purpose

Validate the speculative-decoding acceptance loop and off-by-one alignment without touching the engine.

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
  - `baseline_decode_tps`, `target_forwards`
  - No `projected_tps` or `effective_tps` in P0.

### Success criteria

- `bit_exact == true` for `temperature=0`.
- Tool runs end-to-end without engine modifications.
- Acceptance rate is measured and reported honestly (may be low for non-repetitive prompts).

## Phase 1: Offline E2B/E4B Draft Prototype

### Purpose

Confirm that a small Gemma draft model is tokenizer-compatible, fits in memory alongside the target, and produces a measurable acceptance distribution on a fixed prompt set.

### Draft model candidates

1. **Primary**: `gemma-4-E2B-it-AWQ-INT4` (smaller, faster draft; lower acceptance but potentially better net speedup).
2. **Fallback**: `gemma-4-E4B-it` (larger, higher acceptance; draft latency may eat gains).

### P1.0 Tokenizer gate

Tokenizer compatibility is a hard gate. Use `tests/tools/gemma4_speculative_tokenizer_gate.py` and require **all** of:

- `vocab_size` identical.
- Full `token → ID` mapping identical (stable hash or exhaustive comparison), not just size.
- `added_tokens` and special token IDs (BOS/EOS/PAD) identical in semantics.
- Fixed prefix token IDs can be consumed directly by both target and draft models.

The following are **diagnostic only**; mismatches are reported but do not fail the gate:

- Normalizer / pre-tokenizer text behavior.
- Chat-template text rendering differences that do not affect token IDs.

If any hard check fails, stop. Do not proceed to P1.5 or P2.

### P1.1 Memory feasibility gate

Both models must load simultaneously and complete warmup without OOM.

- Record peak GPU memory after loading target + draft.
- Record each model's KV budget and block count.
- Gate: target + draft + both KV pools must fit below GPU cap with at least 5% headroom.
- If the draft cannot be kept resident, it is disqualified.

### P1.2 Draft prompt alignment

The draft model must consume the same prefix token IDs as the target, not a decode/re-encode of text.

- Build the prefix with the shared tokenizer.
- Pass the token IDs directly to the draft model.
- Do **not** use `decode(..., skip_special_tokens=True)` and re-encode, because that drops chat-control tokens and boundary information.

### P1.3 Acceptance sweep

Run the same offline loop as P0, but the draft source becomes a small `LLM` instance:

1. Load target and draft as separate `LLM` instances.
2. For each prompt in the fixture set:
   - Greedy baseline with target.
   - Speculative decode with draft proposing K tokens per step.
3. Report per-prompt and aggregate metrics:
   - `acceptance_rate = accepted_draft_tokens / all_drafted_tokens`
   - `mean_accepted_length`
   - `target_forwards`
   - `draft_tokens_per_second`
   - `baseline_decode_tps` (decode region only, excluding prefill)

### Fixed prompt fixture and metrics

- Fixture: `tests/tools/fixtures/gemma4_speculative_prompts.json`
- Buckets: context 128 / 512 / 2048 tokens; short answer (≤32 tokens) and long answer (≤128 tokens); English, Chinese, and code/repetitive text.
- K values: 1, 2, 4, 8.
- Each (prompt, K) runs at least 5 times after warmup; report median and worst-case prompt.
- Baseline TPS measured only over the decode region, not the initial prefill.

### P1 success criteria

- Tokenizer gate passes.
- Memory gate passes.
- Acceptance rate is measurable and stable across the fixture set.
- No `effective_tps` gate here; that is P1.5.

## Phase 1.5: Cached Multi-token Verifier Microbenchmark

### Purpose

Predict whether a real engine speculative-decoding fast path would actually be faster than single-token decode. This is the only performance gate before P2.

### What P1.5 may touch

P1.5 needs a cached K+1 verifier, which the current runtime does not expose. The minimal allowed primitive is:

- A verifier-mode path in `Gemma4ForConditionalGeneration.forward` (or a dedicated helper) that returns K+1 logits for input `[last_emitted_token, d_1, ..., d_K]` while reusing cached prefix KV.
- Input metadata changes inside `vllm/model_executor/models/gemma4/` only: positions, slot mapping, query_start_loc, scratch shapes.
- No changes to scheduler, `LiteEngine`, `InputBatchBuilder` public interface, or OpenAI API.

If this primitive cannot be built without touching the scheduler/backend, then P1.5 is blocked and must be re-scoped.

### Initial prefill and first `y`

1. Prefill the same prompt through both target and draft models.
2. Both caches now have length `C`.
3. Discard the draft prefill output.
4. Take the target prefill greedy output as `y`, the first `last_emitted` token.
5. Both models start the first speculative step from identical cached prefix KV and the same uncached `y`.

### Microbenchmark setup

1. Load **both** target and draft models and keep them resident.
2. For each context bucket (128 / 512 / 2048 tokens) in the fixture set:
   - Prefill the context through both models and save a post-prefill snapshot of both `seq_len` and block tables.
   - Obtain `y` from the target prefill output.
   - For each K in {1, 2, 4, 8}:
     - Repeat ≥5 times:
       1. Restore both models to the post-prefill snapshot and reset `y` to the target prefill output.
       2. Run consecutive speculative steps until exactly `N` committed tokens have been generated (e.g. N = 32 or 128).
       3. Each step must perform the real operations P2 would perform:
          - draft generates K tokens,
          - target verifier runs on `[last_emitted, d_1, ..., d_K]`,
          - CPU acceptance logic,
          - commit/truncate both target and draft KV state,
          - draft catch-up forward when all K drafts are accepted,
          - bonus token becomes next `last_emitted`.
       4. Synchronize the GPU and record total wall time for the `N`-token run.
3. Compute predicted decode TPS from measured committed tokens:
   - `mean_committed_tokens_per_step` = `N` / number_of_steps_in_run.
   - `median_step_time_s` = total_wall_time / number_of_steps_in_run.
   - `predicted_decode_tps = mean_committed_tokens_per_step / median_step_time_s`.

### KV reset requirement

Between repetitions the benchmark must restore the same `seq_len` and block tables for both models to the post-prefill snapshot. The tool must report the reset method in its output JSON.

### Bit-exact gate

For every prompt, K, and context bucket:

- Compare the speculative run's committed token IDs with a target-only greedy baseline run on the same prompt.
- `bit_exact == true`.
- Any mismatch fails the gate, regardless of predicted TPS.

### Performance gate

For a fixed K, across all three context buckets (128 / 512 / 2048):

- Aggregate `median predicted_decode_tps >= aggregate median baseline_decode_tps * 1.2`.
- No context bucket may fall below its own baseline (or a documented fallback threshold).
- `predicted_decode_tps >= 13 tok/s` (additional lower bound).
- Report worst-case prompt so high-acceptance repetitive prompts cannot mask poor chat-prompt performance.

If either the bit-exact gate or the performance gate fails, **stop**. Do not proceed to P2 engine integration.

### CLI / tool

- Location: `tests/tools/gemma4_speculative_verifier_microbench.py`
- Outputs JSON:
  - `context_len`, `K`
  - `baseline_decode_tps`
  - `median_step_time_s`, `mean_committed_tokens_per_step`
  - `predicted_decode_tps`, `speedup`, `passed_gate`
  - `bit_exact`
  - per-prompt breakdown including worst case

## Phase 2: Engine Integration

Only enter P2 if P1.5 predicts ≥20% decode speedup across all context buckets and is bit-exact.

### Scope

Integrate the winning draft source into `LiteEngine` / `AsyncLLM` for BS=1 plain-greedy decode.

- Add speculative configuration to `VllmConfig` (reusing existing upstream-like `speculative_config` semantics if present; otherwise extend cleanly).
- Add a speculative fast path inside the engine step for single running requests.
- Keep a separate draft KV cache; implement target KV-cache block truncation and draft KV-cache truncation.
- Preserve `AsyncLLM` streaming semantics.

### Eligibility

P2 first cut only enables speculative decoding when **all** are true:

- Model is Gemma4.
- Batch size is exactly 1.
- Text-only, no LoRA, no multimodal.
- `temperature == 0`.
- No repetition/frequency/presence penalty.
- `ignore_eos == false` and `min_tokens == 0`.
- No structured output or custom token bias.
- Otherwise fall back to existing single-token decode.

### Draft runtime ownership

The draft model is loaded and owned by the runtime initialization layer, not by `LiteEngine.__init__` or the backend directly.

- `LiteRuntimeAssembler` loads the draft model, allocates its KV cache, and injects an internal draft runner into the backend.
- The draft runner has the same lifecycle as the runtime: startup, normal operation, shutdown, abort.
- The backend receives a ready-to-use draft runner handle and only orchestrates draft/verify/commit within an engine step.

### Atomic engine step

A speculative decode step is a single engine step with three internal phases:

1. **Draft**: run the draft model to produce K candidate tokens.
2. **Verify**: run the target verifier on `[last_emitted_token, d_1, ..., d_K]`.
3. **Commit**: accept/reject drafts, truncate both KV caches, run draft catch-up if needed, publish the committed tokens once.

The draft runner is an internal helper, not a public scheduler request. It does not publish output or advance the target request until commit completes.

### Scheduler budget

- The scheduler/plan must reserve K+1 target query tokens for the verifier.
- If the remaining step token budget is insufficient, lower K or fall back to single-token decode.
- Draft K-token generation runs inside the same engine step but does not consume target token budget; its runtime cost is tracked separately for diagnostics.
- Prefill work in the same step must not silently exceed `step_token_budget`.

### Model logits contract

The current Gemma4 public forward returns only the last position logits. Verifier mode must return K+1 logits without changing the normal decode path.

- Add a metadata flag or a dedicated verifier path in `Gemma4ForConditionalGeneration.forward`.
- Normal decode still computes and returns only the last position logits.
- Verifier mode returns logits for positions `[last_emitted, d_1, ..., d_K]`.
- `InputBatchBuilder` must support per-request K+1 query tokens: positions, slot mapping, query_start_loc, and scratch shapes currently fixed at `(batch_size, 1)` need to become variable-length.

### Verify alignment (off-by-one)

Let `C` be the number of cached tokens and `y` be the last emitted token that is **not yet cached**. At the start of a speculative step both target and draft caches have length `C` and the next input position is `C`.

The verifier input is:

```
[y, d_1, d_2, ..., d_K]
```

with positions `C, C+1, ..., C+K`.

- `logits[:, 0]` validates `d_1` (it is the logit produced before seeing `d_1`).
- `logits[:, i-1]` validates `d_i`.
- `logits[:, K]` is the bonus token after all K drafts are accepted.
- If `argmax(logits[:, i-1]) == d_i`, accept `d_i`; otherwise reject and use `argmax(logits[:, i-1])` as the recovered next token.

Do **not** input the full cached prefix into the verifier; that would duplicate KV writes and use wrong positions.

### Dual-cache commit protocol

**Invariant at the end of every speculative step:** target and draft KV caches represent the same cached token prefix and have identical `seq_len`; the only uncached token is the next `last_emitted`.

Before verify, both caches have length `C` and the uncached token is `y`.

During the step:

- The draft model generates `d_1..d_K`. After generation the draft KV cache contains `prefix, y, d_1, ..., d_{K-1}` and has length `C + K`.
- The target verifier processes `[y, d_1, ..., d_K]` and writes `y, d_1, ..., d_K` to the target KV cache, bringing target KV length to `C + K + 1`. The bonus token `b` is sampled but not yet cached.

After verify, one of four states applies:

| State | Condition | Target seq_len / KV | Draft seq_len / KV | Next input |
|---|---|---|---|---|
| First-token reject | `d_1` mismatches | `C + 1`: `prefix, y` | `C + 1`: `prefix, y` | `recovered` |
| Partial reject at `d_i` | `d_i` mismatches, `i > 1` | `C + i`: `prefix, y, d_1, ..., d_{i-1}` | `C + i`: same token prefix | `recovered` |
| All accepted, before catch-up | all `d_1..d_K` match | `C + K + 1`: `prefix, y, d_1, ..., d_K` | `C + K`: `prefix, y, d_1, ..., d_{K-1}` | `d_K` (catch-up) |
| All accepted, after catch-up | catch-up forward wrote `d_K` | `C + K + 1`: `prefix, y, d_1, ..., d_K` | `C + K + 1`: same token prefix | `bonus` |

Notes:

- `recovered` is the target argmax at the rejection position; it becomes the next `last_emitted` but is not yet cached.
- `bonus` is the target argmax at position `C + K`; it becomes the next `last_emitted` but is not yet cached.
- Target truncation: keep blocks up to `ceil(target_seq_len / block_size)`; free blocks beyond.
- Draft truncation: keep blocks up to `ceil(draft_seq_len / block_size)`; free blocks beyond.
- The last retained block in either cache may contain stale tokens after the authoritative `seq_len`; they are invisible because `seq_len` is authoritative.
- Implement `truncate_request_blocks(request_id, num_tokens)` in the block manager; do not introduce a separate "tentative block" abstraction.

### EOS handling

When any accepted, recovered, or bonus token is an effective EOS token, the request completes immediately.

- Publish the output tokens up to and including the EOS.
- Free both target and draft request caches.
- Do not attempt to maintain a separate "EOS truncate" state; there is no next speculative step.

P2 eligibility restricts speculative decoding to `ignore_eos == false` and `min_tokens == 0`. Other configurations fall back to normal single-token decode.

### Batching behavior

- Speculative decoding is only enabled for exactly one running decode request.
- Batch > 1 falls back to normal single-token decode.

### API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="models/gemma-4-26B-A4B-it-AWQ-4bit",
    speculative_config={
        "method": "draft_model",
        "num_draft_tokens": 5,
        "draft_model": "models/gemma-4-E2B-it-AWQ-INT4",
    },
)

outputs = llm.generate("...", SamplingParams(temperature=0))
```

P0/P1/P1.5 use CLI arguments only; no public API changes until P2.

## Phase 3: Broader Sampling and API

Only after P2 proves `median decode TPS >= same-run baseline * 1.20` (and >= 13 tok/s):

- Support non-greedy sampling with distribution correction.
- Support repetition/frequency/presence penalty.
- Support OpenAI API server.
- Explore batched speculative decoding.

## Testing plan

| Phase | Test | Pass Criteria |
|---|---|---|
| P0 | `tests/tools/gemma4_speculative_prototype.py` | `bit_exact == true` for `temperature=0`; acceptance rate reported. |
| P0 | Run on repetitive and non-repetitive prompts | Tool does not crash; no `projected_tps` decision. |
| P1 | `tests/tools/gemma4_speculative_tokenizer_gate.py` | Hard gate passes for target/draft pair. |
| P1 | `tests/tools/gemma4_speculative_prototype.py --draft-model ...` | Memory gate passes; acceptance sweep completes on 128/512/2048 contexts. |
| P1.5 | `tests/tools/gemma4_speculative_verifier_microbench.py` | `bit_exact == true` for every prompt/K/context; aggregate `predicted_decode_tps >= baseline_decode_tps * 1.2` and >= 13 tok/s across 128/512/2048 contexts; no bucket below baseline. |
| P2 | Greedy bit-exact vs. baseline | Identical token sequence. |
| P2 | `tests/e2e_full_benchmark.py` | 26B median decode TPS >= same-run baseline * 1.20, and >= 13 tok/s. |
| P2 | KV truncate stress test | Long prompt + forced mismatches: no OOM, no crash, seq_len authoritative. |
| P2 | Dual-cache commit state test | All four commit states (first-token reject, partial accept, all accepted+catch-up, EOS) verified. |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| N-gram acceptance is near zero on chat prompts | Expected; P0 is a measurement/baseline phase, not a performance target. |
| E2B draft too slow to justify engine work | P1.5 gate measures multi-step run with both models resident; if not met, stop. |
| Tokenizer mismatch between draft and target | P1 hard gate: full token→ID mapping, added tokens, special IDs; stop if mismatch. |
| Off-by-one in verify logic | P0/P1/P1.5 bit-exact checks catch alignment bugs. |
| Verifier microbench does not predict engine behavior | P1.5 uses cached KV, K+1 tokens, real commit/truncate, and draft catch-up. |
| Engine integration breaks streaming / batching | P2 limited to BS=1 plain-greedy; batch path unchanged. |
| Two models do not fit in GPU memory | P1.1 memory gate; cannot be skipped. |
| Draft KV divergence after reject | Dual-cache commit protocol truncates draft KV to accepted prefix. |
| All-accepted draft catch-up cost hidden | Explicit catch-up forward included in P1.5 and P2 step cost. |
| State-machine bug in commit | P1.5 bit-exact gate against target baseline. |

## Success criteria

- P0: offline n-gram prototype merged to `tests/tools/`, bit-exact, acceptance rate measurable.
- P1: tokenizer hard gate + memory gate pass; acceptance sweep on fixed fixture across 128/512/2048 contexts.
- P1.5: cached verifier microbenchmark with both models resident and real state mutation is bit-exact and predicts ≥20% decode speedup (and ≥13 tok/s) across all context buckets, or we stop and document why.
- P2: only if P1.5 passes; engine integration with bit-exact greedy correctness, dual-cache commit protocol, and `median decode TPS >= baseline * 1.20` (and >= 13 tok/s).
- P3: only after P2 meets its gate.
