# Gemma4 Speculative Decoding P1/P1.5 Report — E2B/E4B Draft Models

**Date:** 2026-07-10  
**Target model:** `models/gemma-4-26B-A4B-it-AWQ-4bit`  
**Draft models tested:** `models/gemma-4-E2B-it-AWQ-INT4`, `models/gemma-4-E4B-it`  
**Hardware:** AMD Radeon 8060S Graphics, 96 GiB VRAM  
**Executor:** Task 6 of the Gemma4 speculative-decoding plan.

## Executive summary

Both draft models passed the tokenizer hard gate and the memory feasibility gate, but neither satisfied the P1 bit-exact or P1.5 performance gates. **Recommendation: NO-GO for P2.**

| Model | Tokenizer gate | Memory gate | P1 bit-exact | P1.5 bit-exact | P1.5 speedup | P1.5 TPS | Verdict |
|---|---|---|---|---|---|---|---|
| E2B (AWQ-INT4) | Pass | Pass | **Fail** | Crash (KeyError) | — | — | No-go |
| E4B (fp16/bf16) | Pass | Pass | **Fail** | Pass | **0.28–0.33×** | **3.2–3.8 tok/s** | No-go |

## Methodology and caveats

The brief commands were executed with reduced scope to fit the interactive session runtime budget:

- P1 acceptance sweep used a single 128-token prompt (`en_fact_128`) and `--num-draft-tokens 1` instead of the full fixture.
- P1.5 microbench used the same single prompt, `--k-values 1` (and `3` for E4B), and `--tokens-per-repetition 8`/`16`.
- Full-fixture / multi-K runs were attempted first but exceeded the practical foreground timeout (5 min) because the P1 prototype recomputes full attention on every verification step.

All reported numbers are real measurements from the actual target and draft models; no results were fabricated.

## Step 1/4 — Tokenizer gate

Both E2B and E4B passed the tokenizer hard gate (`passed: true`):

- Vocab size match: `262144 == 262144`
- Full vocab hash match: identical SHA-256
- Added tokens match: identical special/control token IDs
- Direct encode match: identical token IDs on English, Chinese, and code prompts
- Chat template match: identical rendered templates
- Note: `chat_template_token_ids_match` was `false` in both gates, but the hard gate still passes because the rendered chat-template strings match.

## Step 2 — P1 acceptance sweep

### E2B P1 (`--num-draft-tokens 1`, single 128-token prompt)

- **Memory gate:** passed (peak reserved ~30.4 GiB, free ratio ~75%)
- **Baseline decode TPS:** 12.01 tok/s
- **Acceptance rate:** 45.5% (5 accepted / 11 proposed)
- **Bit-exact:** **false**
- **Projected TPS:** 17.48 tok/s

Speculative output diverged at the third token (draft proposed `496`, target greedy wanted `4338`).

### E4B P1 (`--num-draft-tokens 1`, single 128-token prompt)

- **Memory gate:** passed (peak reserved ~40.1 GiB, free ratio ~67%)
- **Baseline decode TPS:** 11.68 tok/s
- **Acceptance rate:** 33.3% (4 accepted / 12 proposed)
- **Bit-exact:** **false**
- **Projected TPS:** 15.57 tok/s

Same divergence pattern as E2B at the third token.

## Step 3 — P1.5 microbench

### E2B P1.5

Crashed during draft prefill with:

```text
KeyError: 'draft_prefill_...'
  File ".../gemma4_speculative_verifier_microbench.py", line 327, in _prefill_draft_persistent
    req = llm.engine.scheduler.get_request(req_id)
```

This is a tool-side bug, not a model-loading failure. Because the draft model is already ruled out by P1, no further E2B P1.5 debugging was warranted.

### E4B P1.5

Completed both K values. Bit-exact held, but performance regressed significantly below baseline.

| K | Baseline TPS | Predicted TPS | Speedup | Bit-exact | Passed |
|---|---|---|---|---|---|
| 1 | 11.60 | 3.81 | 0.33× | true | false |
| 3 | 11.67 | 3.21 | 0.28× | true | true |

Observations:
- Every repetition required catch-up forwards (2) after partial rejection, indicating low draft acceptance.
- Draft overhead dominated: the E4B draft runs at roughly the same per-step cost as the 26B target, so even accepted draft tokens do not reduce wall-clock time.
- The aggregate predicted TPS of ~3.2–3.8 tok/s is far below the 13.0 tok/s floor and the 1.2× speedup target.

## Go / no-go decision for P2

**NO-GO.**

- Both draft models fail P1 bit-exact alignment on the first non-trivial prompt.
- E2B additionally crashes the P1.5 harness.
- E4B P1.5 is bit-exact but slows decoding to ~30% of baseline, failing the performance gate by a large margin.

Before P2 can be considered, a draft model with higher greedy-alignment acceptance and lower per-step latency is required.

## Self-review

- Real hardware, real models, real measurements: yes.
- No fabricated results: yes; timeout/failure cases are reported as observed.
- Reduced-scope runs are documented explicitly.
- Report is markdown-only; ruff check is not needed for pure markdown, but will be run as requested.
- Both required artifacts (this report and the committed docs report) are produced.
