# SamplingDriver Batch Vectorization Design

**Date:** 2026-07-01  
**Author:** Kimi Code CLI  
**Status:** Design review

## Summary

`SamplingDriver.sample_batch_tokens` currently performs repetition / frequency / presence penalties, EOS masking, anti-template masking and context biasing in a per-request Python loop. Microbenchmarks on a ROCm device show that this makes the penalty + multinomial path **10–20x slower** than the greedy path, and the gap grows with batch size. The top-k / top-p / temperature stages are already vectorized; the next optimization target is the **penalty and mask encoding stage**.

This design proposes splitting `sample_batch_tokens` into two focused components:

1. `PenaltyEncoder` – batch-level PyTorch penalty / bias / mask application, with a per-row fallback for structured-output constraints and non-uniform context biases.
2. `Sampler` – existing vectorized temperature / top-k / top-p / multinomial logic, extracted into a reusable unit.

`SamplingDriver` becomes a thin orchestrator: `PenaltyEncoder.encode(...)` → `Sampler.sample(...)`.

## Goals

- Reduce decode-step sampling latency by vectorizing penalty and mask computation.
- Preserve bit-exact behavior for existing sampling parameters and constraints.
- Keep the change localized to the sampling path; no engine or scheduler changes required.
- Leave room for a future fused Triton sampler without another large refactor.

## Non-Goals

- Fusing the entire sampling pipeline into a single Triton kernel in this iteration.
- Vectorizing `torch.multinomial` across requests with different `torch.Generator` instances (measured as non-bottleneck).
- Changing tokenization, stop-string matching, or output post-processing.

## Background & Evidence

A temporary microbenchmark measured `SamplingDriver.sample_batch_tokens` on a Radeon 8060S Graphics device with `generated_len=50`:

| scenario | bs=1 | bs=8 | bs=16 |
|---|---|---|---|
| penalties + multinomial (vocab 32k) | 1.38 ms | 5.61 ms | 10.97 ms |
| greedy only (vocab 32k) | 0.10 ms | 0.48 ms | 0.89 ms |
| penalties + multinomial (vocab 151k) | 1.15 ms | 4.10 ms | 10.84 ms |
| greedy only (vocab 151k) | 0.05 ms | 0.29 ms | 0.56 ms |

Greedy sampling is already fast because it skips the penalty loops and uses `torch.argmax`. The penalty loop path spends most of its time constructing small per-request tensors and applying them one row at a time. Vectorizing these operations should bring the multinomial path much closer to greedy latency.

## Proposed Architecture

```text
SamplingDriver.sample_batch_tokens(logits_2d, requests)
  ├─ PenaltyEncoder.encode(logits_2d, requests)
  │    ├─ vectorized repetition / frequency / presence penalty
  │    ├─ vectorized EOS / anti-template mask
  │    └─ per-row fallback for structured constraints / special biases
  └─ Sampler.sample(encoded_logits, requests)
       ├─ temperature scaling (already vectorized)
       ├─ top-k / top-p filtering (already vectorized)
       └─ multinomial sampling per request
```

### Component 1: `PenaltyEncoder`

Responsibility: take raw logits `[B, V]` and return logits with all penalties, biases and masks applied.

Location: `vllm/engine/sampling/penalty_encoder.py` (new module `vllm.engine.sampling`).

#### Vectorized operations

All operations stay on the GPU. No per-request Python tensor construction.

1. **Repetition penalty**
   - Gather unique generated token ids per row.
   - Use `scatter` to apply `logit / rp` where `logit > 0` and `logit * rp` where `logit <= 0`.

2. **Frequency penalty**
   - Count occurrences of each token per row.
   - Use `scatter` to subtract `fp * count`.

3. **Presence penalty**
   - Same unique-id gather as repetition penalty, but subtract a constant `pp`.

4. **EOS / stop-token mask**
   - Most requests share the same EOS token set. Build a single mask and broadcast.
   - If requests differ, build a `[B, V]` boolean mask and set masked positions to `-inf`.
   - Honor `min_tokens`: only mask EOS while `len(generated_ids) < min_tokens` (or when `ignore_eos` is true, always mask).

5. **Anti-template mask**
   - For rows with `anti_template_token_ids` and `len(generated_ids) < 12`, scatter `-60.0` to those positions.

#### Fallback path

Requests that cannot be safely vectorized are handled individually and their rows are written back into the batch tensor:

- `structured_output_constraint is not None` (arbitrary per-row transformation).
- Context bias that is not the no-op default (rare in the lite mainline).

The fallback uses the existing per-row logic, ensuring behavior is unchanged.

### Component 2: `Sampler`

Responsibility: sample one token per request from already-encoded logits.

Location: `vllm/engine/sampling/sampler.py`.

This is essentially the existing vectorized temperature / top-k / top-p logic extracted from `SamplingDriver`, plus the final per-request `torch.multinomial` call. The per-request `multinomial` is retained because each request owns its own `torch.Generator`, and the benchmark shows it is not the bottleneck.

### Component 3: `SamplingDriver`

`SamplingDriver` keeps its public API (`sample_next_token`, `sample_batch_tokens`, `completion_eos_ids`) but internally delegates to `PenaltyEncoder` and `Sampler`.

The single-request path `sample_next_token` can call `PenaltyEncoder.encode_row(...)` (a thin wrapper around the same vectorized primitives with `B=1`) to avoid code duplication.

## Data Flow

```python
def sample_batch_tokens(self, logits_2d, requests):
    # 1. Normalize shape
    if logits_2d.ndim == 3:
        logits_2d = logits_2d.squeeze(1)
    elif logits_2d.ndim == 1:
        logits_2d = logits_2d.unsqueeze(0)

    # 2. Apply penalties and masks in batch
    encoded = self._penalty_encoder.encode(logits_2d, requests)

    # 3. Sample
    return self._sampler.sample(encoded, requests)
```

## Interface Sketch

```python
class PenaltyEncoder:
    def __init__(self, tokenizer, hf_config, policies): ...

    def encode(
        self,
        logits: torch.Tensor,        # [B, V]
        requests: list[RequestState],
    ) -> torch.Tensor:               # [B, V]
        ...

    def encode_row(
        self,
        logits: torch.Tensor,        # [V]
        request: RequestState,
    ) -> torch.Tensor:               # [V]
        ...


class Sampler:
    def sample(
        self,
        logits: torch.Tensor,        # [B, V]
        requests: list[RequestState],
    ) -> list[int]:
        ...
```

## Testing Plan

1. **Numerical parity tests**
   - Add `tests/test_sampling_driver_penalty_encoder.py`.
   - Compare `PenaltyEncoder.encode` against the existing per-row `_sample_token_from_logits` for random logits and random `generated_ids`.
   - Cover: repetition, frequency, presence, EOS masking with `min_tokens` and `ignore_eos`, anti-template masking, mixed params across a batch.

2. **Edge cases**
   - Empty `generated_ids`.
   - Token ids outside `[0, V)` (should be ignored).
   - Batches where some requests have structured constraints and others do not.

3. **Regression tests**
   - `tests/test_edge_cases.py`
   - `tests/run_regression_suite.sh`
   - `SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh`
   - `tests/e2e_full_benchmark.py` (to ensure no perf regression and ideally an improvement)

## Rollout & Risks

| Risk | Mitigation |
|---|---|
| Different EOS / stop-token sets per request make batch mask construction complex. | Fast path assumes a shared EOS set; divergent rows fall back to per-row processing. |
| Very long `generated_ids` increase scatter memory. | Only unique ids are gathered; if needed later, cap to the most recent N tokens. |
| Structured-output constraints are inherently per-row. | Keep the existing fallback path; vectorized path is used only when constraints are absent. |
| Numerical drift from vectorized `where` / `scatter`. | Parity tests against the old per-row implementation enforce identical results. |

## Recommended Rollout

1. Land `PenaltyEncoder` + `Sampler` behind `SamplingDriver`.
2. Keep the original per-row implementation as a private method for one release, guarded by an opt-out environment variable (default to vectorized).
3. After regression and e2e benchmarks pass, remove the legacy path.

## Future Work

- A fused Triton kernel for temperature + top-k + top-p + multinomial could further reduce launch overhead for large batch sizes.
- Move `OutputPipeline` token decoding / stop-string matching to batch operations if profiling shows it becomes the next bottleneck.
