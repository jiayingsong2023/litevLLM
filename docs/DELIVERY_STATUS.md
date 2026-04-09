# FastInference Delivery Status (P0/P1/P2)

Last updated: 2026-04-09

This document is the external-facing delivery snapshot for the lite runtime track.

## Scope Statement

Within the current Lite runtime scope, **P0, P1, and P2 are delivered**.  
The explicitly deferred item is **speculative decoding**.

## DONE

### P0: Control-plane convergence and scheduler maturity

- Unified control-plane around `RuntimeController` and `StepScheduler`/`RequestScheduler`.
- Parameterized admission policy with:
  - `max_admit_per_step`
  - queue timeout/cancel handling
  - fairness-oriented aging and starvation protection
  - service-class policy knobs (weights/quotas)
- Preemption path integrated into runtime control hooks:
  - `maybe_preempt(step_plan, scheduler)`
  - soft prefill preemption baseline behavior
- Runtime stats export path available through:
  - engine/controller/backend `stats()`
  - debug stats summary endpoint and benchmark snapshots/diffs

### P1: Structured outputs + cache/policy observability

- Structured outputs unified on grammar-backed backend for:
  - `choice`
  - `json`
  - `json_object`
  - `json_schema`
  - `regex`
  - `grammar`
- OpenAI compatibility mapping and behavior coverage expanded, including streaming/compatibility paths and tokenizer fallback behavior.
- Prefix cache minimum runtime implementation plus observability:
  - hit/miss
  - exact/partial hit
  - saved prefill tokens
- Benchmark pipeline supports phase snapshot + phase diff metrics for:
  - prefix cache
  - preemption
  - fairness
  - high-signal terminal summary lines

### P2: LoRA runtime and multimodal serving

- LoRA runtime path delivered with:
  - adapter registry
  - request routing
  - batch-level mixed adapter execution contract
  - per-adapter scheduling observability and fairness/locality metrics
  - adapter-aware policy and adaptive feedback counters
- Multimodal serving delivered with:
  - OpenAI `image_url` single-image path
  - multimodal embeddings integrated into prefill/model input semantics
  - multi-image and streaming behavior coverage
  - multimodal-aware scheduling and preemption
- Combined paths delivered and measured:
  - multimodal + LoRA
  - multimodal + structured outputs
  - multimodal + prefix cache
  - benchmark/runtime derived metrics and summaries for combined policy effects

## KNOWN_GAPS

### Explicitly deferred

- **Speculative decoding** is not part of the delivered P0/P1/P2 scope.

### Remaining tail work (non-blocking to P0/P1/P2 completion)

- Broader production hardening under real traffic distributions:
  - longer-run stability/soak validation
  - additional A/B benchmark coverage at larger scale
- Policy tuning and calibration:
  - threshold/default tuning by workload profile
  - stricter SLO envelopes for fairness under mixed multimodal+LoRA pressure
- Documentation deepening:
  - operator playbooks and rollout runbooks per deployment environment

## Validation Basis

Delivery claims are based on merged runtime/tests and targeted regression suites in this repository, including scheduler/runtime observer/export/benchmark suites and structured outputs + multimodal coverage.

