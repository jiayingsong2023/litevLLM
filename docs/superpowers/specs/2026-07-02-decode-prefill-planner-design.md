# DecodePrefillPlanner Design

**Date:** 2026-07-02  
**Author:** Kimi Code CLI  
**Status:** Design approved

## Summary

`StepScheduler` currently mixes two responsibilities:

1. **Step-level policy:** decide whether to run prefills, decodes, or both, and compute the token budget.
2. **Plan assembly:** turn a list of candidate prefill/decode requests into a `PrefillPlan` / `DecodePlan`, applying round-robin rotation, multimodal limits, LoRA limits, service-class weighted selection, and fast-path flags.

The admission and budget computations have already been extracted into `AdmissionPlanner` and `BudgetComputer`. This design extracts the remaining plan-assembly responsibility into a new `DecodePrefillPlanner` under `vllm/engine/planners/`.

## Goals

- Make `StepScheduler` responsible only for step-level orchestration and `StepPlanMetrics` assembly.
- Isolate prefill/decode cursor state and request-selection logic in one focused component.
- Preserve exact behavior: all existing `StepScheduler` tests must pass without modification.
- Keep the change localized to the scheduling path; no engine, executor, or model-layer changes.

## Non-Goals

- Changing admission logic (already in `AdmissionPlanner`).
- Changing budget computation (already in `BudgetComputer`).
- Refactoring `scheduling_constraints.py` internals. The constraints keep their current `step_scheduler` interface; `DecodePrefillPlanner` will expose the attributes they expect.
- Changing KV cache layout or the deferred KV flat-pool work.

## Proposed Architecture

```text
StepScheduler.build_plan()
  ├─ AdmissionPlanner.plan()        -> AdmissionPlan
  ├─ BudgetComputer.compute()       -> BudgetResult
  ├─ DecodePrefillPlanner.build_prefill_plan() -> PrefillPlanResult
  ├─ DecodePrefillPlanner.build_decode_plan()  -> DecodePlanResult
  └─ assemble StepPlanMetrics and StepPlan
```

### Component 1: `DecodePrefillPlanner`

Location: `vllm/engine/planners/decode_prefill_planner.py`

Responsibility: build `PrefillPlan` / `DecodePlan` from candidate request lists, applying rotation and constraints.

#### State owned by the planner

The following cursors move from `StepScheduler` to `DecodePrefillPlanner`:

- `_prefill_rr_cursor`
- `_decode_rr_cursor`
- `_prefill_lora_cursor`
- `_decode_lora_cursor`
- `_prefill_multimodal_cursor`
- `_decode_multimodal_cursor`
- `_prefill_multimodal_lora_cursor`
- `_decode_multimodal_lora_cursor`
- `_decode_service_cursor`

The following configuration values are copied at construction time so that `scheduling_constraints.py` can read them as `step_scheduler.<name>`:

- `service_class_weights`
- `decode_service_class_quotas`
- `max_prefill_lora_adapters_per_batch`
- `max_decode_lora_adapters_per_batch`
- `max_prefill_multimodal_requests_per_batch`
- `max_decode_multimodal_requests_per_batch`
- `max_prefill_multimodal_lora_requests_per_batch`
- `max_decode_multimodal_lora_requests_per_batch`
- `lora_fairness_relax_threshold`
- `lora_locality_tighten_threshold`
- `lora_limit_relax_delta`
- `lora_limit_tighten_delta`
- `multimodal_prefix_cache_relax_threshold`
- `multimodal_prefix_cache_tighten_threshold`
- `multimodal_prefill_limit_relax_delta`
- `multimodal_prefill_limit_tighten_delta`
- `multimodal_lora_prefill_limit_relax_delta`
- `multimodal_lora_prefill_limit_tighten_delta`
- `multimodal_lora_fairness_relax_threshold`
- `multimodal_lora_locality_tighten_threshold`
- `_multimodal_prefix_cache_hit_rate_feedback` (updated each step)

Other runtime parameters needed for plan assembly:

- `prefill_chunk_size`
- `prefill_microbatch_size`

#### Public methods

```python
class DecodePrefillPlanner:
    def __init__(
        self,
        *,
        service_class_weights: dict[str, int],
        decode_service_class_quotas: dict[str, int],
        # lora/multimodal limits and thresholds copied from StepScheduler
        prefill_chunk_size: int,
        prefill_microbatch_size: int,
    ) -> None: ...

    def update_runtime_feedback(self, feedback: dict[str, Any]) -> None:
        """Propagate observer feedback such as prefix-cache hit rate."""

    def build_prefill_plan(
        self,
        scheduler,
        prefills: list[str],
        token_budget: int,
    ) -> PrefillPlanResult: ...

    def build_decode_plan(
        self,
        scheduler,
        decodes: list[str],
        decode_limit: int,
        no_prefills_selected: bool,
    ) -> DecodePlanResult: ...
```

### Component 2: Result dataclasses

Location: `vllm/engine/planners/types.py`

```python
@dataclass(frozen=True)
class PrefillPlanResult:
    plan: PrefillPlan | None
    fairness_gap: dict[str, float]
    effective_multimodal_request_limit: int
    effective_lora_adapter_limit: int
    effective_multimodal_lora_request_limit: int
    multimodal_limit_triggered: bool
    multimodal_lora_limit_triggered: bool
    multimodal_limit_relaxed: bool
    multimodal_limit_tightened: bool
    multimodal_lora_limit_relaxed: bool
    multimodal_lora_limit_tightened: bool
    multimodal_lora_relaxed_by_fairness: bool
    multimodal_lora_tightened_by_locality: bool
    multimodal_lora_max_fairness_gap: float
    lora_limit_relaxed: bool
    lora_limit_tightened: bool


@dataclass(frozen=True)
class DecodePlanResult:
    plan: DecodePlan | None
    fairness_gap: dict[str, float]
    effective_lora_adapter_limit: int
    effective_multimodal_lora_request_limit: int
    multimodal_lora_limit_triggered: bool
    multimodal_lora_limit_relaxed: bool
    multimodal_lora_limit_tightened: bool
    multimodal_lora_relaxed_by_fairness: bool
    multimodal_lora_tightened_by_locality: bool
    multimodal_lora_max_fairness_gap: float
    lora_limit_relaxed: bool
    lora_limit_tightened: bool
```

These replace the large tuple returns of `_build_prefill_plan` and `_build_decode_plan`.

### Component 3: `StepScheduler` changes

- Remove `_build_prefill_plan` and `_build_decode_plan`.
- Remove all prefill/decode cursor attributes listed above.
- Remove helper methods that are only used by plan assembly:
  - `_shape_multimodal_prefill_batch`
  - `_effective_prefill_multimodal_lora_limit`
  - `_effective_decode_multimodal_lora_limit`
  - `_select_weighted_requests`
  - `_apply_lora_adapter_batch_limit`
  - `_apply_multimodal_request_limit`
  - `_apply_multimodal_lora_request_limit`
- Keep counting helpers that `StepPlanMetrics` still needs:
  - `_count_service_classes`
  - `_count_lora_adapters`
  - `_count_multimodal_requests`
  - `_count_multimodal_lora_requests`
  - `_max_abs_share_gap`
- Instantiate `_decode_prefill_planner` in `__init__`.
- In `build_plan`, call planner methods and unpack results into `StepPlanMetrics`.
- In `update_runtime_feedback`, propagate feedback to the planner.

### Constraint-interface compatibility

`scheduling_constraints.py` reads attributes from the object passed as `step_scheduler`. To avoid changing the constraints, `DecodePrefillPlanner` exposes the same attribute names that the constraints expect, including cursor names and configuration values. This makes the planner a drop-in replacement for the `step_scheduler` argument within the plan-assembly scope.

## Data Flow

```python
def build_plan(self, scheduler):
    admissions = self._admission_planner.plan(scheduler)
    budget = self._budget_computer.compute(...)
    prefills, decodes = scheduler.classify_requests()
    starvation_protected = self._should_protect_prefills(...)
    prefill_result = self._decode_prefill_planner.build_prefill_plan(
        scheduler, prefills, budget.prefill_budget
    )
    decode_result = self._decode_prefill_planner.build_decode_plan(
        scheduler,
        decodes,
        budget.decode_budget,
        prefill_result.plan is None,
    )
    # ... assemble StepPlanMetrics from admissions, prefill_result, decode_result ...
    return StepPlan(...)
```

## Testing Plan

1. **Behavioral parity:** run `tests/test_step_scheduler.py` before and after the refactor. No test should change.
2. **New unit tests:** create `tests/test_decode_prefill_planner.py` covering:
   - Basic prefill plan assembly and chunk calculation.
   - Decode plan assembly and `use_fast_path` flag.
   - Round-robin cursor advancement across multiple calls.
   - Multimodal prefill limit tightening/relaxing.
   - LoRA adapter limits and fairness gaps.
   - Service-class weighted decode selection.
3. **Regression gates:**
   - `bash tests/run_regression_suite.sh`
   - `SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh`
   - `uv run python tests/e2e_full_benchmark.py`

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Constraint code relies on `step_scheduler` attributes not fully mirrored. | After implementation, run the full step-scheduler test suite; any missing attribute will surface as an `AttributeError`. |
| Cursor state migration changes ordering in subtle ways. | Parity tests against existing `test_step_scheduler.py` and correctness regression catch behavioral drift. |
| `StepPlanMetrics` field mapping becomes error-prone. | Use dataclass result objects with named fields; avoid positional tuple unpacking in `build_plan`. |

## Rollout

1. Implement `DecodePrefillPlanner` and result dataclasses.
2. Replace `_build_prefill_plan` / `_build_decode_plan` calls in `StepScheduler` with planner calls.
3. Remove migrated methods and cursor state from `StepScheduler`.
4. Add unit tests and run regression/correctness/e2e gates.
