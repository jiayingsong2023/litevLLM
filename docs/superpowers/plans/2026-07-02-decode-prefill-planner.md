# DecodePrefillPlanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract prefill/decode plan assembly from `StepScheduler` into a focused `DecodePrefillPlanner` under `vllm/engine/planners/`.

**Architecture:** Add result dataclasses to `vllm/engine/planners/types.py`, implement `DecodePrefillPlanner` with its own cursor state and constraint-compatible configuration attributes, then thin `StepScheduler` to delegate plan assembly and keep only step-level orchestration and metrics assembly.

**Tech Stack:** Python 3.12, PyTorch, `uv`, `pytest`, `ruff`.

## Global Constraints

- Python 3.12 only; use `uv`.
- All tensor logic uses PyTorch; no NumPy in hot path.
- New files carry `SPDX-License-Identifier: Apache-2.0` header.
- Every new public function/class has type hints and a docstring.
- Do not change `scheduling_constraints.py`; `DecodePrefillPlanner` must expose the attributes it expects.
- Existing `tests/test_step_scheduler.py` must pass without modification.
- Production runtime must not read `FASTINFERENCE_*` env vars directly outside the config/registry bridge.

---

## File Structure

| File | Responsibility |
|---|---|
| `vllm/engine/planners/types.py` | `PrefillPlanResult` and `DecodePlanResult` dataclasses. |
| `vllm/engine/planners/decode_prefill_planner.py` | `DecodePrefillPlanner`: prefill/decode plan assembly, cursor state, constraint config mirror. |
| `vllm/engine/planners/__init__.py` | Re-export `DecodePrefillPlanner` (already re-exports `AdmissionPlanner`, `BudgetComputer`). |
| `vllm/engine/step_scheduler.py` | Thin orchestrator; removes plan-assembly helpers and cursors. |
| `tests/test_decode_prefill_planner.py` | Unit tests for planner cursor behavior and plan assembly. |

---

## Task 1: Add `PrefillPlanResult` and `DecodePlanResult` to `vllm/engine/planners/types.py`

**Files:**
- Modify: `vllm/engine/planners/types.py`
- Test: `tests/test_planner_types.py` (temporary, deleted after Task 4)

**Interfaces:**
- Consumes: existing `PrefillPlan` and `DecodePlan` from `vllm.engine.step_plan`.
- Produces: `PrefillPlanResult` and `DecodePlanResult` dataclasses used by `DecodePrefillPlanner` and `StepScheduler`.

- [ ] **Step 1: Write the failing import test**

Create `tests/test_planner_types.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from vllm.engine.planners.types import DecodePlanResult, PrefillPlanResult


def test_result_classes_exist() -> None:
    assert PrefillPlanResult is not None
    assert DecodePlanResult is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_planner_types.py::test_result_classes_exist -v
```

Expected: `ImportError: cannot import name 'PrefillPlanResult' from 'vllm.engine.planners.types'`

- [ ] **Step 3: Implement the dataclasses**

Edit `vllm/engine/planners/types.py` and add:

```python
from dataclasses import dataclass
from typing import Any

from vllm.engine.step_plan import DecodePlan, PrefillPlan


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

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_planner_types.py::test_result_classes_exist -v
```

Expected: `1 passed`

- [ ] **Step 5: Lint changed files**

```bash
uv run ruff check vllm/engine/planners/types.py tests/test_planner_types.py
```

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add vllm/engine/planners/types.py tests/test_planner_types.py
git commit -m "feat(planners): add PrefillPlanResult and DecodePlanResult dataclasses"
```

---

## Task 2: Implement `DecodePrefillPlanner.build_prefill_plan`

**Files:**
- Create: `vllm/engine/planners/decode_prefill_planner.py`
- Modify: `vllm/engine/planners/__init__.py`
- Test: `tests/test_decode_prefill_planner.py`

**Interfaces:**
- Consumes: `PrefillPlanResult` from Task 1; `PrefillPlan` from `vllm.engine.step_plan`; constraint helpers in `vllm.engine.scheduling_constraints` and `vllm.engine.scheduling_helpers`.
- Produces: `DecodePrefillPlanner.build_prefill_plan(scheduler, prefills, token_budget) -> PrefillPlanResult`.

- [ ] **Step 1: Write the failing prefill test**

Create `tests/test_decode_prefill_planner.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from vllm.engine.planners.decode_prefill_planner import DecodePrefillPlanner
from vllm.engine.request_scheduler import RequestScheduler
from vllm.engine.request_state import RequestState
from vllm.sampling_params import SamplingParams


def _make_scheduler_with_prefills(n: int):
    scheduler = RequestScheduler(max_active_requests=n)
    for i in range(n):
        req = RequestState(
            request_id=f"r{i}",
            prompt="",
            input_ids=list(range(16)),
            sampling_params=SamplingParams(),
            slot_idx=i,
            is_prefill=True,
            seq_len=0,
        )
        scheduler.add_request(f"r{i}", req)
    return scheduler


def test_build_prefill_plan_selects_first_chunk() -> None:
    planner = DecodePrefillPlanner(
        service_class_weights={"latency": 4},
        decode_service_class_quotas={},
        max_prefill_lora_adapters_per_batch=0,
        max_decode_lora_adapters_per_batch=0,
        max_prefill_multimodal_requests_per_batch=0,
        max_decode_multimodal_requests_per_batch=0,
        max_prefill_multimodal_lora_requests_per_batch=0,
        max_decode_multimodal_lora_requests_per_batch=0,
        lora_fairness_relax_threshold=0.0,
        lora_locality_tighten_threshold=0.0,
        lora_limit_relax_delta=1,
        lora_limit_tighten_delta=1,
        multimodal_prefix_cache_relax_threshold=0.2,
        multimodal_prefix_cache_tighten_threshold=0.8,
        multimodal_prefill_limit_relax_delta=1,
        multimodal_prefill_limit_tighten_delta=1,
        multimodal_lora_prefill_limit_relax_delta=1,
        multimodal_lora_prefill_limit_tighten_delta=1,
        multimodal_lora_fairness_relax_threshold=0.0,
        multimodal_lora_locality_tighten_threshold=0.0,
        prefill_chunk_size=4,
        prefill_microbatch_size=2,
    )
    scheduler = _make_scheduler_with_prefills(3)
    prefills, _ = scheduler.classify_requests()
    result = planner.build_prefill_plan(scheduler, prefills, token_budget=10)
    assert result.plan is not None
    assert result.plan.request_ids == ["r0", "r1"]
    assert result.plan.chunk_len == 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_decode_prefill_planner.py::test_build_prefill_plan_selects_first_chunk -v
```

Expected: `ModuleNotFoundError: No module named 'vllm.engine.planners.decode_prefill_planner'`

- [ ] **Step 3: Implement `DecodePrefillPlanner` prefill path**

Create `vllm/engine/planners/decode_prefill_planner.py` with the constructor and `build_prefill_plan`. Copy the body from `StepScheduler._build_prefill_plan` verbatim, but:

- Return `PrefillPlanResult(...)` instead of the raw tuple.
- Replace `self._prefill_rr_cursor` with `self._prefill_rr_cursor` (same name, now owned by the planner).
- Keep `_shape_multimodal_prefill_batch` as a private method on the planner.
- Keep `_shape_lora_batch` and `_apply_multimodal_lora_request_limit` calls; these helper methods will be moved in Task 3, or you can move them now if needed for the prefill path. For this task, move only what the prefill path needs:
  - `_shape_lora_batch`
  - `_apply_multimodal_lora_request_limit`
  - `_count_lora_adapters` (static)

Constructor mirrors all configuration attributes that `scheduling_constraints.py` reads. Example skeleton:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from vllm.engine.planners.types import PrefillPlanResult
from vllm.engine.scheduling_constraints import (
    LoRAConstraint,
    MultiModalConstraint,
    MultiModalLoRAConstraint,
    ServiceClassSelector,
)
from vllm.engine.scheduling_helpers import lora_adapter_key, rotate_candidates
from vllm.engine.step_plan import PrefillPlan


class DecodePrefillPlanner:
    """Build PrefillPlan and DecodePlan for a single engine step."""

    def __init__(
        self,
        *,
        service_class_weights: dict[str, int],
        decode_service_class_quotas: dict[str, int],
        max_prefill_lora_adapters_per_batch: int,
        max_decode_lora_adapters_per_batch: int,
        max_prefill_multimodal_requests_per_batch: int,
        max_decode_multimodal_requests_per_batch: int,
        max_prefill_multimodal_lora_requests_per_batch: int,
        max_decode_multimodal_lora_requests_per_batch: int,
        lora_fairness_relax_threshold: float,
        lora_locality_tighten_threshold: float,
        lora_limit_relax_delta: int,
        lora_limit_tighten_delta: int,
        multimodal_prefix_cache_relax_threshold: float,
        multimodal_prefix_cache_tighten_threshold: float,
        multimodal_prefill_limit_relax_delta: int,
        multimodal_prefill_limit_tighten_delta: int,
        multimodal_lora_prefill_limit_relax_delta: int,
        multimodal_lora_prefill_limit_tighten_delta: int,
        multimodal_lora_fairness_relax_threshold: float,
        multimodal_lora_locality_tighten_threshold: float,
        prefill_chunk_size: int,
        prefill_microbatch_size: int,
    ) -> None:
        self.service_class_weights = service_class_weights
        self.decode_service_class_quotas = decode_service_class_quotas
        self.max_prefill_lora_adapters_per_batch = max_prefill_lora_adapters_per_batch
        self.max_decode_lora_adapters_per_batch = max_decode_lora_adapters_per_batch
        self.max_prefill_multimodal_requests_per_batch = max_prefill_multimodal_requests_per_batch
        self.max_decode_multimodal_requests_per_batch = max_decode_multimodal_requests_per_batch
        self.max_prefill_multimodal_lora_requests_per_batch = max_prefill_multimodal_lora_requests_per_batch
        self.max_decode_multimodal_lora_requests_per_batch = max_decode_multimodal_lora_requests_per_batch
        self.lora_fairness_relax_threshold = lora_fairness_relax_threshold
        self.lora_locality_tighten_threshold = lora_locality_tighten_threshold
        self.lora_limit_relax_delta = lora_limit_relax_delta
        self.lora_limit_tighten_delta = lora_limit_tighten_delta
        self.multimodal_prefix_cache_relax_threshold = multimodal_prefix_cache_relax_threshold
        self.multimodal_prefix_cache_tighten_threshold = multimodal_prefix_cache_tighten_threshold
        self.multimodal_prefill_limit_relax_delta = multimodal_prefill_limit_relax_delta
        self.multimodal_prefill_limit_tighten_delta = multimodal_prefill_limit_tighten_delta
        self.multimodal_lora_prefill_limit_relax_delta = multimodal_lora_prefill_limit_relax_delta
        self.multimodal_lora_prefill_limit_tighten_delta = multimodal_lora_prefill_limit_tighten_delta
        self.multimodal_lora_fairness_relax_threshold = multimodal_lora_fairness_relax_threshold
        self.multimodal_lora_locality_tighten_threshold = multimodal_lora_locality_tighten_threshold
        self.prefill_chunk_size = prefill_chunk_size
        self.prefill_microbatch_size = prefill_microbatch_size

        self._multimodal_prefix_cache_hit_rate_feedback = 0.0
        self._prefill_rr_cursor = 0
        self._prefill_lora_cursor = 0
        self._prefill_multimodal_cursor = 0
        self._prefill_multimodal_lora_cursor = 0
        self._decode_rr_cursor = 0
        self._decode_service_cursor = 0
        self._decode_lora_cursor = 0
        self._decode_multimodal_cursor = 0
        self._decode_multimodal_lora_cursor = 0

        self._service_class_selector = ServiceClassSelector()
        self._lora_constraint = LoRAConstraint()
        self._multimodal_constraint = MultiModalConstraint()
        self._multimodal_lora_constraint = MultiModalLoRAConstraint()

    def update_runtime_feedback(self, feedback: dict[str, Any]) -> None:
        self._multimodal_prefix_cache_hit_rate_feedback = float(
            feedback.get("observer", {})
            .get("multimodal", {})
            .get("prefix_cache_hit_rate", 0.0)
        )

    @staticmethod
    def _count_lora_adapters(scheduler, request_ids: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rid in request_ids:
            key = lora_adapter_key(scheduler.get_request(rid))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _shape_lora_batch(self, *, scheduler, request_ids, baseline_counts, max_lora_adapters, cursor_attr):
        return self._lora_constraint.apply_lora_adapter_batch_limit(
            scheduler=scheduler,
            step_scheduler=self,
            request_ids=request_ids,
            baseline_counts=baseline_counts,
            max_lora_adapters=max_lora_adapters,
            cursor_attr=cursor_attr,
        )

    def _apply_multimodal_lora_request_limit(
        self, *, scheduler, request_ids, max_multimodal_lora_requests, cursor_attr, candidate_request_ids
    ):
        return self._multimodal_lora_constraint.apply_multimodal_lora_request_limit(
            scheduler=scheduler,
            step_scheduler=self,
            request_ids=request_ids,
            max_multimodal_lora_requests=max_multimodal_lora_requests,
            cursor_attr=cursor_attr,
            candidate_request_ids=candidate_request_ids,
        )

    def _shape_multimodal_prefill_batch(self, *, scheduler, request_ids, candidate_prefills):
        return self._multimodal_constraint.shape_multimodal_prefill_batch(
            scheduler=scheduler,
            step_scheduler=self,
            request_ids=request_ids,
            candidate_prefills=candidate_prefills,
        )

    @staticmethod
    def _rotate_candidates(request_ids: list[str], cursor: int) -> list[str]:
        return rotate_candidates(request_ids, cursor)

    def build_prefill_plan(self, scheduler, prefills: list[str], token_budget: int) -> PrefillPlanResult:
        # (copy body from StepScheduler._build_prefill_plan, return PrefillPlanResult)
        ...
```

- [ ] **Step 4: Re-export from planners package**

Update `vllm/engine/planners/__init__.py`:

```python
from vllm.engine.planners.decode_prefill_planner import DecodePrefillPlanner
```

- [ ] **Step 5: Run prefill tests**

```bash
uv run pytest tests/test_decode_prefill_planner.py::test_build_prefill_plan_selects_first_chunk -v
```

Expected: `1 passed`

- [ ] **Step 6: Lint changed files**

```bash
uv run ruff check vllm/engine/planners/ tests/test_decode_prefill_planner.py tests/test_planner_types.py
```

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add vllm/engine/planners/ tests/test_decode_prefill_planner.py tests/test_planner_types.py
git commit -m "feat(planners): DecodePrefillPlanner with prefill plan assembly"
```

---

## Task 3: Add `DecodePrefillPlanner.build_decode_plan`

**Files:**
- Modify: `vllm/engine/planners/decode_prefill_planner.py`
- Modify: `vllm/engine/planners/types.py` (already done in Task 1)
- Test: `tests/test_decode_prefill_planner.py`

**Interfaces:**
- Consumes: `DecodePlanResult` from Task 1; existing decode helpers in `StepScheduler`.
- Produces: `DecodePrefillPlanner.build_decode_plan(scheduler, decodes, decode_limit, no_prefills_selected) -> DecodePlanResult`.

- [ ] **Step 1: Write the failing decode test**

Append to `tests/test_decode_prefill_planner.py`:

```python
def test_build_decode_plan_fast_path() -> None:
    planner = DecodePrefillPlanner(
        service_class_weights={"latency": 4},
        decode_service_class_quotas={},
        max_prefill_lora_adapters_per_batch=0,
        max_decode_lora_adapters_per_batch=0,
        max_prefill_multimodal_requests_per_batch=0,
        max_decode_multimodal_requests_per_batch=0,
        max_prefill_multimodal_lora_requests_per_batch=0,
        max_decode_multimodal_lora_requests_per_batch=0,
        lora_fairness_relax_threshold=0.0,
        lora_locality_tighten_threshold=0.0,
        lora_limit_relax_delta=1,
        lora_limit_tighten_delta=1,
        multimodal_prefix_cache_relax_threshold=0.2,
        multimodal_prefix_cache_tighten_threshold=0.8,
        multimodal_prefill_limit_relax_delta=1,
        multimodal_prefill_limit_tighten_delta=1,
        multimodal_lora_prefill_limit_relax_delta=1,
        multimodal_lora_prefill_limit_tighten_delta=1,
        multimodal_lora_fairness_relax_threshold=0.0,
        multimodal_lora_locality_tighten_threshold=0.0,
        prefill_chunk_size=4,
        prefill_microbatch_size=2,
    )
    scheduler = RequestScheduler(max_active_requests=3)
    for i in range(3):
        req = RequestState(
            request_id=f"r{i}",
            prompt="",
            input_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            slot_idx=i,
            is_prefill=False,
            seq_len=5,
            generated_ids=[10],
        )
        scheduler.add_request(f"r{i}", req)
    decodes, _ = scheduler.classify_requests()
    result = planner.build_decode_plan(scheduler, decodes, decode_limit=10, no_prefills_selected=True)
    assert result.plan is not None
    assert result.plan.use_fast_path is True
    assert result.plan.request_ids == ["r0", "r1", "r2"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_decode_prefill_planner.py::test_build_decode_plan_fast_path -v
```

Expected: `AttributeError: 'DecodePrefillPlanner' object has no attribute 'build_decode_plan'`

- [ ] **Step 3: Implement `build_decode_plan`**

In `vllm/engine/planners/decode_prefill_planner.py`, add:

```python
from vllm.engine.planners.types import DecodePlanResult, PrefillPlanResult
from vllm.engine.step_plan import DecodePlan, PrefillPlan
```

Add private helpers moved from `StepScheduler`:

```python
def _select_weighted_requests(
    self,
    *,
    scheduler,
    request_ids,
    limit,
    cursor_attr,
    prefer_short_prompts,
    intra_class_rotate,
    quotas,
):
    return self._service_class_selector.select_weighted_requests(
        scheduler=scheduler,
        step_scheduler=self,
        request_ids=request_ids,
        limit=limit,
        cursor_attr=cursor_attr,
        prefer_short_prompts=prefer_short_prompts,
        intra_class_rotate=intra_class_rotate,
        quotas=quotas,
    )

def _apply_multimodal_request_limit(
    self, *, scheduler, request_ids, max_multimodal_requests, cursor_attr
):
    return self._multimodal_constraint.apply_multimodal_request_limit(
        scheduler=scheduler,
        step_scheduler=self,
        request_ids=request_ids,
        max_multimodal_requests=max_multimodal_requests,
        cursor_attr=cursor_attr,
    )
```

Implement `build_decode_plan` by copying the body of `StepScheduler._build_decode_plan`, returning `DecodePlanResult(...)`.

- [ ] **Step 4: Run decode tests**

```bash
uv run pytest tests/test_decode_prefill_planner.py -v
```

Expected: both prefill and decode tests pass.

- [ ] **Step 5: Lint changed files**

```bash
uv run ruff check vllm/engine/planners/decode_prefill_planner.py tests/test_decode_prefill_planner.py
```

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add vllm/engine/planners/decode_prefill_planner.py tests/test_decode_prefill_planner.py
git commit -m "feat(planners): add decode plan assembly to DecodePrefillPlanner"
```

---

## Task 4: Refactor `StepScheduler` to use `DecodePrefillPlanner`

**Files:**
- Modify: `vllm/engine/step_scheduler.py`
- Delete: `tests/test_planner_types.py` (optional cleanup)
- Test: `tests/test_step_scheduler.py`

**Interfaces:**
- Consumes: `DecodePrefillPlanner`, `PrefillPlanResult`, `DecodePlanResult`.
- Produces: `StepScheduler` with thinner `build_plan` and no plan-assembly helpers.

- [ ] **Step 1: Verify the baseline passes before changes**

```bash
uv run pytest tests/test_step_scheduler.py -q
```

Expected: all pass.

- [ ] **Step 2: Instantiate the planner in `StepScheduler.__init__`**

In `vllm/engine/step_scheduler.py`, after `_budget_computer = BudgetComputer(...)`, add:

```python
self._decode_prefill_planner = DecodePrefillPlanner(
    service_class_weights=self.service_class_weights,
    decode_service_class_quotas=self.decode_service_class_quotas,
    max_prefill_lora_adapters_per_batch=self.max_prefill_lora_adapters_per_batch,
    max_decode_lora_adapters_per_batch=self.max_decode_lora_adapters_per_batch,
    max_prefill_multimodal_requests_per_batch=self.max_prefill_multimodal_requests_per_batch,
    max_decode_multimodal_requests_per_batch=self.max_decode_multimodal_requests_per_batch,
    max_prefill_multimodal_lora_requests_per_batch=self.max_prefill_multimodal_lora_requests_per_batch,
    max_decode_multimodal_lora_requests_per_batch=self.max_decode_multimodal_lora_requests_per_batch,
    lora_fairness_relax_threshold=self.lora_fairness_relax_threshold,
    lora_locality_tighten_threshold=self.lora_locality_tighten_threshold,
    lora_limit_relax_delta=self.lora_limit_relax_delta,
    lora_limit_tighten_delta=self.lora_limit_tighten_delta,
    multimodal_prefix_cache_relax_threshold=self.multimodal_prefix_cache_relax_threshold,
    multimodal_prefix_cache_tighten_threshold=self.multimodal_prefix_cache_tighten_threshold,
    multimodal_prefill_limit_relax_delta=self.multimodal_prefill_limit_relax_delta,
    multimodal_prefill_limit_tighten_delta=self.multimodal_prefill_limit_tighten_delta,
    multimodal_lora_prefill_limit_relax_delta=self.multimodal_lora_prefill_limit_relax_delta,
    multimodal_lora_prefill_limit_tighten_delta=self.multimodal_lora_prefill_limit_tighten_delta,
    multimodal_lora_fairness_relax_threshold=self.multimodal_lora_fairness_relax_threshold,
    multimodal_lora_locality_tighten_threshold=self.multimodal_lora_locality_tighten_threshold,
    prefill_chunk_size=self.prefill_chunk_size,
    prefill_microbatch_size=self.prefill_microbatch_size,
)
```

Add the import at the top:

```python
from vllm.engine.planners import DecodePrefillPlanner
```

- [ ] **Step 3: Replace `_build_prefill_plan` / `_build_decode_plan` calls in `build_plan`**

Replace:

```python
(
    prefill_plan,
    prefill_lora_gap,
    effective_prefill_multimodal_limit,
    ...
) = self._build_prefill_plan(scheduler, prefills, prefill_budget)
```

with:

```python
prefill_result = self._decode_prefill_planner.build_prefill_plan(
    scheduler, prefills, prefill_budget
)
prefill_plan = prefill_result.plan
prefill_lora_gap = prefill_result.fairness_gap
effective_prefill_multimodal_limit = prefill_result.effective_multimodal_request_limit
effective_prefill_lora_limit = prefill_result.effective_lora_adapter_limit
effective_prefill_multimodal_lora_limit = prefill_result.effective_multimodal_lora_request_limit
prefill_multimodal_limit_triggered = prefill_result.multimodal_limit_triggered
prefill_multimodal_lora_limit_triggered = prefill_result.multimodal_lora_limit_triggered
prefill_multimodal_relaxed = prefill_result.multimodal_limit_relaxed
prefill_multimodal_tightened = prefill_result.multimodal_limit_tightened
prefill_multimodal_lora_relaxed = prefill_result.multimodal_lora_limit_relaxed
prefill_multimodal_lora_tightened = prefill_result.multimodal_lora_limit_tightened
prefill_multimodal_lora_relaxed_by_fairness = prefill_result.multimodal_lora_relaxed_by_fairness
prefill_multimodal_lora_tightened_by_locality = prefill_result.multimodal_lora_tightened_by_locality
prefill_multimodal_lora_max_fairness_gap = prefill_result.multimodal_lora_max_fairness_gap
prefill_lora_relaxed = prefill_result.lora_limit_relaxed
prefill_lora_tightened = prefill_result.lora_limit_tightened
```

Replace the decode tuple unpack similarly using `decode_result`.

- [ ] **Step 4: Propagate runtime feedback to the planner**

In `StepScheduler.update_runtime_feedback`, after updating `self._multimodal_prefix_cache_hit_rate_feedback`, add:

```python
self._decode_prefill_planner.update_runtime_feedback(feedback)
```

- [ ] **Step 5: Remove migrated methods and cursor state from `StepScheduler`**

Delete from `StepScheduler.__init__`:

- `_prefill_rr_cursor`
- `_decode_rr_cursor`
- `_prefill_lora_cursor`
- `_decode_lora_cursor`
- `_prefill_multimodal_cursor`
- `_decode_multimodal_cursor`
- `_prefill_multimodal_lora_cursor`
- `_decode_multimodal_lora_cursor`
- `_decode_service_cursor`

Delete methods:

- `_build_prefill_plan`
- `_build_decode_plan`
- `_shape_multimodal_prefill_batch`
- `_effective_prefill_multimodal_lora_limit`
- `_effective_decode_multimodal_lora_limit`
- `_select_weighted_requests`
- `_apply_lora_adapter_batch_limit`
- `_apply_multimodal_request_limit`
- `_apply_multimodal_lora_request_limit`
- `_service_classes_for_ids`

Keep `_count_service_classes`, `_count_lora_adapters`, `_count_multimodal_requests`, `_count_multimodal_lora_requests`, `_max_abs_share_gap`, `_rotate_candidates` because they are still used by metrics/fast paths.

- [ ] **Step 6: Run step scheduler tests**

```bash
uv run pytest tests/test_step_scheduler.py -q
```

Expected: all pass.

- [ ] **Step 7: Run the new planner tests**

```bash
uv run pytest tests/test_decode_prefill_planner.py -q
```

Expected: all pass.

- [ ] **Step 8: Lint changed files**

```bash
uv run ruff check vllm/engine/step_scheduler.py vllm/engine/planners/
```

Expected: `All checks passed!`

- [ ] **Step 9: Commit**

```bash
git add vllm/engine/step_scheduler.py vllm/engine/planners/ tests/test_decode_prefill_planner.py
git commit -m "refactor(scheduler): StepScheduler delegates plan assembly to DecodePrefillPlanner"
```

---

## Task 5: Run regression and update docs

**Files:**
- Modify: `docs/ARCHITECTURE_LITE.md`, `docs/LITE_ONLY_STATUS.md`, `docs/README.md`, `README.md`
- Test: full regression suite

- [ ] **Step 1: Run the fast regression suite**

```bash
bash tests/run_regression_suite.sh
```

Expected: `134 passed, 2 skipped`.

- [ ] **Step 2: Run inference correctness regression**

```bash
SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh
```

Expected: exit 0, all Tier-B spotchecks PASS.

- [ ] **Step 3: Run e2e benchmark**

```bash
uv run python tests/e2e_full_benchmark.py --json-out /tmp/decode_prefill_e2e.json
```

Expected: exit 0.

- [ ] **Step 4: Update architecture docs**

Update the following documents to reflect that `StepScheduler` now delegates prefill/decode plan assembly to `DecodePrefillPlanner`:

- `docs/ARCHITECTURE_LITE.md` — Scheduling Model section.
- `docs/LITE_ONLY_STATUS.md` — Delivered Architecture Work section.
- `docs/README.md` — Maintained Runtime Path diagram.
- `README.md` — 当前主路径 / 核心特性 section.

- [ ] **Step 5: Lint all changed files**

```bash
uv run ruff check vllm/engine/step_scheduler.py vllm/engine/planners/ tests/test_decode_prefill_planner.py
uv run ruff format vllm/engine/step_scheduler.py vllm/engine/planners/ tests/test_decode_prefill_planner.py
```

Expected: `All checks passed!`

- [ ] **Step 6: Commit doc updates**

```bash
git add docs/ README.md
git commit -m "docs: reflect DecodePrefillPlanner in architecture docs"
```

---

## Self-Review

- **Spec coverage:**
  - Result dataclasses → Task 1.
  - Prefill plan assembly in planner → Task 2.
  - Decode plan assembly in planner → Task 3.
  - StepScheduler delegation and cleanup → Task 4.
  - Regression/docs → Task 5.
- **Placeholder scan:** no TBD/TODO; every step has concrete code/commands.
- **Type consistency:** `build_prefill_plan` returns `PrefillPlanResult`; `build_decode_plan` returns `DecodePlanResult`; both used consistently in Task 4.
