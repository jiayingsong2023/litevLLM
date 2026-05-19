# Runtime Factory Assembly Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the loose `LiteRuntimeFactory.build(engine: Any)` assembly boundary with an explicit `RuntimeAssemblyContext`.

**Architecture:** `LiteEngine` remains the owner of initialization, but it packages the fields needed by `LiteRuntimeFactory` into a dataclass before assembly. `LiteRuntimeFactory` consumes that dataclass, reducing dependence on `LiteEngine` private field names and making factory tests construct a stable contract directly.

**Tech Stack:** Python 3.12, dataclasses, pytest, existing FastInference lite runtime modules.

---

### Task 1: Runtime Factory Contract

**Files:**
- Modify: `tests/test_runtime_factory.py`
- Modify: `vllm/engine/runtime_factory.py`
- Modify: `vllm/engine/lite_engine.py`

- [ ] **Step 1: Write the failing test**

Update `tests/test_runtime_factory.py` to import `RuntimeAssemblyContext` and instantiate it directly. The test should call `LiteRuntimeFactory.build(context)` and assert the same component and policy propagation behavior as the previous loose-engine test.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runtime_factory.py -q`

Expected: FAIL because `RuntimeAssemblyContext` is not exported yet.

- [ ] **Step 3: Write minimal implementation**

Add `RuntimeAssemblyContext` to `vllm/engine/runtime_factory.py`, type `LiteRuntimeFactory.build()` to accept it, and replace all `engine.*` reads with `context.*` reads. Update `LiteEngine` to construct the context immediately before factory assembly.

- [ ] **Step 4: Run local tests**

Run:

```bash
uv run pytest tests/test_runtime_factory.py -q
uv run pytest tests/test_runtime_controller.py tests/test_step_scheduler.py -q
bash tests/run_regression_suite.sh
```

Expected: all commands exit 0.

- [ ] **Step 5: Run correctness and performance gates**

Run:

```bash
SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh
uv run python tests/e2e_full_benchmark.py
```

Expected: both commands complete successfully, or any failure is investigated before further changes.
