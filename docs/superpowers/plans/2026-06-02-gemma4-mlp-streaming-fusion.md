# Gemma4 MLP Streaming Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate and safely introduce Gemma4-31B M=1 AWQ MLP streaming fusion for `gate/up -> activation*up -> down_proj`.

**Architecture:** P2 starts with policy/audit/reference gates and an isolated microbenchmark. Runtime integration remains default-off until the isolated candidate beats the current verified path and passes layer-output numerical checks.

**Tech Stack:** Python 3.12, uv, PyTorch, Triton through `vllm/triton_utils`, Gemma4 LiteLinear/AWQ helpers.

---

### Task 1: Policy And Audit Guardrails

**Files:**
- Modify: `vllm/kernels/triton/awq_fused_gemm.py`
- Modify: `vllm/model_executor/models/_fused_awq_pair.py`
- Test: `tests/test_awq_gemm_m1_specialization.py`
- Test: `tests/test_awq_audit_summary.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert:

```python
from vllm.kernels.triton.awq_fused_gemm import awq_mlp_streaming_fusion_enabled


def test_mlp_streaming_fusion_default_disabled_and_policy_overridable() -> None:
    assert awq_mlp_streaming_fusion_enabled() is False
    assert awq_mlp_streaming_fusion_enabled(policy={"awq_mlp_streaming_fusion": True}) is True
```

and:

```python
record_awq_audit_event("model.layers.0.mlp", "mlp_streaming_fallback", reason="disabled")
summary = get_awq_runtime_audit_summary()
assert summary["events"]["mlp_streaming_fallback"] == 1
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/test_awq_gemm_m1_specialization.py::test_mlp_streaming_fusion_default_disabled_and_policy_overridable tests/test_awq_audit_summary.py::test_awq_audit_summary_tracks_mlp_streaming_events -q
```

Expected: first test fails because `awq_mlp_streaming_fusion_enabled` does not exist.

- [ ] **Step 3: Add minimal policy helper**

Add:

```python
def awq_mlp_streaming_fusion_enabled(
    *,
    config: Any | None = None,
    policy: dict[str, object] | None = None,
) -> bool:
    raw = _kernel_policy_value(config, policy, "awq_mlp_streaming_fusion", False)
    return raw if isinstance(raw, bool) else truthy(raw)
```

- [ ] **Step 4: Run tests to verify GREEN**

Run the same targeted pytest command and confirm PASS.

### Task 2: Reference Numerical Gate

**Files:**
- Create: `tests/tools/validate_gemma4_mlp_streaming_reference.py`
- Test: `tests/test_awq_audit_summary.py`

- [ ] **Step 1: Add reference script**

Create a tool that loads one Gemma4 MLP layer, computes:

```text
current = down_proj(fused_gate_up_activation(x))
reference = down_proj(silu(gate_proj(x)) * up_proj(x))
```

and reports `max_diff`, `mean_diff`, `cosine`, and `argmax_match` for fp32 and fp16/bf16 accumulation where available.

- [ ] **Step 2: Run reference gate**

Run:

```bash
uv run python tests/tools/validate_gemma4_mlp_streaming_reference.py \
  --model models/gemma-4-31B-it-AWQ-4bit \
  --layer 0 \
  --dtype bfloat16 \
  --json-out /tmp/gemma31_mlp_reference_layer0.json
```

Expected: existing two-stage fused path matches unfused reference within documented tolerance.

### Task 3: Isolated Microbenchmark Scaffold

**Files:**
- Create: `tests/tools/benchmark_gemma4_mlp_streaming_m1.py`
- Modify: `docs/GEMMA4_31B_P1_WRAPUP.md` or create `docs/GEMMA4_31B_P2_MLP_STREAMING.md`

- [ ] **Step 1: Add benchmark script**

Compare:

```text
current_two_stage = fused_gate_up_activation + down_proj group32 GEMV
candidate_half_fusion = fused_gate_up_activation with tuned output layout + down_proj
candidate_streaming = only when a Triton candidate exists
```

For missing candidates, report `"status": "not_implemented"` instead of silently skipping.

- [ ] **Step 2: Run baseline benchmark**

Run:

```bash
uv run python tests/tools/benchmark_gemma4_mlp_streaming_m1.py \
  --warmup 8 \
  --repeat 64 \
  --json-out /tmp/gemma31_mlp_streaming_baseline.json
```

Expected: baseline current two-stage latency recorded.

### Task 4: Runtime Integration Gate

**Files:**
- Modify: `vllm/model_executor/models/gemma4/mlp.py`
- Modify: `vllm/model_executor/models/_fused_awq_pair.py`
- Test: `tests/test_awq_audit_summary.py`

- [ ] **Step 1: Add default-off helper**

Add `try_fused_awq_mlp_streaming(...)` that:

```text
returns None when awq_mlp_streaming_fusion is false
records mlp_streaming_fallback with reason=disabled
requires M=1, group_size=32, symmetric packed-int4, no LoRA, no bias
returns None for unsupported shapes
```

- [ ] **Step 2: Wire Gemma4MLP before current gate/up path**

In `Gemma4MLP.forward`, attempt streaming helper first. Keep the existing gate/up fused path as fallback.

- [ ] **Step 3: Verify default behavior unchanged**

Run:

```bash
uv run pytest tests/test_awq_audit_summary.py tests/test_awq_gemm_m1_specialization.py -q
bash tests/run_regression_suite.sh
```

Expected: PASS, no default behavior change.

### Task 5: Optional Triton Candidate

**Files:**
- Modify: `vllm/kernels/triton/awq_fused_gemm.py`
- Test: `tests/test_awq_gemm_m1_specialization.py`

- [ ] **Step 1: Implement only after benchmark gate**

Implement a candidate only if Task 3 shows a specific design can avoid recomputing gate/up per down output tile.

- [ ] **Step 2: Compare against current path**

Accept only if isolated single-layer MLP latency improves by at least 15% over current two-stage path and passes Task 2 numerical checks.

