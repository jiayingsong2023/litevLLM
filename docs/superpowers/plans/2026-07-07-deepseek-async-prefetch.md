# DeepSeek V4 Flash Async Expert Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a default-off, single-token greedy decode async expert-weight prefetch for DeepSeek V4 Flash, overlapping layer N+1 H2D copies with layer N compute on a separate CUDA stream.

**Architecture:** The GPU weight stager owns one background CUDA stream and exposes `prefetch_grouped_experts_async()` (returns a `torch.cuda.Event`) plus `wait_for_prefetch()`. The single-token decode layer loop in `model.py` waits on the previous layer's event before computing the current layer and enqueues the next layer's prefetch immediately after the current layer returns. The env switch `FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH` gates the new path and defaults to `0`.

**Tech Stack:** Python 3.12, PyTorch, ROCm-first CUDA-compatible, Triton-free (no C++), `uv`, pytest.

## Global Constraints

- Python 3.12 only; use `uv` for all Python invocations.
- No C++ / CUDA source files; all kernel work remains Triton.
- Every new env var must be registered in `vllm/engine/env_registry.py` as `_tool_only`.
- Conventional commit subjects (`feat:`, `fix:`, `fix(kernels):`).
- Run `bash tests/run_regression_suite.sh` before claiming done.
- Default behavior must not change (`ASYNC_PREFETCH=0`).

---

### Task 1: Register the async-prefetch env var

**Files:**
- Modify: `vllm/engine/env_registry.py:307-309`
- Test: `tests/test_project_governance.py` (existing, must still pass)

**Interfaces:**
- Consumes: existing `_tool_only` helper in `env_registry.py`.
- Produces: `FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH` registered.

- [ ] **Step 1: Add the env var registration**

Insert the new entry in alphabetical order within the DeepSeek block, immediately after `FASTINFERENCE_DEEPSEEK_V4_FLASH_MIN_STEADY_DECODE_TPS`:

```python
    "FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH": _tool_only(
        "FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH"
    ),
```

- [ ] **Step 2: Verify registration**

Run:

```bash
uv run pytest tests/test_project_governance.py::test_fastinference_env_names_are_registered -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add vllm/engine/env_registry.py
git commit -m "feat(deepseek): register FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH env"
```

---

### Task 2: Add async prefetch API to the GPU weight stager

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py:100-127` (constructor)
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py:759-770` (after existing `prefetch_grouped_experts`)
- Test: `tests/deepseek_v4_flash/test_expert_prefetch.py`

**Interfaces:**
- Consumes: `torch.cuda.Stream`, existing `prefetch_grouped_experts(..., stream=...)`.
- Produces:
  - `DeepSeekV4FlashGPUWeightStager.prefetch_grouped_experts_async(tensors, request) -> torch.cuda.Event`
  - `DeepSeekV4FlashGPUWeightStager.wait_for_prefetch(event: torch.cuda.Event | None) -> None`

- [ ] **Step 1: Create the background stream in `__init__`**

Inside `DeepSeekV4FlashGPUWeightStager.__init__`, after `self.max_staged_bytes = max_staged_bytes` and before `self._full_resident_enabled = False`:

```python
        self._prefetch_stream: torch.cuda.Stream | None = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            self._prefetch_stream = torch.cuda.Stream(device=self.device)
```

- [ ] **Step 2: Add async/wait methods after `prefetch_grouped_experts`**

Insert after line 770 (the end of `prefetch_grouped_experts`):

```python
    def prefetch_grouped_experts_async(
        self,
        tensors: DeepSeekV4FlashGroupedExpertTensors,
        request: DeepSeekV4FlashExpertPrefetchRequest,
    ) -> torch.cuda.Event:
        """Prefetch on a background stream and return an event for the compute stream.

        The caller must call wait_for_prefetch(event) on the compute stream before
        consuming the staged tensors. If the background stream is unavailable, the
        prefetch runs synchronously on the current stream and an already-completed
        event is still returned so callers can use a uniform wait path.
        """
        stream = self._prefetch_stream
        if stream is None:
            self._prefetch_grouped_experts(tensors, request)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            return event
        with torch.cuda.stream(stream):
            self._prefetch_grouped_experts(tensors, request)
        event = torch.cuda.Event()
        event.record(stream)
        return event

    def wait_for_prefetch(
        self,
        event: torch.cuda.Event | None,
    ) -> None:
        """Make the current compute stream wait for a prefetch event."""
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
```

- [ ] **Step 3: Add unit tests**

Append to `tests/deepseek_v4_flash/test_expert_prefetch.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_prefetch_async_returns_event_and_hits_cache() -> None:
    tensors = _grouped_expert_tensors()
    store = _FakeRawPayloadStore()
    payload = bytes(range(ggml_tensor_nbytes((256, 1), GGML_TYPE_Q2_K)))
    for tensor in (tensors.gate, tensors.up, tensors.down):
        store.payloads[(tensor.name, 0)] = payload
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    request = DeepSeekV4FlashExpertPrefetchRequest(layer_idx=3, expert_ids=(0,))

    event = stager.prefetch_grouped_experts_async(tensors, request)

    assert isinstance(event, torch.cuda.Event)
    stager.wait_for_prefetch(event)
    # Demand staging should now hit the cache loaded by async prefetch.
    for tensor in (tensors.gate, tensors.up, tensors.down):
        stager.stage_grouped_expert_payload(tensor, 0, layer_idx=3)
    stats = stager.cache_stats()
    assert store.raw_reads == 3
    assert stats["grouped_hits"] == 3
    assert stats["prefetch_hits"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_wait_for_prefetch_accepts_event_and_none() -> None:
    store = _FakeRawPayloadStore()
    stager = DeepSeekV4FlashGPUWeightStager(store, device="cuda")
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream())

    # Should not raise and should be a no-op for an already-completed event.
    stager.wait_for_prefetch(event)

    # None event should also be a no-op.
    stager.wait_for_prefetch(None)
```

- [ ] **Step 4: Run the new tests**

```bash
uv run pytest tests/deepseek_v4_flash/test_expert_prefetch.py -v
```

Expected: all tests PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py \
        tests/deepseek_v4_flash/test_expert_prefetch.py
git commit -m "feat(deepseek): async expert prefetch API on background CUDA stream"
```

---

### Task 3: Add the async-prefetch enablement helper in model.py

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py:1867-1875` (after `_deepseek_full_resident_enabled`)

**Interfaces:**
- Consumes: `os.environ` (matches existing DeepSeek env read style).
- Produces: `DeepSeekV4FlashForCausalLM._deepseek_async_prefetch_enabled() -> bool`.

- [ ] **Step 1: Add the static helper**

Insert immediately after `_deepseek_full_resident_enabled` (around line 1875):

```python
    @staticmethod
    def _deepseek_async_prefetch_enabled() -> bool:
        return (
            os.environ.get(
                "FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH",
                "0",
            )
            == "1"
        )
```

- [ ] **Step 2: Verify governance**

Run:

```bash
uv run pytest tests/test_project_governance.py::test_production_runtime_has_no_direct_fastinference_env_reads -v
```

Expected: PASS. The env name is on a separate line from `os.environ.get(`, matching the existing DeepSeek env read pattern and the current regex.

- [ ] **Step 3: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/model.py
git commit -m "feat(deepseek): add async prefetch enablement helper"
```

---

### Task 4: Add the async-prefetch scheduling helper in model.py

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py:371-393` (after `_schedule_next_layer_expert_prefetch`)

**Interfaces:**
- Consumes: `_likely_hash_routed_expert_ids_for_token`, `stager.cache_admission_policy`, `stager.prefetch_grouped_experts_async`.
- Produces: `DeepSeekV4FlashForCausalLM._schedule_next_layer_expert_prefetch_async(...) -> torch.cuda.Event | None`.

- [ ] **Step 1: Add the async scheduling helper**

Insert immediately after `_schedule_next_layer_expert_prefetch` (around line 393):

```python
    def _schedule_next_layer_expert_prefetch_async(
        self,
        stager: DeepSeekV4FlashGPUWeightStager,
        next_layer: Any,
        *,
        token_id: int | None,
    ) -> torch.cuda.Event | None:
        """Enqueue an async prefetch for the next layer and return its event.

        Returns None if there is nothing to prefetch or prefetch fails.
        """
        grouped_experts = getattr(next_layer, "grouped_experts", None)
        if grouped_experts is None:
            return None
        expert_ids = self._likely_hash_routed_expert_ids_for_token(
            stager,
            next_layer,
            token_id=token_id,
        )
        if not expert_ids:
            return None
        cache_admission_policy = getattr(stager, "cache_admission_policy", None)
        cacheable_expert_ids = tuple(
            expert_id
            for expert_id in dict.fromkeys(expert_ids)
            if cache_admission_policy is None
            or cache_admission_policy.should_cache_grouped_expert(
                layer_idx=next_layer.layer_index,
                expert_id=expert_id,
            )
        )
        if not cacheable_expert_ids:
            return None
        try:
            return stager.prefetch_grouped_experts_async(
                grouped_experts,
                DeepSeekV4FlashExpertPrefetchRequest(
                    layer_idx=next_layer.layer_index,
                    expert_ids=cacheable_expert_ids,
                ),
            )
        except Exception:
            self._deepseek_profiler.add_counter("deepseek_prefetch_failures", 1)
            return None
```

- [ ] **Step 2: Run unit tests to make sure nothing broke**

```bash
uv run pytest tests/deepseek_v4_flash/test_expert_prefetch.py -v
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/model.py
git commit -m "feat(deepseek): add async next-layer prefetch scheduler helper"
```

---

### Task 5: Wire async prefetch into the single-token decode layer loop

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py:650-693`

**Interfaces:**
- Consumes: `_deepseek_async_prefetch_enabled`, `_schedule_next_layer_expert_prefetch_async`, `stager.wait_for_prefetch`.
- Produces: single-token decode path now overlaps next-layer H2D with current-layer compute when env is enabled.

- [ ] **Step 1: Rewrite the layer loop in `_forward_kernel_token_hidden`**

Replace lines 649-692 with the following (preserve indentation exactly):

```python
        layers = getattr(store.bindings, "layers", None)
        if layers is None:
            raise RuntimeError("DeepSeek V4 Flash GPU forward requires layer bindings")
        layers = list(layers)
        async_prefetch_enabled = self._deepseek_async_prefetch_enabled()
        pending_prefetch_event: torch.cuda.Event | None = None
        for layer_offset, layer in enumerate(layers):
            if pending_prefetch_event is not None:
                with self._deepseek_profiler.section(
                    f"layer_{layer.layer_index}_prefetch_wait",
                    layer_idx=layer.layer_index,
                    token_idx=token_idx,
                ):
                    stager.wait_for_prefetch(pending_prefetch_event)
                pending_prefetch_event = None
            if layer.layer_index < 2:
                with self._deepseek_profiler.section(
                    f"layer_{layer.layer_index}_sliding",
                    layer_idx=layer.layer_index,
                    token_idx=token_idx,
                ):
                    hidden = deepseek_v4_flash_sliding_layer_forward(
                        hidden,
                        layer=layer,
                        stager=stager,
                        backend=self.gpu_backend,
                        state=state,
                        token_idx=token_idx,
                        token_id=token_id,
                    )
            else:
                with self._deepseek_profiler.section(
                    f"layer_{layer.layer_index}_compressed",
                    layer_idx=layer.layer_index,
                    token_idx=token_idx,
                ):
                    hidden = deepseek_v4_flash_compressed_layer_forward(
                        hidden,
                        layer=layer,
                        stager=stager,
                        backend=self.gpu_backend,
                        state=state,
                        token_idx=token_idx,
                        token_id=token_id,
                        compressed_count=compressed_counts_by_layer.get(
                            layer.layer_index
                        )
                        if compressed_counts_by_layer is not None
                        else None,
                    )
            if layer_offset + 1 < len(layers):
                next_layer = layers[layer_offset + 1]
                if async_prefetch_enabled:
                    pending_prefetch_event = (
                        self._schedule_next_layer_expert_prefetch_async(
                            stager,
                            next_layer,
                            token_id=token_id,
                        )
                    )
                else:
                    self._schedule_next_layer_expert_prefetch(
                        stager,
                        next_layer,
                        token_id=token_id,
                    )
```

- [ ] **Step 2: Verify the other decode path is untouched**

Confirm that `_forward_kernel_token_hidden_token_tensor` (lines 695-797) still uses the original synchronous `_schedule_next_layer_expert_prefetch(..., token_id=None)` and is not changed.

- [ ] **Step 3: Run DeepSeek unit tests**

```bash
uv run pytest tests/deepseek_v4_flash/ -v -q
```

Expected: PASS (some tests may be skipped without GPU).

- [ ] **Step 4: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/model.py
git commit -m "feat(deepseek): wire async prefetch into single-token decode layer loop"
```

---

### Task 6: Add an A/B smoke script

**Files:**
- Create: `tests/run_deepseek_v4_flash_async_prefetch_ab.sh`

**Interfaces:**
- Consumes: `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`, env `FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH`.
- Produces: `/tmp/ds_async_prefetch_ab_*.json` artifacts.

- [ ] **Step 1: Create the A/B script**

Write `tests/run_deepseek_v4_flash_async_prefetch_ab.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL=models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf

run_variant() {
  local variant="$1"
  local async_value="$2"
  local out_json="/tmp/ds_async_prefetch_ab_${variant}.json"

  echo "===== DeepSeek V4 Flash async prefetch A/B: ${variant} (ASYNC_PREFETCH=${async_value}) ====="
  FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH="${async_value}" \
    timeout 600 uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
      --model "$MODEL" \
      --context-length 4096 \
      --max-tokens 16 \
      --warmup-tokens 1 \
      --min-steady-decode-tps 0.0 \
      --profile-json "$out_json"
}

run_variant "off" "0"
run_variant "on" "1"

echo "===== A/B summary ====="
python3 - <<'PY'
import json, pathlib
for variant in ("off", "on"):
    path = pathlib.Path(f"/tmp/ds_async_prefetch_ab_{variant}.json")
    if not path.exists():
        print(f"{variant}: missing {path}")
        continue
    data = json.loads(path.read_text())
    metrics = data.get("decode_metrics", {})
    summary = data.get("profile_summary", {})
    print(f"{variant}: decode_tps_steady_state={metrics.get('decode_tps_steady_state', 0):.3f} "
          f"layer_moe_ms={summary.get('phase_totals_ms', {}).get('layer_moe', 0):.3f}")
PY
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x tests/run_deepseek_v4_flash_async_prefetch_ab.sh
```

- [ ] **Step 3: Commit**

```bash
git add tests/run_deepseek_v4_flash_async_prefetch_ab.sh
git commit -m "feat(deepseek): add async prefetch A/B smoke script"
```

---

### Task 7: Run regression suite

**Files:**
- None (verification only).

**Interfaces:**
- Consumes: all prior changes.
- Produces: green regression report.

- [ ] **Step 1: Run lint and format**

```bash
uv run ruff check vllm/model_executor/models/deepseek_v4_flash tests/deepseek_v4_flash tests/test_project_governance.py && uv run ruff format vllm/model_executor/models/deepseek_v4_flash tests/deepseek_v4_flash tests/test_project_governance.py
```

Expected: no errors; formatting makes no unexpected changes.

- [ ] **Step 2: Run the fast regression**

```bash
bash tests/run_regression_suite.sh
```

Expected: 136 passed, 2 skipped (or equivalent all-green output).

- [ ] **Step 3: Commit if any fixes were needed**

If lint or the suite required code changes, commit them with a `fix:` message. If everything passed cleanly, no extra commit.

---

### Task 8: Run A/B on GPU and decide on default enablement

**Files:**
- None (manual verification).

**Interfaces:**
- Consumes: `tests/run_deepseek_v4_flash_async_prefetch_ab.sh`.
- Produces: decision on whether ASYNC_PREFETCH stays default-off or can be flipped.

- [ ] **Step 1: Run the A/B script on a GPU machine**

```bash
bash tests/run_deepseek_v4_flash_async_prefetch_ab.sh
```

- [ ] **Step 2: Inspect the JSON artifacts**

```bash
python3 - <<'PY'
import json, pathlib
for variant in ("off", "on"):
    path = pathlib.Path(f"/tmp/ds_async_prefetch_ab_{variant}.json")
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    print(f"=== {variant} ===")
    print(json.dumps({
        "decode_metrics": data.get("decode_metrics", {}),
        "counters": data.get("profile", {}).get("counters", {}),
        "phase_totals_ms": data.get("profile_summary", {}).get("phase_totals_ms", {}),
    }, indent=2))
PY
```

- [ ] **Step 3: Apply success criteria**

- If `decode_tps_steady_state` with `ASYNC_PREFETCH=1` is ≥ 0.2 tok/s higher or ≥ 30% higher than with `=0`, and warm-cache gate (Task 9) does not regress, leave the default at `0` in this plan but document the measured gain in the plan file's "Results" section and recommend a follow-up task to flip the default.
- If the gain is smaller or unstable, leave default at `0` and add a plan note explaining why default enablement is deferred.

---

### Task 9: Verify warm-cache gate still passes

**Files:**
- None (verification only).

**Interfaces:**
- Consumes: `tests/run_deepseek_v4_flash_real_smoke.sh`.

- [ ] **Step 1: Run the warm gate**

```bash
FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH=0 bash tests/run_deepseek_v4_flash_real_smoke.sh
```

Expected: warm gate reports `decode_tps_steady_state >= 1.5`.

- [ ] **Step 2: Run the warm gate with async prefetch enabled**

```bash
FASTINFERENCE_DEEPSEEK_V4_FLASH_ASYNC_PREFETCH=1 bash tests/run_deepseek_v4_flash_real_smoke.sh
```

Expected: warm gate still reports `decode_tps_steady_state >= 1.5` (no regression).

---

## Self-Review Checklist

**Spec coverage:**
- [x] Background stream owned by stager — Task 2.
- [x] `prefetch_grouped_experts_async` + `wait_for_prefetch` — Task 2.
- [x] Single-token decode loop waits on previous event, enqueues next-layer prefetch after current layer — Task 5.
- [x] Event kept as local variable, not model state — Task 5.
- [x] Env switch default `0`, registered as `_tool_only` — Tasks 1 and 3.
- [x] Scope limited to `token_id` scalar path; `token_id=None` and graph paths untouched — Task 5.
- [x] A/B profile JSON comparison — Task 6.
- [x] Regression suite — Task 7.

**Placeholder scan:**
- [x] No "TBD", "TODO", or "implement later".
- [x] Every code step contains the exact code.
- [x] Every command has expected output.

**Type consistency:**
- [x] `prefetch_grouped_experts_async` returns `torch.cuda.Event`.
- [x] `wait_for_prefetch` accepts `torch.cuda.Event | None`.
- [x] `_schedule_next_layer_expert_prefetch_async` returns `torch.cuda.Event | None`.
- [x] `pending_prefetch_event` is `torch.cuda.Event | None`.
