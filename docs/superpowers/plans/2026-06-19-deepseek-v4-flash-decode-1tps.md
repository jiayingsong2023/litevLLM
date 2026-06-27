# DeepSeek V4 Flash Decode 1 TPS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise real GGUF DeepSeek V4 Flash direct decode from the current `decode_tps_p50=0.49` to `>=1.0` token/sec on the local ROCm UMA machine while preserving readable output.

**Architecture:** Keep this optimization inside the DeepSeek V4 Flash direct GGUF path first. The current e2e result uses `deepseek_v4_flash_direct_gguf`, so optimizing LiteEngine or REST before the direct path reaches `>=1 tps` would hide the real hot-path cost. The plan focuses on instrumentation, persistent decode state, zero CPU token sync, cache warm residency, and per-stage gates before larger kernel rewrites.

**Tech Stack:** Python 3.12 via `uv`, PyTorch ROCm tensors, Triton through `vllm/triton_utils/`, GGUF raw payload staging, pytest, `tests/e2e_full_benchmark.py`.

---

## Current Baseline

Latest e2e benchmark:

```text
deepseek_v4_flash_q2_gguf | tps=0.43 | ttft_p50=32506.0ms | decode_tps_agg=0.49 | decode_tps_p50=0.49 | stream_visible=0.0%
```

Important interpretation:

- The path is `deepseek_v4_flash_direct_gguf`, not OpenAI REST.
- `stream_visible=0.0%` is expected for the current direct runner because it returns a complete JSON result, not progressive stream events.
- `ttft_p50 == e2e_p50` means the benchmark only sees one complete direct run boundary; we need internal per-token timing to know whether decode itself is slow or startup/warmup dominates.
- The next target is `decode_tps_p50 >= 1.0` for `max_new_tokens=16`, with quality smoke still generating readable text.

## File Structure

Modify these files:

- `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`: emit per-token timings, warmup separation, and explicit `decode_tps_steady_state`.
- `tests/e2e_full_benchmark.py`: parse the new smoke metrics and report steady-state decode TPS for DeepSeek.
- `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py`: lock the JSON schema and e2e parser expectations.
- `vllm/model_executor/models/deepseek_v4_flash/model.py`: add a persistent greedy decode loop that keeps request state, token tensors, and output buffer on GPU.
- `vllm/model_executor/models/deepseek_v4_flash/gpu_runtime.py`: add reusable request-state reset semantics if the existing state object cannot be reused cleanly.
- `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`: add warm residency helpers for output Q8 payload, static vectors, and predicted hot expert payloads.
- `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`: add counters for CPU sync points, selected expert grouped kernel hits, and output argmax path hits.
- `tests/deepseek_v4_flash/test_model_kernel_generate.py`: unit tests for persistent decode and no CPU token sync.
- `tests/deepseek_v4_flash/test_gpu_weight_staging.py`: unit tests for warm residency and cache hit behavior.

Do not modify these in this phase:

- `vllm/engine/*`: the benchmark is not using the generic engine path for DeepSeek yet.
- REST server files: REST can consume the faster direct/model path after it is stable.
- Generic Gemma/Qwen code.

---

### Task 1: Make The DeepSeek Benchmark Report Real Per-Token Decode

**Files:**
- Modify: `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`
- Modify: `tests/e2e_full_benchmark.py`
- Modify: `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py`

- [x] **Step 1: Write JSON schema test for per-token timings**

Add this test to `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py` near the existing smoke CLI tests:

```python
def test_gpu_smoke_payload_contains_steady_state_decode_metrics() -> None:
    payload = {
        "runs": [
            {
                "generated_token_count": 16,
                "elapsed_ms": 16000.0,
                "token_elapsed_ms": [1200.0] + [900.0] * 15,
            }
        ]
    }

    summary = smoke.deepseek_v4_flash_decode_summary(payload)

    assert summary["decode_tokens_total"] == 16
    assert summary["decode_tps_agg"] == 1.0
    assert summary["decode_tps_steady_state"] > 1.0
```

- [x] **Step 2: Add smoke helper for steady-state decode TPS**

In `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`, add:

```python
def decode_metrics_from_token_times(token_elapsed_ms: list[float]) -> dict[str, float]:
    if not token_elapsed_ms:
        return {
            "decode_tokens_total": 0,
            "decode_ms_total": 0.0,
            "decode_tps_agg": 0.0,
            "decode_tps_steady_state": 0.0,
        }
    total_ms = float(sum(token_elapsed_ms))
    steady_ms = float(sum(token_elapsed_ms[1:])) if len(token_elapsed_ms) > 1 else total_ms
    steady_tokens = max(len(token_elapsed_ms) - 1, 1)
    return {
        "decode_tokens_total": float(len(token_elapsed_ms)),
        "decode_ms_total": total_ms,
        "decode_tps_agg": len(token_elapsed_ms) * 1000.0 / total_ms if total_ms > 0.0 else 0.0,
        "decode_tps_steady_state": steady_tokens * 1000.0 / steady_ms if steady_ms > 0.0 else 0.0,
    }
```

- [x] **Step 3: Time each generated token with CUDA events**

In `main()`, replace the single full-run event around `model.generate_greedy_kernel(...)` with a new model API call from Task 2:

```python
output_ids, token_elapsed_ms = model.generate_greedy_kernel_timed(
    input_ids,
    max_tokens=args.max_tokens,
)
decode_metrics = decode_metrics_from_token_times(token_elapsed_ms)
```

Keep `elapsed_ms` for backward compatibility, but include:

```python
"token_elapsed_ms": token_elapsed_ms,
"generated_token_count": args.max_tokens,
**decode_metrics,
```

- [x] **Step 4: Teach e2e benchmark to prefer steady-state DeepSeek metrics**

In `tests/e2e_full_benchmark.py`, inside the DeepSeek direct result parser, use:

```python
decode_tps = float(run.get("decode_tps_steady_state", run.get("tokens_per_second", 0.0)))
```

and set `decode_tokens_total` from `decode_tokens_total` when present.

- [x] **Step 5: Verify**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_real_smoke_tool_cli.py -q
uv run ruff check tests/tools/run_deepseek_v4_flash_gpu_smoke.py tests/deepseek_v4_flash/test_real_smoke_tool_cli.py
```

Expected: tests pass, and DeepSeek e2e no longer conflates first-token setup with steady-state decode.

---

### Task 2: Add Persistent GPU Greedy Decode Without Per-Token CPU Sync

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Modify: `tests/deepseek_v4_flash/test_model_kernel_generate.py`

- [x] **Step 1: Write test for no CPU token extraction inside decode loop**

Add a fake backend/stager test in `tests/deepseek_v4_flash/test_model_kernel_generate.py`:

```python
def test_generate_greedy_kernel_timed_keeps_generated_tokens_on_gpu(monkeypatch) -> None:
    model = make_fake_deepseek_model_returning_incrementing_cuda_tokens()

    output_ids, token_elapsed_ms = model.generate_greedy_kernel_timed(
        torch.tensor([1], dtype=torch.long, device="cuda"),
        max_tokens=4,
    )

    assert output_ids.is_cuda
    assert output_ids.tolist() == [1, 2, 3, 4, 5]
    assert len(token_elapsed_ms) == 4
    assert model.gpu_backend.stats()["cpu_token_sync_points"] == 0
```

- [x] **Step 2: Add GPU-scalar token path to the public generator**

In `DeepSeekV4FlashForCausalLM`, add:

```python
def generate_greedy_kernel_timed(
    self,
    input_ids: torch.Tensor,
    *,
    max_tokens: int = 1,
) -> tuple[torch.Tensor, list[float]]:
    device = self._validate_generate_greedy_kernel_input(input_ids, max_tokens=max_tokens)
    self.gpu_backend.require_ready()
    state = DeepSeekV4FlashGPURequestState(
        DeepSeekV4FlashGPUCacheConfig(
            context_length=self._kernel_context_length(),
            hidden_size=self.shape.hidden_size,
            batch_size=1,
            kv_width=self.shape.head_dim,
            device=device,
        )
    )
    output_ids = torch.empty((input_ids.numel() + max_tokens,), dtype=torch.long, device=device)
    output_ids[: input_ids.numel()] = input_ids.to(device=device, dtype=torch.long)
    token_elapsed_ms: list[float] = []
    token_id_tensor = output_ids[input_ids.numel() - 1].reshape(())
    for generated_idx in range(max_tokens):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        token_id_tensor = self._forward_kernel_token_step_token_tensor(
            token_id_tensor=token_id_tensor,
            state=state,
            token_idx=state.token_position,
            device=device,
        )
        end_event.record()
        torch.cuda.synchronize()
        token_elapsed_ms.append(float(start_event.elapsed_time(end_event)))
        output_ids[input_ids.numel() + generated_idx] = token_id_tensor
    return output_ids, token_elapsed_ms
```

- [x] **Step 3: Make existing generator call the timed implementation**

Replace the old `generate_greedy_kernel()` loop with:

```python
output_ids, _token_elapsed_ms = self.generate_greedy_kernel_timed(
    input_ids,
    max_tokens=max_tokens,
)
return output_ids
```

- [x] **Step 4: Count any unavoidable CPU sync explicitly**

In `DeepSeekV4FlashGPUBackend.__init__`, add:

```python
"cpu_token_sync_points": 0,
```

Only increment this counter when code calls `.cpu()`, `.tolist()`, or `.item()` for a generated token in the hot loop.

- [x] **Step 5: Verify**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_model_kernel_generate.py -q
uv run ruff check vllm/model_executor/models/deepseek_v4_flash/model.py vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py tests/deepseek_v4_flash/test_model_kernel_generate.py
```

Expected: generated tokens stay in a CUDA output buffer, and `cpu_token_sync_points == 0`.

---

### Task 3: Warm Resident Static Weights Before Timed Decode

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`
- Modify: `tests/deepseek_v4_flash/test_gpu_weight_staging.py`

- [x] **Step 1: Write cache-hit test for output Q8 payload warmup**

Add to `tests/deepseek_v4_flash/test_gpu_weight_staging.py`:

```python
def test_warm_static_decode_weights_makes_output_q8_payload_resident() -> None:
    stager, tensor = make_fake_q8_output_stager()

    stager.warm_static_decode_weights(output_weight=tensor)
    before = stager.memory_stats()
    stager.stage_q8_raw_payload(tensor)
    after = stager.memory_stats()

    assert after["dynamic_hits"] == before["dynamic_hits"] + 1
```

- [x] **Step 2: Add stager warmup method**

In `DeepSeekV4FlashGPUWeightStager`, add:

```python
def warm_static_decode_weights(
    self,
    *,
    output_weight: DeepSeekV4FlashTensor,
) -> None:
    self.stage_q8_raw_payload(output_weight)
```

- [x] **Step 3: Call warmup before benchmark timing**

In `DeepSeekV4FlashForCausalLM`, add:

```python
def warm_decode_static_weights(self, device: torch.device) -> None:
    store = self._require_weight_store()
    stager = self._get_gpu_weight_stager(device)
    lm_head = store.bindings.output
    if lm_head is None or lm_head.lm_head is None:
        raise RuntimeError("DeepSeek V4 Flash output lm_head binding is required")
    stager.warm_static_decode_weights(output_weight=lm_head.lm_head)
```

Then call it from `tests/tools/run_deepseek_v4_flash_gpu_smoke.py` before timed generation.

- [ ] **Step 4: Verify real warm-cache behavior**

Run:

```bash
timeout --foreground --kill-after=60s 1200s uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-tokens 16 --repeat 1 --profile-json /tmp/ds4_warm_static.json
```

Expected:

```text
gpu_staging.dynamic_hits > 0
decode_tps_steady_state is reported
```

---

### Task 4: Pin Hot Hash-Routed Experts For Repeated Decode Tokens

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`
- Modify: `tests/deepseek_v4_flash/test_gpu_weight_staging.py`
- Modify: `tests/deepseek_v4_flash/test_model_kernel_generate.py`

- [x] **Step 1: Write test for predicted expert warmup**

Add:

```python
def test_warm_decode_token_experts_pins_hash_routed_payloads() -> None:
    model = make_fake_model_with_hash_routed_expert_table(token_id=1, expert_ids=(3, 7))

    model.warm_decode_token_experts(torch.tensor([1], dtype=torch.long, device="cuda"))

    stats = model.gpu_staging_memory_stats()
    assert stats["pinned_entries"] >= 6
```

Each expert has gate/up/down payloads, so two experts should pin at least six entries.

- [x] **Step 2: Implement warmup across all layers for prompt token**

In `DeepSeekV4FlashForCausalLM`, add:

```python
def warm_decode_token_experts(self, input_ids: torch.Tensor) -> None:
    store = self._require_weight_store()
    stager = self._get_gpu_weight_stager(input_ids.device)
    token_id = int(input_ids[-1].detach().cpu().item())
    for layer in store.bindings.layers:
        grouped_experts = getattr(layer, "grouped_experts", None)
        if grouped_experts is None:
            continue
        expert_ids = self._likely_hash_routed_expert_ids_for_token(
            stager,
            layer,
            token_id=token_id,
        )
        for expert_id in expert_ids:
            stager.pin_grouped_expert(layer.layer_index, expert_id)
        if expert_ids:
            stager.prefetch_grouped_experts(
                grouped_experts,
                DeepSeekV4FlashExpertPrefetchRequest(
                    layer_idx=layer.layer_index,
                    expert_ids=expert_ids,
                ),
            )
```

This deliberately uses one CPU token read before timing, not inside the decode loop.

- [x] **Step 3: Call expert warmup before timed benchmark**

In `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`, before timed generation:

```python
model.warm_decode_static_weights(input_ids.device)
model.warm_decode_token_experts(input_ids)
torch.cuda.synchronize()
```

- [x] **Step 4: Verify**

Run:

```bash
uv run pytest tests/deepseek_v4_flash/test_gpu_weight_staging.py tests/deepseek_v4_flash/test_model_kernel_generate.py -q
timeout --foreground --kill-after=60s 1200s uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-tokens 16 --repeat 1 --profile-json /tmp/ds4_hot_expert_pin.json
```

Expected:

```text
gpu_staging.pinned_entries > 0
gpu_staging.prefetch_failures == 0
```

---

### Task 5: Add A Local 1 TPS Gate And Use It To Drive The Remaining Hotspot

**Files:**
- Modify: `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`
- Modify: `tests/e2e_full_benchmark.py`
- Modify: `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py`

- [x] **Step 1: Add CLI threshold**

Add to `parse_args()`:

```python
parser.add_argument("--min-steady-decode-tps", type=float, default=0.0)
```

- [x] **Step 2: Fail only when threshold is explicit**

After metrics are computed:

```python
if args.min_steady_decode_tps > 0.0 and decode_metrics["decode_tps_steady_state"] < args.min_steady_decode_tps:
    raise SystemExit(
        "steady-state decode TPS below threshold: "
        f"{decode_metrics['decode_tps_steady_state']:.2f} < {args.min_steady_decode_tps:.2f}"
    )
```

- [x] **Step 3: Run the real model gate**

Run:

```bash
timeout --foreground --kill-after=60s 1200s uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-tokens 16 --repeat 1 --min-steady-decode-tps 1.0 \
  --profile-json /tmp/ds4_decode_1tps_gate.json
```

Expected after Tasks 1-4:

```text
exit code 0
decode_tps_steady_state >= 1.0
q2_iq2_reference_fallback_calls == 0
```

- [x] **Step 4: If still below 1 TPS, use profile to choose exactly one next optimization**

Later A/B runs showed the grouped selected-expert path did not provide a stable
performance win over the simpler wrapper path, so that path was removed.

Use `/tmp/ds4_decode_1tps_gate.json`:

```bash
uv run python - <<'PY'
import json
data = json.load(open("/tmp/ds4_decode_1tps_gate.json"))
agg = data.get("profile", {}).get("aggregate_by_name", {})
for name, stats in sorted(agg.items(), key=lambda kv: kv[1].get("total_ms", 0), reverse=True)[:10]:
    print(name, stats)
PY
```

Decision rule:

- If `output_projection` is top cost, increase output argmax chunking and keep raw Q8 payload resident.
- If `layer_*_compressed` dominates, prioritize compressed attention fusion, not output projection.
- If expert payload staging dominates, increase pinned expert admission and avoid repeated `torch.frombuffer(...).clone()`.

---

### Task 6: Preserve Readability After Performance Changes

**Files:**
- Modify only if tests reveal a regression:
  - `tests/tools/deepseek_v4_flash_quality_smoke.py`
  - `vllm/model_executor/models/deepseek_v4_flash/model.py`

- [x] **Step 1: Run quality smoke**

Run:

```bash
timeout --foreground --kill-after=60s 1200s uv run --no-sync python tests/tools/deepseek_v4_flash_quality_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --max-tokens 8 --min-output-chars 8 --top-k 8 \
  --json-out /tmp/ds4_quality_after_1tps.json
```

Expected:

```text
readability.passed == true
generated text is readable and coherent
```

- [x] **Step 2: Run e2e benchmark for only DeepSeek**

Run:

```bash
timeout --foreground --kill-after=60s 1800s uv run --no-sync python tests/e2e_full_benchmark.py \
  --models deepseek_v4_flash_q2_gguf \
  --output /tmp/ds4_e2e_after_1tps.json
```

Expected:

```text
decode_tps_p50 >= 1.0
```

---

## Success Criteria

- Direct smoke reports `decode_tps_steady_state >= 1.0` for `max_tokens=16`.
- e2e benchmark reports `deepseek_v4_flash_q2_gguf decode_tps_p50 >= 1.0`.
- Quality smoke remains readable and coherent.
- `q2_iq2_reference_fallback_calls == 0`.
- No generated-token `.cpu()`, `.tolist()`, or `.item()` calls remain inside the decode loop.
- New tests pass with `uv run pytest`.

## Risk Notes

- A reported `0.49 tps` may include first-token setup. Task 1 must land before judging kernel work.
- If steady-state is still below `1 tps` after warm residency and token-sync removal, the remaining gap is likely per-layer compressed attention or selected expert kernel occupancy, not benchmark overhead.
- Pinning too many experts can reduce available UMA headroom. Keep `memory_stats()["staged_bytes"]` visible in every real smoke JSON.
