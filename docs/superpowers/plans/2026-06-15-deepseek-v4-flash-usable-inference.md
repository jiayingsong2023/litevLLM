# DeepSeek V4 Flash Usable Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move DeepSeek V4 Flash batch=1 greedy REST inference from correctness-first GPU smoke toward locally usable decode by reducing staging churn, CPU synchronization, launch overhead, and unnecessary full-logits materialization.

**Architecture:** Keep `vllm/engine/` and REST model-agnostic. Put performance policy in `vllm/model_executor/models/deepseek_v4_flash/`, low-level kernels in `vllm/kernels/triton/deepseek_v4_flash/`, and expose metrics through the existing smoke/profile helpers. Each task must preserve the existing reference path and must not silently fall back from the production GPU path to CPU decode.

**Tech Stack:** Python 3.12, PyTorch CUDA/ROCm API, Triton through `vllm.triton_utils`, pytest, ruff, GGUF target `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.

---

## File Map

- `vllm/model_executor/models/deepseek_v4_flash/expert_cache.py`: hot expert policy, LRU admission policy, and small value objects shared by model/stager.
- `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`: GPU resident cache, stream-only admission, pinned expert handling, prefetch counters, and memory accounting.
- `vllm/model_executor/models/deepseek_v4_flash/model.py`: request warmup, stager construction, per-layer prefetch scheduling, and greedy decode control flow.
- `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`: routed expert execution and removal of avoidable CPU tensor syncs.
- `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`: quantized expert and output projection/argmax dispatch.
- `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`: IQ2/Q2 fused or grouped expert kernels.
- `vllm/kernels/triton/deepseek_v4_flash/output.py`: output projection and greedy argmax kernels.
- `tests/deepseek_v4_flash/`: focused unit tests and GPU contract tests.
- `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`: real-smoke metrics.
- `docs/design/deepseek_v4_flash_q2_native.md`: updated measured performance notes after each verified performance phase.

---

### Task 1: LRU Admission And Hot Expert Residency

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/expert_cache.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`
- Test: `tests/deepseek_v4_flash/test_expert_cache_policy.py`
- Test: `tests/deepseek_v4_flash/test_gpu_weight_staging.py`

- [ ] **Step 1: Write failing tests for stream-only admission**

Add tests that construct a stager with a small budget and a policy that refuses one expert. Verify refused grouped payloads are transferred as temporary CUDA tensors, do not increase `staged_bytes`, increment `streamed_bytes`, and do not count as resident cache misses.

- [ ] **Step 2: Write failing tests for pinned hot experts**

Add tests that pin `(layer_idx, expert_id)` entries and verify their gate/up/down raw payload cache keys are counted under `pinned_entries`, are not evicted when later dynamic entries exceed budget, and remain hit-able by repeated staging.

- [ ] **Step 3: Implement admission policy**

Add a small immutable policy type in `expert_cache.py` with:

```python
@dataclass(frozen=True)
class DeepSeekV4FlashCacheAdmissionPolicy:
    min_reuse_score: int = 1
    stream_experts: frozenset[tuple[int, int]] = frozenset()

    def should_cache_grouped_expert(self, *, layer_idx: int | None, expert_id: int) -> bool:
        if layer_idx is not None and (layer_idx, expert_id) in self.stream_experts:
            return False
        return self.min_reuse_score <= 1
```

Wire it into `DeepSeekV4FlashGPUWeightStager` and apply it before inserting grouped expert matrices or raw payloads.

- [ ] **Step 4: Preserve correctness and counters**

Make stream-only paths still return CUDA tensors and record `streamed_bytes`. Do not increment `loaded_bytes` for entries that were never admitted into resident cache. Pinned entries must be immune to `_prepare_cache_insert()` eviction.

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_expert_cache_policy.py tests/deepseek_v4_flash/test_gpu_weight_staging.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash/expert_cache.py vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py tests/deepseek_v4_flash/test_expert_cache_policy.py tests/deepseek_v4_flash/test_gpu_weight_staging.py
git add vllm/model_executor/models/deepseek_v4_flash/expert_cache.py vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py tests/deepseek_v4_flash/test_expert_cache_policy.py tests/deepseek_v4_flash/test_gpu_weight_staging.py
git commit -m "feat: add deepseek expert cache admission"
```

Expected: tests pass; no ruff errors.

---

### Task 2: Model-Level Warm Residency Preparation

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Modify: `tests/deepseek_v4_flash/test_model_kernel_generate.py`
- Modify: `tests/deepseek_v4_flash/test_gpu_weight_staging.py`

- [ ] **Step 1: Write failing tests for hot expert preparation**

Add a unit test that creates a model with a fake stager and calls `prepare_for_serving()`. Verify the model can request a bounded hot expert preparation pass without changing context length, and repeated calls reuse the same stager.

- [ ] **Step 2: Add warmup entry point**

Add a model method:

```python
def prepare_deepseek_hot_experts(
    self,
    *,
    device: torch.device,
    max_layers: int = 4,
    experts_per_layer: int = 2,
) -> dict[str, int | None]:
    ...
```

It should stage only raw grouped expert payloads when available, bounded by `max_layers * experts_per_layer`, and return `gpu_staging_memory_stats()`.

- [ ] **Step 3: Keep warmup bounded**

Do not load all 43 layers or all routed experts. Use semantic bindings already available from the weight store, choose the lowest expert ids unless a hot policy has explicit pinned experts, and skip missing tensors.

- [ ] **Step 4: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_model_kernel_generate.py tests/deepseek_v4_flash/test_gpu_weight_staging.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash/model.py tests/deepseek_v4_flash/test_model_kernel_generate.py tests/deepseek_v4_flash/test_gpu_weight_staging.py
git add vllm/model_executor/models/deepseek_v4_flash/model.py tests/deepseek_v4_flash/test_model_kernel_generate.py tests/deepseek_v4_flash/test_gpu_weight_staging.py
git commit -m "feat: warm deepseek hot expert residency"
```

Expected: tests pass; no full-model load required.

---

### Task 3: Effective Next-Layer Prefetch Scheduling

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
- Test: `tests/deepseek_v4_flash/test_expert_prefetch.py`
- Test: `tests/deepseek_v4_flash/test_model_kernel_generate.py`

- [ ] **Step 1: Write failing test for real prefetch hit accounting**

Add a test where token step `N` schedules prefetch for layer `L+1`; the subsequent layer staging sees resident raw payloads and increments `prefetch_hits`.

- [ ] **Step 2: Move prefetch before demand staging**

In the per-layer decode loop, schedule best-effort prefetch for the next layer's likely routed experts before entering the next layer demand stage. Use a separate CUDA stream if available, and avoid synchronizing unless the demand path needs the exact tensor.

- [ ] **Step 3: Avoid prefetching uncachable entries**

Call the admission policy before prefetch staging. Stream-only experts should increment neither prefetch misses nor resident entries.

- [ ] **Step 4: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_expert_prefetch.py tests/deepseek_v4_flash/test_model_kernel_generate.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash/model.py vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_expert_prefetch.py tests/deepseek_v4_flash/test_model_kernel_generate.py
git add vllm/model_executor/models/deepseek_v4_flash/model.py vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_expert_prefetch.py tests/deepseek_v4_flash/test_model_kernel_generate.py
git commit -m "feat: schedule deepseek next layer expert prefetch"
```

Expected: prefetch counters become meaningful in unit tests.

---

### Task 4: Remove Avoidable CPU Synchronization In Routed Expert Selection

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
- Modify: `tests/deepseek_v4_flash/test_gpu_layer_forward.py`
- Modify: `tests/deepseek_v4_flash/test_model_kernel_generate.py`

- [ ] **Step 1: Write failing test for tensor expert-id path**

Add a fake backend/stager test proving `_run_staged_routed_experts()` can consume a CUDA `expert_ids` tensor without calling `.cpu().tolist()` when a bounded top-k path is available.

- [ ] **Step 2: Add bounded tensor iteration helper**

For `batch=1`, `top_k` is tiny. Use a helper that accepts either an explicit Python list from hash routing or a CUDA tensor plus `max_experts`; only convert exactly the small top-k tensor when needed by Python staging. Record a profiler counter when conversion happens.

- [ ] **Step 3: Prefer hash table on GPU**

Keep `expert_token_to_expert_ids` cached via stager instead of re-reading it through `store.tensor_to_torch(...).to(cuda)` every layer/token.

- [ ] **Step 4: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_layer_forward.py tests/deepseek_v4_flash/test_model_kernel_generate.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_gpu_layer_forward.py tests/deepseek_v4_flash/test_model_kernel_generate.py
git add vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_gpu_layer_forward.py tests/deepseek_v4_flash/test_model_kernel_generate.py
git commit -m "perf: reduce deepseek routed expert cpu syncs"
```

Expected: existing routed expert behavior unchanged; new counters expose remaining syncs.

---

### Task 5: Greedy Output Argmax Fast Path

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Modify: `vllm/kernels/triton/deepseek_v4_flash/output.py`
- Test: `tests/deepseek_v4_flash/test_gpu_output_path.py`
- Test: `tests/deepseek_v4_flash/test_model_kernel_generate.py`

- [ ] **Step 1: Write failing test that greedy decode does not request full logits**

Extend the existing fake backend tests so `generate_greedy_kernel()` raises if `output_logits()` is called when `output_argmax()` is available.

- [ ] **Step 2: Ensure output argmax returns only token id**

Keep `deepseek_v4_output_argmax()` as the production greedy path and make model code pass row offsets/chunk information without materializing `[1, vocab]` logits.

- [ ] **Step 3: Add output cache metrics**

Expose counters for output chunk resident hits/misses through `gpu_staging_memory_stats()` or backend stats so the smoke tool can report whether first-token output projection is cold or warm.

- [ ] **Step 4: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_output_path.py tests/deepseek_v4_flash/test_model_kernel_generate.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash/model.py vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py vllm/kernels/triton/deepseek_v4_flash/output.py tests/deepseek_v4_flash/test_gpu_output_path.py tests/deepseek_v4_flash/test_model_kernel_generate.py
git add vllm/model_executor/models/deepseek_v4_flash/model.py vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py vllm/kernels/triton/deepseek_v4_flash/output.py tests/deepseek_v4_flash/test_gpu_output_path.py tests/deepseek_v4_flash/test_model_kernel_generate.py
git commit -m "perf: use deepseek greedy output argmax path"
```

Expected: greedy path avoids full logits where possible.

---

### Task 6: Single-Launch IQ2 Gate/Up/SILU Prototype

**Files:**
- Modify: `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Test: `tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py`
- Test: `tests/deepseek_v4_flash/test_gpu_backend_contract.py`

- [ ] **Step 1: Write failing correctness test**

Add a CUDA test that compares a new fused `IQ2 gate/up/SILU*multiply` helper against the existing two-matvec reference for small deterministic payloads.

- [ ] **Step 2: Implement fused helper**

Add a Triton helper that dequantizes gate and up payloads, computes both dot products in one launch per row, applies `silu(gate) * up`, and returns the intermediate vector consumed by down projection.

- [ ] **Step 3: Route backend through fused helper**

When gate/up are both `IQ2_XXS` with matching shape, call the fused helper and increment a new `iq2_xxs_gate_up_fused_calls` stat. Keep the old two-kernel helper for fallback tests.

- [ ] **Step 4: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py tests/deepseek_v4_flash/test_gpu_backend_contract.py -q
uv run --no-sync ruff check vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py tests/deepseek_v4_flash/test_gpu_backend_contract.py
git add vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py tests/deepseek_v4_flash/test_gpu_backend_contract.py
git commit -m "perf(kernels): fuse deepseek iq2 gate up activation"
```

Expected: fused path matches reference tolerance and stats prove it is used.

---

### Task 7: Real-Smoke Performance Gate And Documentation

**Files:**
- Modify: `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`
- Modify: `docs/design/deepseek_v4_flash_q2_native.md`

- [ ] **Step 1: Add smoke metrics**

Extend `phase3_metrics`/`phase4_metrics` or add `usable_inference_metrics` with `pinned_entries`, `streamed_bytes`, `lru_evictions`, `prefetch_hits`, `prefetch_misses`, `output_cache_hits`, `output_cache_misses`, and fused IQ2 call counts.

- [ ] **Step 2: Run bounded validation**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_real_smoke_tool_cli.py -q
timeout --foreground --kill-after=60s 2400s uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --context-length 4096 --max-tokens 1 --repeat 1 --profile-json /tmp/deepseek-v4-flash-usable-one.json
timeout --foreground --kill-after=60s 7200s uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --context-length 4096 --max-tokens 8 --repeat 1 --profile-json /tmp/deepseek-v4-flash-usable-max8.json
```

Expected: output token ids remain finite and no Q2/IQ2 reference fallback is recorded. Performance should improve or the profile must identify the remaining blocker.

- [ ] **Step 3: Update docs**

Record measured metrics, compare against Phase 4, and explicitly state whether the current branch reached the local usability target.

- [ ] **Step 4: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_real_smoke_tool_cli.py tests/deepseek_v4_flash/test_gpu_backend_contract.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py -q
uv run --no-sync ruff check tests/tools/run_deepseek_v4_flash_gpu_smoke.py
git add tests/tools/run_deepseek_v4_flash_gpu_smoke.py docs/design/deepseek_v4_flash_q2_native.md
git commit -m "docs: record deepseek usable inference profile"
```

Expected: smoke tool reports actionable metrics and docs reflect the measured state.

---

## Final Verification

Run:

```bash
bash -n tests/run_inference_correctness_regression.sh
uv run --no-sync pytest tests/deepseek_v4_flash/test_expert_cache_policy.py tests/deepseek_v4_flash/test_gpu_weight_staging.py tests/deepseek_v4_flash/test_expert_prefetch.py tests/deepseek_v4_flash/test_gpu_layer_forward.py tests/deepseek_v4_flash/test_gpu_output_path.py tests/deepseek_v4_flash/test_model_kernel_generate.py tests/deepseek_v4_flash/test_gpu_backend_contract.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py tests/deepseek_v4_flash/test_real_smoke_tool_cli.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash vllm/kernels/triton/deepseek_v4_flash tests/deepseek_v4_flash tests/tools/run_deepseek_v4_flash_gpu_smoke.py
```

For kernel/KV-cache/numerics changes, run the full correctness regression when time permits:

```bash
bash tests/run_inference_correctness_regression.sh
```
