# DeepSeek V4 Flash Phase 3 Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move DeepSeek V4 Flash batch=1 greedy decode from a warm correctness path toward an interactive service path by adding hot expert residency, LRU staging, async expert prefetch scaffolding, fused raw Q2/IQ2 matvec interfaces, and fewer CPU synchronization points.

**Architecture:** Keep REST, `AsyncLLM`, `LiteEngine`, and schedulers as control plane. Keep DeepSeek-specific hot-path ownership under `vllm/model_executor/models/deepseek_v4_flash/` and GPU-facing kernels under `vllm/kernels/triton/deepseek_v4_flash/`. Phase 3 does not expand supported modes beyond batch=1 greedy decode; unsupported batching/sampling must continue to fail explicitly.

**Tech Stack:** Python 3.12, `uv`, PyTorch CUDA/ROCm tensors, Triton via `vllm/triton_utils/`, GGUF mmap reader, pytest, JSON profiling output.

---

## Current Baseline

The branch already supports real GGUF batch=1 greedy decode through the model GPU path:

```bash
timeout --foreground --kill-after=60s 7200s \
  uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --context-length 4096 \
  --max-tokens 8 \
  --repeat 1 \
  --profile-json /tmp/deepseek-v4-flash-max8-profile.json
```

Observed Phase 0/1/2 numbers:

- Cold one-token: about `410319 ms`, `0.0024 tok/s`.
- Warm repeat one-token: about `640 ms`, `1.56 tok/s`.
- Cold `max_tokens=8`: about `877174 ms`, `0.0091 tok/s`.
- Staged bytes can reach about `61.3 GiB`; overflow currently falls back to streaming decoded tensors instead of failing.

Remaining Phase 3 bottlenecks:

- `DeepSeekV4FlashGPUWeightStager` stages decoded fp32/fp16 matrices, not raw GGUF Q2/IQ2 payloads.
- Cache residency is “insert until budget, then stream”; there is no LRU eviction or pinned hot expert policy.
- Expert staging happens synchronously exactly when the layer executes.
- Greedy decode still uses `next_token_tensor.item()` between generated tokens.
- Layer routing loops call `.item()` per selected expert, forcing CPU synchronization in the MoE hot path.

## File Responsibilities

- `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`: dynamic/grouped cache ownership, LRU accounting, pinned hot experts, async prefetch API, raw quantized expert payload staging.
- `vllm/model_executor/models/deepseek_v4_flash/expert_cache.py`: small model-local policy helpers for hot expert selection, LRU keys, and prefetch request planning.
- `vllm/model_executor/models/deepseek_v4_flash/weight_store.py`: raw grouped expert payload accessors for GGUF tensors.
- `vllm/model_executor/models/deepseek_v4_flash/quant.py`: CPU reference packing/dequant helpers used by tests.
- `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`: GPU-facing Q2_K/IQ2_XXS matvec interface and fallback implementation boundary.
- `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`: backend methods for quantized expert matvec and token-id carrying.
- `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`: route expert IDs/weights as GPU tensors into backend calls; avoid per-expert `.item()` in the common path.
- `vllm/model_executor/models/deepseek_v4_flash/model.py`: greedy loop without generated-token `.item()`; prefetch planning hooks between layers/tokens.
- `tests/deepseek_v4_flash/`: unit and CUDA tests for cache policy, raw quantized payload staging, fused matvec contracts, and CPU sync reduction.
- `docs/design/deepseek_v4_flash_q2_native.md`: record Phase 3 architecture and measured results.

## Acceptance Target

- The real GGUF smoke remains functional for `batch=1`, `context_length=4096`, `max_tokens=1` and `max_tokens=8`.
- Warm one-token path remains at least as fast as Phase 2 and exposes lower CPU sync count in profiler metadata.
- Staging cache supports deterministic LRU eviction and pinned hot expert keys.
- Expert prefetch API can stage the next layer’s selected experts without changing numerical output.
- Q2_K/IQ2_XXS expert matvec has a GPU-facing backend contract and reference-equivalent tests.
- Unsupported modes still fail clearly; no CPU reference fallback is silently introduced into the service path.

## Task 1: Add LRU and Hot Expert Cache Policy

**Files:**
- Create: `vllm/model_executor/models/deepseek_v4_flash/expert_cache.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`
- Test: `tests/deepseek_v4_flash/test_expert_cache_policy.py`
- Test: `tests/deepseek_v4_flash/test_gpu_weight_staging.py`

- [ ] **Step 1: Add cache policy tests**

Create tests that verify:

- `DeepSeekV4FlashHotExpertPolicy` pins explicit `(layer_idx, expert_id)` pairs.
- LRU eviction removes the oldest non-pinned cache entry when `max_staged_bytes` would be exceeded.
- Pinned entries are not evicted; if all entries are pinned and the budget is exceeded, staging streams the new tensor without caching.
- `memory_stats()` reports `lru_evictions`, `pinned_entries`, and `streamed_bytes`.

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_expert_cache_policy.py tests/deepseek_v4_flash/test_gpu_weight_staging.py -q
```

Expected before implementation: import failure or missing stats fields.

- [ ] **Step 2: Implement policy helpers**

Add:

```python
@dataclass(frozen=True)
class DeepSeekV4FlashCacheKey:
    namespace: str
    name: str
    device: str
    dtype: str
    extra: tuple[int | str, ...] = ()


@dataclass(frozen=True)
class DeepSeekV4FlashHotExpertPolicy:
    pinned_experts: frozenset[tuple[int, int]] = frozenset()

    def is_pinned_expert(self, layer_idx: int, expert_id: int) -> bool:
        return (layer_idx, expert_id) in self.pinned_experts
```

Keep this file policy-only; it must not import torch.

- [ ] **Step 3: Add LRU accounting to the stager**

Update `DeepSeekV4FlashGPUWeightStager` to track cache entry bytes, insertion/use order, and pinned keys. Replace budget overflow errors with:

1. Evict non-pinned least-recently-used entries until enough room exists.
2. If no eviction can make room, stream the tensor to device and increment `streamed_bytes`.
3. Record `lru_evictions`.

Existing stream-on-overflow behavior must remain valid for callers.

- [ ] **Step 4: Commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_expert_cache_policy.py tests/deepseek_v4_flash/test_gpu_weight_staging.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash/expert_cache.py vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py tests/deepseek_v4_flash/test_expert_cache_policy.py tests/deepseek_v4_flash/test_gpu_weight_staging.py
```

Commit:

```bash
git add vllm/model_executor/models/deepseek_v4_flash/expert_cache.py \
  vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py \
  tests/deepseek_v4_flash/test_expert_cache_policy.py \
  tests/deepseek_v4_flash/test_gpu_weight_staging.py
git commit -m "feat: add deepseek expert cache lru policy"
```

## Task 2: Add Raw Q2/IQ2 Expert Payload Staging

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/weight_store.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/quant.py`
- Test: `tests/deepseek_v4_flash/test_quantized_expert_payload.py`

- [ ] **Step 1: Add raw payload tests**

Test that grouped expert tensors can expose raw payload slices with:

- GGUF type name or numeric type.
- logical `rows`, `columns`, `expert_id`.
- raw payload bytes as a CPU `torch.uint8` tensor or `memoryview`.
- CUDA-staged payload reuse by `data_ptr()`.

Include synthetic Q2_K and IQ2_XXS payloads with tiny dimensions where the reference decoder produces deterministic values.

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_quantized_expert_payload.py -q
```

Expected before implementation: missing payload API.

- [ ] **Step 2: Implement payload descriptors**

Add a frozen descriptor:

```python
@dataclass(frozen=True)
class DeepSeekV4FlashQuantizedExpertPayload:
    tensor_name: str
    expert_id: int
    ggml_type: int
    rows: int
    columns: int
    payload: memoryview
```

Add `raw_grouped_expert_payload(tensor, expert_id)` to `DeepSeekV4FlashWeightStore`.

- [ ] **Step 3: Stage raw payloads to CUDA**

Add `stage_grouped_expert_payload(tensor, expert_id)` to the stager. It should cache CUDA `torch.uint8` payload tensors under the grouped cache namespace, with LRU and hot expert pinning support from Task 1.

- [ ] **Step 4: Commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_quantized_expert_payload.py tests/deepseek_v4_flash/test_gpu_weight_staging.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash/weight_store.py vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py vllm/model_executor/models/deepseek_v4_flash/quant.py tests/deepseek_v4_flash/test_quantized_expert_payload.py
```

Commit:

```bash
git add vllm/model_executor/models/deepseek_v4_flash/weight_store.py \
  vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py \
  vllm/model_executor/models/deepseek_v4_flash/quant.py \
  tests/deepseek_v4_flash/test_quantized_expert_payload.py
git commit -m "feat: stage deepseek raw quantized experts"
```

## Task 3: Add Q2/IQ2 Fused Expert Matvec Backend Contract

**Files:**
- Create: `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`
- Modify: `vllm/kernels/triton/deepseek_v4_flash/__init__.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Test: `tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py`
- Test: `tests/deepseek_v4_flash/test_gpu_backend_contract.py`

- [ ] **Step 1: Add fused matvec tests**

Add CUDA tests that compare:

- `deepseek_v4_q2_k_matvec()` against `decode_q2_k_matrix_reference(...).matmul(hidden)`.
- `deepseek_v4_iq2_xxs_matvec()` against `decode_iq2_xxs_matrix_reference(...).matmul(hidden)`.
- Backend `quantized_expert_gemm()` returns the same shape and close values as decoded expert GEMM for a synthetic gate/up/down expert.

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py tests/deepseek_v4_flash/test_gpu_backend_contract.py -q
```

Expected before implementation: missing module/backend method.

- [ ] **Step 2: Implement GPU-facing fallback interface**

Implement Python CUDA-tensor fallback first:

- Accept raw `torch.uint8` payloads already on CUDA.
- Decode on device using vectorized PyTorch operations where possible.
- Return matvec results without materializing full decoded expert matrices in the stager.

The file must include ASCII comments describing raw GGUF memory layout and intended Triton tiling. Direct `import triton` is forbidden.

- [ ] **Step 3: Wire backend method**

Add `DeepSeekV4FlashGPUBackend.quantized_expert_gemm(...)`. It should:

1. Compute gate and up matvec from raw quantized payloads.
2. Apply SiLU gate activation.
3. Compute down matvec from the intermediate vector.
4. Fall back to staged decoded GEMM only when the payload type is unsupported.

- [ ] **Step 4: Commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py tests/deepseek_v4_flash/test_gpu_backend_contract.py -q
uv run --no-sync ruff check vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py vllm/kernels/triton/deepseek_v4_flash/__init__.py vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py tests/deepseek_v4_flash/test_gpu_backend_contract.py
```

Commit:

```bash
git add vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py \
  vllm/kernels/triton/deepseek_v4_flash/__init__.py \
  vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py \
  tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py \
  tests/deepseek_v4_flash/test_gpu_backend_contract.py
git commit -m "feat: add deepseek q2 iq2 expert matvec backend"
```

## Task 4: Route MoE Through Raw Quantized Expert Path

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Test: `tests/deepseek_v4_flash/test_gpu_layer_forward.py`
- Test: `tests/deepseek_v4_flash/test_gpu_compressed_layer_forward.py`

- [ ] **Step 1: Add routing tests**

Add tests that verify `_run_staged_routed_experts()` prefers `quantized_expert_gemm()` when raw payload staging is available, and falls back to decoded staged GEMM when it is not.

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_layer_forward.py tests/deepseek_v4_flash/test_gpu_compressed_layer_forward.py -q
```

Expected before implementation: backend raw quantized path is unused.

- [ ] **Step 2: Implement raw path selection**

Change routed expert execution to request raw gate/up/down payloads from the stager and pass them to `backend.quantized_expert_gemm()`. Keep decoded matrix staging for shared experts and unsupported quant types.

- [ ] **Step 3: Reduce per-expert CPU sync**

Keep `expert_ids` and `expert_weights` as CUDA tensors for backend math. If a CPU expert id is needed only for staging lookup, move exactly the small selected id tensor once per layer to CPU using `expert_ids.detach().cpu().tolist()` and record this in profiler metadata. Do not call `.item()` inside the per-expert loop.

- [ ] **Step 4: Commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_layer_forward.py tests/deepseek_v4_flash/test_gpu_compressed_layer_forward.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py tests/deepseek_v4_flash/test_gpu_layer_forward.py tests/deepseek_v4_flash/test_gpu_compressed_layer_forward.py
```

Commit:

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py \
  vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py \
  tests/deepseek_v4_flash/test_gpu_layer_forward.py \
  tests/deepseek_v4_flash/test_gpu_compressed_layer_forward.py
git commit -m "feat: route deepseek moe through quantized experts"
```

## Task 5: Add Async Expert Prefetch Scaffolding

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/expert_cache.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Test: `tests/deepseek_v4_flash/test_expert_prefetch.py`

- [ ] **Step 1: Add prefetch tests**

Test that:

- A prefetch request contains `layer_idx`, `expert_ids`, and tensor names.
- Prefetch stages raw payloads without changing output values.
- Re-prefetching cached experts records hits, not misses.
- A CUDA stream can be injected for tests; CPU-only tests skip CUDA stream behavior.

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_expert_prefetch.py -q
```

Expected before implementation: missing prefetch API.

- [ ] **Step 2: Implement prefetch API**

Add:

```python
@dataclass(frozen=True)
class DeepSeekV4FlashExpertPrefetchRequest:
    layer_idx: int
    expert_ids: tuple[int, ...]
```

Add stager method:

```python
def prefetch_grouped_experts(
    self,
    tensors: DeepSeekV4FlashGroupedExpertTensors,
    request: DeepSeekV4FlashExpertPrefetchRequest,
    *,
    stream: torch.cuda.Stream | None = None,
) -> None:
    ...
```

Use `with torch.cuda.stream(stream)` when a stream is provided. The implementation should call raw payload staging when available and decoded staging otherwise.

- [ ] **Step 3: Wire model-level next-layer prefetch**

After router top-k is known for layer `L`, create a prefetch request for layer `L + 1` only if the next layer has grouped experts and a prefetch stream exists. Keep this as best-effort: failed prefetch should not change correctness, but should record a profiler counter.

- [ ] **Step 4: Commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_expert_prefetch.py tests/deepseek_v4_flash/test_model_kernel_generate.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash/expert_cache.py vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py vllm/model_executor/models/deepseek_v4_flash/model.py tests/deepseek_v4_flash/test_expert_prefetch.py
```

Commit:

```bash
git add vllm/model_executor/models/deepseek_v4_flash/expert_cache.py \
  vllm/model_executor/models/deepseek_v4_flash/gpu_weight_staging.py \
  vllm/model_executor/models/deepseek_v4_flash/model.py \
  tests/deepseek_v4_flash/test_expert_prefetch.py
git commit -m "feat: prefetch deepseek routed experts"
```

## Task 6: Reduce Greedy Decode CPU Synchronization

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Test: `tests/deepseek_v4_flash/test_model_kernel_generate.py`

- [ ] **Step 1: Add CPU sync guard tests**

Add tests that monkeypatch a fake scalar CUDA tensor whose `.item()` raises inside generated-token carry. Verify `generate_greedy_kernel(max_tokens=2)` can call the second step without using `.item()` on `next_token_tensor`.

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_model_kernel_generate.py -q
```

Expected before implementation: `.item()` is called between generated tokens.

- [ ] **Step 2: Add tensor-token forward entrypoint**

Add `_forward_kernel_token_step_token_tensor(token_id_tensor, ...)` that accepts a scalar CUDA long tensor. It should use tensor-based embedding lookup where possible and only convert to CPU for unsupported debug/reference paths.

- [ ] **Step 3: Update greedy loop**

Change generated-token iteration to pass `next_token_tensor` directly into `_forward_kernel_token_step_token_tensor()` instead of `int(next_token_tensor.item())`.

- [ ] **Step 4: Commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_model_kernel_generate.py tests/deepseek_v4_flash/test_lite_engine_deepseek_gpu.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash/model.py vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py tests/deepseek_v4_flash/test_model_kernel_generate.py
```

Commit:

```bash
git add vllm/model_executor/models/deepseek_v4_flash/model.py \
  vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py \
  tests/deepseek_v4_flash/test_model_kernel_generate.py
git commit -m "feat: keep deepseek greedy token carry on gpu"
```

## Task 7: Real Smoke, Profiling, and Documentation

**Files:**
- Modify: `docs/design/deepseek_v4_flash_q2_native.md`
- Modify: `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`
- Test: `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py`

- [ ] **Step 1: Add profile schema tests**

Extend CLI tests to assert profile JSON includes:

- `lru_evictions`
- `streamed_bytes`
- `prefetch_hits`
- `prefetch_misses`
- `quantized_expert_calls`
- `cpu_sync_points`

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_real_smoke_tool_cli.py -q
```

- [ ] **Step 2: Record Phase 3 measurements**

Run real smoke commands:

```bash
timeout --foreground --kill-after=60s 2400s \
  uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --context-length 4096 \
  --max-tokens 1 \
  --repeat 2 \
  --profile-json /tmp/deepseek-v4-flash-phase3-warm-profile.json
```

```bash
timeout --foreground --kill-after=60s 7200s \
  uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --context-length 4096 \
  --max-tokens 8 \
  --repeat 1 \
  --profile-json /tmp/deepseek-v4-flash-phase3-max8-profile.json
```

- [ ] **Step 3: Update design doc**

Document:

- What Phase 3 implemented.
- Whether the path is still fallback/PyTorch CUDA or true Triton fused for each quant type.
- Real warm/cold timings and cache stats.
- Known remaining bottlenecks for Phase 4.

- [ ] **Step 4: Final verification and commit**

Run:

```bash
bash -n tests/run_inference_correctness_regression.sh
uv run --no-sync pytest tests/deepseek_v4_flash/test_profiler.py tests/deepseek_v4_flash/test_real_smoke_tool_cli.py tests/deepseek_v4_flash/test_gpu_weight_staging.py tests/deepseek_v4_flash/test_model_kernel_generate.py tests/deepseek_v4_flash/test_gpu_backend_contract.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash vllm/kernels/triton/deepseek_v4_flash tests/deepseek_v4_flash tests/tools/run_deepseek_v4_flash_gpu_smoke.py
```

Commit:

```bash
git add docs/design/deepseek_v4_flash_q2_native.md \
  tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  tests/deepseek_v4_flash/test_real_smoke_tool_cli.py
git commit -m "docs: record deepseek phase 3 performance"
```

## Self-Review

- Spec coverage: the plan covers expert async prefetch, raw Q2/IQ2 fused matvec interface, CPU synchronization reduction, LRU staging, hot expert pinning, tests, real smoke, and documentation.
- Placeholder scan: no `TBD`, open-ended `TODO`, or unspecified test instructions are used.
- Type consistency: cache policy, prefetch request, payload descriptor, and backend method names are defined before later tasks reference them.
