# DeepSeek V4 Flash GPU Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move DeepSeek V4 Flash from CPU reference execution toward a GPU-backed serving path while preserving reference correctness, REST compatibility, and regression coverage.

**Architecture:** Keep `api_server.py`, `AsyncLLM`, `LiteEngine`, and the executor layer as control-plane plumbing. Keep `deepseek_v4_flash/model.py`, `block.py`, `compressed_kv.py`, and `weight_store.py` as model-semantic and reference layers, but split out anything that is hot-path math so it can later dispatch to GPU kernels instead of dense Python/Torch reference code. Add kernel-facing boundaries in `vllm/kernels/triton/` and preserve reference implementations for tests and numerical alignment.

**Tech Stack:** Python 3.12, `uv`, PyTorch, Triton/ROCm, FastAPI, pytest.

---

### Task 1: Keep REST entrypoints thin and remove production direct-reference branching

**Files:**
- Modify: `vllm/entrypoints/openai/api_server.py`
- Modify: `tests/smoke/test_deepseek_v4_flash_http_smoke.py`
- Modify: `tests/test_openai_api_server.py` or the nearest existing OpenAI server test file

- [x] **Step 1: Write the failing test**

Add a REST test that asserts the OpenAI chat route still works through the normal `AsyncLLM.generate()` path and does not require any direct-reference helper on the engine. Cover one non-stream request and one stream request.

- [x] **Step 2: Run the test to verify current behavior**

Run: `uv run pytest tests/smoke/test_deepseek_v4_flash_http_smoke.py -q`

Expected: the test should fail if the route still depends on direct-reference-only behavior.

- [x] **Step 3: Implement the minimal change**

Keep request parsing, structured output parsing, prompt assembly, and HTTP response shaping in `api_server.py`. Remove any production branching that prefers `generate_greedy_reference_chat()` when the normal engine path is available. If a compatibility hook remains, confine it to tests or explicit debug-only code paths.

- [x] **Step 4: Run the REST tests again**

Run:
`timeout 120s uv run --no-sync pytest tests/smoke/test_deepseek_v4_flash_http_smoke.py -q`

Expected: both pass, with the normal engine path still handling chat completions.

- [x] **Step 5: Commit**

```bash
git add vllm/entrypoints/openai/api_server.py tests/smoke/test_deepseek_v4_flash_http_smoke.py tests/test_openai_api_server.py
git commit -m "refactor: keep openai server on normal engine path"
```

### Task 2: Keep AsyncLLM as a request bridge only

**Files:**
- Modify: `vllm/engine/async_llm.py`
- Modify: `tests/deepseek_v4_flash/test_async_llm_direct_reference.py`
- Modify: `tests/test_async_llm.py` or the nearest existing AsyncLLM test file

- [x] **Step 1: Write the failing test**

Add a test that proves `AsyncLLM.generate()` remains the public bridge for request submission and streaming, while any direct-reference helper is either absent or isolated behind a test-only path.

- [x] **Step 2: Run the test to verify current behavior**

Run: `timeout 120s uv run --no-sync pytest tests/deepseek_v4_flash/test_async_llm_direct_reference.py -q`

Expected: the test should fail if the production API still exposes direct-reference serving as a first-class path.

- [x] **Step 3: Implement the minimal change**

Keep `AsyncLLM` responsible for request submission, abort, stats, and stream forwarding. If `generate_greedy_reference_chat()` is still present, move it toward a test helper or guard it so it is not part of the normal serving contract.

- [x] **Step 4: Run the AsyncLLM tests again**

Run: `timeout 120s uv run --no-sync pytest tests/deepseek_v4_flash/test_async_llm_direct_reference.py -q`

Expected: pass, with the serving contract remaining focused on `generate()` and `abort()`.

- [x] **Step 5: Commit**

```bash
git add vllm/engine/async_llm.py tests/deepseek_v4_flash/test_async_llm_direct_reference.py tests/test_async_llm.py
git commit -m "refactor: keep async llm as bridge only"
```

### Task 3: Keep engine and executor layers thin and explicit

**Files:**
- Modify: `vllm/engine/lite_engine.py`
- Modify: `vllm/engine/prefill_executor.py`
- Modify: `vllm/engine/decode_executor.py`
- Modify: `tests/test_step_scheduler.py`
- Modify: `tests/test_runtime_observer.py`

- [x] **Step 1: Write the failing test**

Add or extend a test that asserts `LiteEngine` still owns scheduling, KV planning, and model assembly, while `PrefillExecutor` and `DecodeExecutor` only batch inputs and call `model(...)`.

- [x] **Step 2: Run the scheduling tests**

Run:
`timeout 120s uv run --no-sync pytest tests/test_engine_executor_contracts.py tests/test_step_scheduler.py tests/test_runtime_observer.py -q`

Expected: one or more assertions should fail if the engine layer has started to absorb model-specific math.

- [x] **Step 3: Implement the minimal change**

Keep `LiteEngine` as the control-plane owner for runtime policy, request scheduling, KV cache planning, and backend selection. Keep the executors as thin batch builders that prepare `attn_metadata`, `lora_mapping`, and multimodal inputs, then invoke the model object.

- [x] **Step 4: Run the engine tests again**

Run:
`timeout 120s uv run --no-sync pytest tests/test_engine_executor_contracts.py tests/test_step_scheduler.py tests/test_runtime_observer.py -q`

Expected: pass.

- [x] **Step 5: Commit**

```bash
git add vllm/engine/lite_engine.py vllm/engine/prefill_executor.py vllm/engine/decode_executor.py tests/test_step_scheduler.py tests/test_runtime_observer.py
git commit -m "refactor: keep engine executors as thin orchestration"
```

### Task 4: Split DeepSeek reference math from future kernel dispatch

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/block.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/compressed_kv.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/weight_store.py`
- Modify: `tests/deepseek_v4_flash/test_block_reference.py`
- Modify: `tests/deepseek_v4_flash/test_direct_decode_real_smoke.py`

- [x] **Step 1: Write the failing test**

Add a test that distinguishes reference-only helpers from future dispatch points. The reference path must still return finite logits and append one greedy token, but the public model entrypoint should be shaped so it can later switch to kernel-backed implementations without changing the external API.

- [x] **Step 2: Run the DeepSeek reference tests**

Run:
`timeout 180s uv run --no-sync pytest tests/deepseek_v4_flash/test_block_reference.py tests/deepseek_v4_flash/test_model_dispatch_boundary.py -q`

Expected: the test should expose any accidental coupling between model semantics and one-off direct-reference helpers.

- [x] **Step 3: Implement the minimal change**

Keep `forward_full_reference()` and the layer runners as reference implementations. Add or clarify dispatch boundaries in `model.py` so the model entrypoint can later call GPU-backed attention, MoE, KV cache, and output-collapse kernels. Keep `weight_store.py` as load/decode infrastructure and `compressed_kv.py` as cache-state semantics.

- [x] **Step 4: Run the DeepSeek reference tests again**

Run:
`timeout 180s uv run --no-sync pytest tests/deepseek_v4_flash/test_block_reference.py tests/deepseek_v4_flash/test_model_dispatch_boundary.py -q`

Expected: pass, with reference behavior unchanged.

- [x] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/model.py vllm/model_executor/models/deepseek_v4_flash/block.py vllm/model_executor/models/deepseek_v4_flash/compressed_kv.py vllm/model_executor/models/deepseek_v4_flash/weight_store.py tests/deepseek_v4_flash/test_block_reference.py tests/deepseek_v4_flash/test_direct_decode_real_smoke.py
git commit -m "refactor: separate deepseek reference math from dispatch"
```

### Task 5: Add GPU kernel-facing scaffolding for DeepSeek hot-path math

**Files:**
- Create or modify: `vllm/kernels/triton/deepseek_v4_flash/__init__.py`
- Create or modify: `vllm/kernels/triton/deepseek_v4_flash/attention.py`
- Create or modify: `vllm/kernels/triton/deepseek_v4_flash/moe.py`
- Create or modify: `vllm/kernels/triton/deepseek_v4_flash/cache.py`
- Create or modify: `vllm/kernels/triton/deepseek_v4_flash/output.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/block.py`

- [x] **Step 1: Write the failing test**

Add a small interface test that imports the new kernel-facing modules and verifies the model can detect the presence of kernel dispatch hooks without changing reference outputs.

- [x] **Step 2: Run the kernel interface test**

Run: `timeout 120s uv run --no-sync pytest tests/deepseek_v4_flash/test_kernel_scaffolding.py tests/deepseek_v4_flash/test_model_dispatch_boundary.py -q`

Expected: fail until the kernel-facing entrypoints exist.

- [x] **Step 3: Implement the minimal change**

Create thin GPU-facing function signatures for attention, compressed/indexer attention, cache updates, MoE routing, and output projection. Keep the initial implementation as adapters or explicit `NotImplementedError` stubs if the kernel is not ready yet, but wire the call sites so the future move from reference math to Triton is mechanical.

- [x] **Step 4: Run the DeepSeek suite again**

Run: `timeout 120s uv run --no-sync pytest tests/deepseek_v4_flash/test_kernel_scaffolding.py tests/deepseek_v4_flash/test_model_dispatch_boundary.py -q`

Expected: pass, while reference code still handles the actual math for now.

- [x] **Step 5: Commit**

```bash
git add vllm/kernels/triton/deepseek_v4_flash vllm/model_executor/models/deepseek_v4_flash/model.py vllm/model_executor/models/deepseek_v4_flash/block.py
git commit -m "feat: add deepseek gpu kernel scaffolding"
```

### Task 6: Re-run the right regressions and update the design docs

**Files:**
- Modify: `docs/design/deepseek_v4_flash_q2_native.md`
- Modify: `docs/superpowers/plans/2026-06-07-deepseek-v4-flash-gpu-migration.md` if scope changes
- Modify: any test file touched by the above tasks only if a regression exposed a real contract gap

- [x] **Step 1: Run the fast regression suite**

Run: `bash tests/run_regression_suite.sh`

Expected: unit/smoke coverage passes without GPU model loads.

- [x] **Step 2: Run the correctness regression**

Run: `bash tests/run_inference_correctness_regression.sh`

Expected: DeepSeek reference coverage still passes, and any existing model-specific regression gates remain green.

- [x] **Step 3: Update the design doc**

Document the current boundary explicitly: control plane in `api_server.py` / `AsyncLLM` / `LiteEngine`, reference semantics in `deepseek_v4_flash/`, and GPU execution surfaces in `vllm/kernels/triton/`.

- [x] **Step 4: Commit**

```bash
git add docs/design/deepseek_v4_flash_q2_native.md docs/superpowers/plans/2026-06-07-deepseek-v4-flash-gpu-migration.md
git commit -m "docs: clarify deepseek gpu migration boundaries"
```


## Execution Notes

- Task 1 committed as `c0a524d0c refactor: keep openai chat on normal engine path`.
- Task 2 committed as `729c678d3 refactor: keep async llm as bridge only`.
- Task 3 committed as `746b1f1f9 test: cover engine executor thin contracts`.
- Task 4 committed as `392b32f7f refactor: separate deepseek reference dispatch boundary`.
- Task 5 committed as `e7c2ff5d3 feat: add deepseek gpu kernel scaffolding`.
- Task 6 updates this plan and the DeepSeek design document with the resulting boundaries and validation results.
- `bash tests/run_regression_suite.sh` passed with `123 passed, 2 skipped`.
- `bash tests/run_inference_correctness_regression.sh` completed all requested correctness regression stages successfully.
