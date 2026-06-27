# DeepSeek V4 Flash GPU Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first production-shaped DeepSeek V4 Flash path where OpenAI-compatible REST can run batch=1 greedy decode through GPU-owned model math instead of the CPU direct-reference path.

**Architecture:** Keep REST, `AsyncLLM`, `LiteEngine`, and executors as control plane. Add a DeepSeek GPU backend under `vllm/model_executor/models/deepseek_v4_flash/` that owns runtime state and dispatches to kernel-facing modules under `vllm/kernels/triton/deepseek_v4_flash/`. CPU reference remains available only as an oracle for tests and explicit reference helpers; production REST must fail explicitly if a required GPU kernel is missing.

**Tech Stack:** Python 3.12, `uv`, PyTorch, Triton/ROCm, FastAPI, pytest.

---

### Task 1: GPU Backend Capability Contract

**Files:**
- Create: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Test: `tests/deepseek_v4_flash/test_gpu_backend_contract.py`

- [ ] **Step 1: Write the failing test**

Add `tests/deepseek_v4_flash/test_gpu_backend_contract.py`:

```python
from __future__ import annotations

import pytest

from vllm.model_executor.models.deepseek_v4_flash.gpu_backend import (
    DeepSeekV4FlashGPUBackend,
    DeepSeekV4FlashGPUCapabilities,
)
from vllm.model_executor.models.deepseek_v4_flash.model import (
    DeepSeekV4FlashForCausalLM,
)


def test_gpu_backend_reports_missing_required_kernels() -> None:
    caps = DeepSeekV4FlashGPUCapabilities(
        q8_linear=True,
        attention=False,
        compressed_attention=False,
        cache_update=False,
        moe=False,
        output=False,
    )
    backend = DeepSeekV4FlashGPUBackend(capabilities=caps)

    assert backend.is_ready is False
    assert backend.missing_kernels == (
        "attention",
        "compressed_attention",
        "cache_update",
        "moe",
        "output",
    )
    with pytest.raises(RuntimeError, match="missing GPU kernels"):
        backend.require_ready()


def test_model_keeps_kernel_execution_disabled_until_backend_ready() -> None:
    model = DeepSeekV4FlashForCausalLM(gpu_backend=DeepSeekV4FlashGPUBackend())

    assert model.kernel_execution_available is False
    with pytest.raises(NotImplementedError, match="kernel execution is not available"):
        model.forward_full_reference_or_kernel_placeholder()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`timeout 120s uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_backend_contract.py -q`

Expected: import failure for `gpu_backend`.

- [ ] **Step 3: Implement minimal backend contract**

Create `gpu_backend.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekV4FlashGPUCapabilities:
    q8_linear: bool = True
    attention: bool = False
    compressed_attention: bool = False
    cache_update: bool = False
    moe: bool = False
    output: bool = False

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, enabled in (
                ("q8_linear", self.q8_linear),
                ("attention", self.attention),
                ("compressed_attention", self.compressed_attention),
                ("cache_update", self.cache_update),
                ("moe", self.moe),
                ("output", self.output),
            )
            if not enabled
        )

    @property
    def is_ready(self) -> bool:
        return not self.missing


class DeepSeekV4FlashGPUBackend:
    def __init__(
        self,
        *,
        capabilities: DeepSeekV4FlashGPUCapabilities | None = None,
    ) -> None:
        self.capabilities = capabilities or DeepSeekV4FlashGPUCapabilities()

    @property
    def is_ready(self) -> bool:
        return self.capabilities.is_ready

    @property
    def missing_kernels(self) -> tuple[str, ...]:
        return self.capabilities.missing

    def require_ready(self) -> None:
        if not self.is_ready:
            missing = ", ".join(self.missing_kernels)
            raise RuntimeError(f"DeepSeek V4 Flash missing GPU kernels: {missing}")
```

Modify `DeepSeekV4FlashForCausalLM.__init__` to accept `gpu_backend` and set `kernel_execution_available = gpu_backend.is_ready`.

- [ ] **Step 4: Run test to verify it passes**

Run:
`timeout 120s uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_backend_contract.py tests/deepseek_v4_flash/test_model_dispatch_boundary.py -q`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py vllm/model_executor/models/deepseek_v4_flash/model.py tests/deepseek_v4_flash/test_gpu_backend_contract.py
git commit -m "feat: add deepseek gpu backend capability contract"
```

### Task 2: GPU Output Path for Q8 Logits

**Files:**
- Modify: `vllm/kernels/triton/deepseek_v4_flash/output.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Test: `tests/deepseek_v4_flash/test_gpu_output_path.py`

- [ ] **Step 1: Write the failing test**

Add a small tensor test that calls a backend output method with `[4, hidden]` streams, synthetic Q8-like unpacked `lm_head_values`, scales, and output HC tensors. Assert it returns finite `[vocab]` logits and matches the existing reference helpers for the same tiny tensors.

- [ ] **Step 2: Run test to verify it fails**

Run:
`timeout 120s uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_output_path.py -q`

Expected: missing backend output method.

- [ ] **Step 3: Implement minimal GPU-shaped output path**

Implement an output backend method that validates device placement and calls the current PyTorch Q8 reference only when tensors are already GPU tensors. This is an interim GPU-owned path: no CPU tensor movement, explicit shape checks, and no REST fallback to CPU reference.

- [ ] **Step 4: Run tests**

Run:
`timeout 120s uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_output_path.py tests/deepseek_v4_flash/test_kernel_scaffolding.py -q`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/kernels/triton/deepseek_v4_flash/output.py vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py tests/deepseek_v4_flash/test_gpu_output_path.py
git commit -m "feat: add deepseek gpu output path"
```

### Task 3: GPU Sliding Layer Execution for Layers 0 and 1

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/block.py`
- Modify: `vllm/kernels/triton/deepseek_v4_flash/attention.py`
- Test: `tests/deepseek_v4_flash/test_gpu_sliding_layer.py`

- [ ] **Step 1: Write failing tests**

Add tests that construct tiny synthetic layer tensors and assert GPU-backed sliding attention runs on `cuda` tensors without calling `forward_full_reference()`. Include a missing-kernel test that fails explicitly.

- [ ] **Step 2: Run test to verify it fails**

Run:
`timeout 120s uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_sliding_layer.py -q`

Expected: missing GPU sliding layer method.

- [ ] **Step 3: Implement minimal GPU sliding layer dispatch**

Implement backend methods for RMSNorm, Q-LoRA projection, RoPE tail, raw SWA attention, and output projection using torch GPU ops first. Keep the API shaped so Triton kernels can replace these methods without changing `model.py`.

- [ ] **Step 4: Run tests**

Run:
`timeout 180s uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_sliding_layer.py tests/deepseek_v4_flash/test_block_reference.py -q`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py vllm/model_executor/models/deepseek_v4_flash/block.py vllm/kernels/triton/deepseek_v4_flash/attention.py tests/deepseek_v4_flash/test_gpu_sliding_layer.py
git commit -m "feat: add deepseek gpu sliding layer path"
```

### Task 4: GPU MoE Dispatch

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Modify: `vllm/kernels/triton/deepseek_v4_flash/moe.py`
- Test: `tests/deepseek_v4_flash/test_gpu_moe_path.py`

- [ ] **Step 1: Write failing tests**

Add tests for top-6 routed MoE on tiny synthetic tensors, with all math on `cuda` when available. Assert no CPU movement happens by checking output device and input tensor devices.

- [ ] **Step 2: Run test to verify it fails**

Run:
`timeout 120s uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_moe_path.py -q`

Expected: missing GPU MoE method.

- [ ] **Step 3: Implement minimal GPU MoE dispatch**

Implement router top-k, grouped selected expert matvec, and shared+routed accumulation with torch GPU ops. Keep quantized Q2/IQ2 dequant behind backend methods so Triton kernels can replace them later.

- [ ] **Step 4: Run tests**

Run:
`timeout 180s uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_moe_path.py tests/deepseek_v4_flash/test_moe_reference.py -q`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py vllm/kernels/triton/deepseek_v4_flash/moe.py tests/deepseek_v4_flash/test_gpu_moe_path.py
git commit -m "feat: add deepseek gpu moe path"
```

### Task 5: GPU Compressed/Indexer Attention Dispatch

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Modify: `vllm/kernels/triton/deepseek_v4_flash/compressed_attention.py`
- Modify: `vllm/kernels/triton/deepseek_v4_flash/cache.py`
- Test: `tests/deepseek_v4_flash/test_gpu_compressed_attention.py`

- [ ] **Step 1: Write failing tests**

Add tiny synthetic ratio-4 and ratio-128 tests that update paged cache state, select compressed rows, and return finite attention output on GPU tensors.

- [ ] **Step 2: Run test to verify it fails**

Run:
`timeout 120s uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_compressed_attention.py -q`

Expected: missing compressed attention backend method.

- [ ] **Step 3: Implement minimal GPU compressed attention**

Implement GPU tensor cache update, ratio emit, indexer score/top-k, selected-row attention, and explicit no-contiguous-full-context checks. Use torch GPU ops first but keep kernel boundary signatures stable.

- [ ] **Step 4: Run tests**

Run:
`timeout 180s uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_compressed_attention.py tests/deepseek_v4_flash/test_compressed_kv.py tests/deepseek_v4_flash/test_compressed_attention_contract.py -q`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py vllm/kernels/triton/deepseek_v4_flash/compressed_attention.py vllm/kernels/triton/deepseek_v4_flash/cache.py tests/deepseek_v4_flash/test_gpu_compressed_attention.py
git commit -m "feat: add deepseek gpu compressed attention path"
```

### Task 6: Wire DeepSeek GPU Forward and Greedy Decode

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/model.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Test: `tests/deepseek_v4_flash/test_gpu_forward_smoke.py`

- [ ] **Step 1: Write failing tests**

Add a test that loads the target GGUF when present, calls `model.forward(..., attn_metadata={"config": ...})` or the production forward entry with one token, and asserts finite logits without invoking `forward_full_reference()`. Add a monkeypatch that makes `forward_full_reference()` raise to prove the GPU path is used.

- [ ] **Step 2: Run test to verify it fails**

Run:
`timeout 900s uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_forward_smoke.py -q`

Expected: production forward does not yet route through GPU backend.

- [ ] **Step 3: Implement production forward routing**

Make `DeepSeekV4FlashForCausalLM.forward()` dispatch to the GPU backend when backend capabilities are ready. Keep CPU reference available only through `forward_full_reference()`.

- [ ] **Step 4: Run tests**

Run:
`timeout 900s uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_forward_smoke.py tests/deepseek_v4_flash/test_direct_decode_real_smoke.py -q`

Expected: GPU smoke passes when the target GGUF exists; reference smoke remains available.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/model.py vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py tests/deepseek_v4_flash/test_gpu_forward_smoke.py
git commit -m "feat: wire deepseek gpu forward"
```

### Task 7: Wire LiteEngine and OpenAI REST Smoke

**Files:**
- Modify: `vllm/adapters/deepseek_v4_flash.py`
- Modify: `vllm/engine/lite_engine.py` only if capability metadata is required
- Test: `tests/smoke/test_deepseek_v4_flash_rest_gpu_smoke.py`

- [ ] **Step 1: Write failing REST smoke**

Add a skipped-if-missing-GGUF test that starts the app with an initialized `AsyncLLM` or equivalent test harness, posts to `/v1/chat/completions` with `temperature=0`, `max_tokens=1`, and asserts HTTP 200 with non-empty assistant content. Monkeypatch CPU reference methods to raise so the REST path cannot silently fall back.

- [ ] **Step 2: Run test to verify it fails**

Run:
`timeout 1200s uv run --no-sync pytest tests/smoke/test_deepseek_v4_flash_rest_gpu_smoke.py -q`

Expected: REST path fails until production forward is wired.

- [ ] **Step 3: Implement LiteEngine/REST integration**

Ensure the DeepSeek adapter advertises the GPU backend surface, LiteEngine allocates only the required DeepSeek cache/runtime state, and the standard OpenAI route uses `AsyncLLM.generate()` without direct-reference hooks.

- [ ] **Step 4: Run REST smoke**

Run:
`timeout 1200s uv run --no-sync pytest tests/smoke/test_deepseek_v4_flash_rest_gpu_smoke.py -q`

Expected: pass when the target GGUF exists.

- [ ] **Step 5: Commit**

```bash
git add vllm/adapters/deepseek_v4_flash.py vllm/engine/lite_engine.py tests/smoke/test_deepseek_v4_flash_rest_gpu_smoke.py
git commit -m "feat: enable deepseek gpu rest smoke"
```

### Task 8: Regression and Documentation

**Files:**
- Modify: `docs/design/deepseek_v4_flash_q2_native.md`
- Modify: `docs/superpowers/plans/2026-06-11-deepseek-v4-flash-gpu-serving.md`
- Modify: `tests/run_inference_correctness_regression.sh` only after REST GPU smoke passes and runtime cost is acceptable

- [ ] **Step 1: Run focused DeepSeek tests**

Run:
`timeout 1800s uv run --no-sync pytest tests/deepseek_v4_flash tests/smoke/test_deepseek_v4_flash_rest_gpu_smoke.py -q`

Expected: pass.

- [ ] **Step 2: Run fast regression**

Run:
`bash tests/run_regression_suite.sh`

Expected: pass.

- [ ] **Step 3: Decide whether to add DeepSeek to correctness regression**

Only add DeepSeek V4 Flash to `tests/run_inference_correctness_regression.sh` if the REST GPU smoke is stable and has a bounded timeout. If added, make it opt-in by default with `RUN_DEEPSEEK_V4_FLASH=1` until runtime is acceptable for regular CI.

- [ ] **Step 4: Update docs**

Document what is truly GPU-backed, what remains torch-GPU transitional code, what remains reference-only, and the exact REST smoke command/result.

- [ ] **Step 5: Commit**

```bash
git add docs/design/deepseek_v4_flash_q2_native.md docs/superpowers/plans/2026-06-11-deepseek-v4-flash-gpu-serving.md tests/run_inference_correctness_regression.sh
git commit -m "docs: record deepseek gpu serving status"
```

