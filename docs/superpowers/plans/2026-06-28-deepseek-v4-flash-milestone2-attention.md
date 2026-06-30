# DeepSeek V4 Flash Milestone 2: Fused Sliding-Window Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the PyTorch-reference sliding-window attention in `_run_real_sliding_attention` with a single fused Triton kernel that computes softmax attention over all 64 heads in one launch, reducing kernel launch overhead and intermediate tensors while keeping the reference path as fallback.

**Architecture:** Keep query projection, per-head RMS norm, RoPE, KV cache read, and output projection unchanged. Only the inner `shared_kv_swa_attention_reference(query, kv_rows, attn_sinks)` call is replaced by a fused kernel. The kernel is one program per attention head, uses online softmax to avoid materializing the full score tensor, and supports the optional `attn_sinks` logit with no value contribution.

**Tech Stack:** Python 3.12, PyTorch/ROCm, Triton via `vllm/triton_utils`, `uv`, `pytest`.

## Global Constraints

- Never import `triton` directly; use `vllm.triton_utils`.
- Every new Triton kernel must include ASCII comments describing memory layout and thread/block tiling.
- All changes must pass `uv run ruff check . && uv run ruff format .`.
- All DeepSeek-specific tests and `bash tests/run_regression_suite.sh` must pass.
- Model output quality must pass `tests/run_deepseek_v4_flash_real_smoke.sh` and the DeepSeek path of `tests/run_inference_correctness_regression.sh`.
- Do not change upstream vLLM code paths outside `vllm/model_executor/models/deepseek_v4_flash/` and `vllm/kernels/triton/deepseek_v4_flash/`.
- The fused path must be opt-in with fallback to `shared_kv_swa_attention_reference` so correctness is preserved if the kernel raises.

---

### Task 1: Fused sliding-window attention Triton kernel

**Files:**
- Modify: `vllm/kernels/triton/deepseek_v4_flash/attention.py`
- Test: `tests/deepseek_v4_flash/test_attention_reference.py`

**Interfaces:**
- Consumes: `query [num_heads, head_dim]` fp32 CUDA; `kv_rows [window, head_dim]` fp32 CUDA (or any dtype, cast inside kernel); `attn_sinks [num_heads]` fp32 CUDA optional.
- Produces: `output [num_heads, head_dim]` fp32 CUDA.

- [ ] **Step 1: Add the Triton kernel**

Append to `vllm/kernels/triton/deepseek_v4_flash/attention.py`:

```python
from vllm.triton_utils import tl, triton


@triton.jit
def _deepseek_v4_fused_swa_attention_kernel(
    query_ptr,
    kv_rows_ptr,
    attn_sinks_ptr,
    output_ptr,
    stride_qh,
    stride_qd,
    stride_kvw,
    stride_kvd,
    stride_oh,
    stride_od,
    num_heads,
    head_dim,
    window,
    scale,
    HAS_SINKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused sliding-window attention over shared K=V rows.

    Memory layout:
    - query is [num_heads, head_dim] row-major fp32.
    - kv_rows is [window, head_dim] row-major; loaded as fp32.
    - attn_sinks is [num_heads] fp32; contributes a softmax logit with no value.
    - output is [num_heads, head_dim] fp32.

    Tiling:
    - grid (num_heads,): one program per head.
    - BLOCK_D threads cover the head dimension; masked when head_dim < BLOCK_D.
    - Each program streams over `window` KV rows in a loop, accumulating the
      weighted value using online softmax to keep only O(head_dim) SRAM.
    """
    head_idx = tl.program_id(0)
    if head_idx >= num_heads:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    q_ptrs = query_ptr + head_idx * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    m = float("-inf")
    d = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for w in range(window):
        kv_ptrs = kv_rows_ptr + w * stride_kvw + offs_d * stride_kvd
        kv = tl.load(kv_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        score = tl.sum(q * kv) * scale

        new_m = tl.maximum(m, score)
        exp_old = tl.exp(m - new_m)
        exp_score = tl.exp(score - new_m)
        d = d * exp_old + exp_score
        acc = acc * exp_old + exp_score * kv
        m = new_m

    if HAS_SINKS:
        sink = tl.load(attn_sinks_ptr + head_idx).to(tl.float32)
        new_m = tl.maximum(m, sink)
        d = d * tl.exp(m - new_m) + tl.exp(sink - new_m)
        m = new_m

    out = acc / d
    out_ptrs = output_ptr + head_idx * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, out, mask=mask_d)
```

- [ ] **Step 2: Add the Python wrapper**

Append to the same file:

```python
def deepseek_v4_fused_sliding_window_attention(
    query: torch.Tensor,
    kv_rows: torch.Tensor,
    attn_sinks: torch.Tensor | None,
) -> torch.Tensor:
    """Fused sliding-window attention over shared K=V latent rows.

    Memory layout:
    - query is [num_heads, head_dim] fp32.
    - kv_rows is [window, head_dim] and is used as both keys and values.
    - attn_sinks is [num_heads] optional; each sink contributes one extra
      softmax logit but has no associated value row.
    - output is [num_heads, head_dim] fp32.

    Tiling:
    - grid (num_heads,); one block per head.
    - BLOCK_D is the next power of two >= head_dim.
    """
    if query.ndim != 2:
        raise ValueError(f"query must be 2-D; got {query.ndim}-D")
    if kv_rows.ndim != 2:
        raise ValueError(f"kv_rows must be 2-D; got {kv_rows.ndim}-D")
    if query.shape[1] != kv_rows.shape[1]:
        raise ValueError(
            "query and kv_rows widths must match; "
            f"got {query.shape[1]} and {kv_rows.shape[1]}"
        )
    if not query.is_cuda or not kv_rows.is_cuda:
        raise ValueError("fused sliding attention inputs must be CUDA tensors")
    if attn_sinks is not None:
        if attn_sinks.shape != (query.shape[0],):
            raise ValueError(
                f"attn_sinks shape must be {(query.shape[0],)}; "
                f"got {tuple(attn_sinks.shape)}"
            )
        if not attn_sinks.is_cuda:
            raise ValueError("attn_sinks must be a CUDA tensor")

    num_heads, head_dim = query.shape
    window = kv_rows.shape[0]
    output = torch.empty_like(query)
    scale = 1.0 / math.sqrt(float(head_dim))
    has_sinks = attn_sinks is not None
    sink_ptr = attn_sinks if attn_sinks is not None else torch.empty(
        0, dtype=query.dtype, device=query.device
    )
    block_d = triton.next_power_of_2(head_dim)

    _deepseek_v4_fused_swa_attention_kernel[(num_heads,)](
        query,
        kv_rows,
        sink_ptr,
        output,
        query.stride(0),
        query.stride(1),
        kv_rows.stride(0),
        kv_rows.stride(1),
        output.stride(0),
        output.stride(1),
        num_heads,
        head_dim,
        window,
        scale,
        HAS_SINKS=has_sinks,
        BLOCK_D=block_d,
    )
    return output
```

- [ ] **Step 3: Write the failing test**

Add to `tests/deepseek_v4_flash/test_attention_reference.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_sliding_window_attention_matches_reference() -> None:
    from vllm.kernels.triton.deepseek_v4_flash.attention import (
        deepseek_v4_fused_sliding_window_attention,
    )
    num_heads = 64
    head_dim = 512
    window = 128
    torch.manual_seed(42)
    query = torch.randn((num_heads, head_dim), dtype=torch.float32, device="cuda")
    kv_rows = torch.randn((window, head_dim), dtype=torch.float32, device="cuda")
    attn_sinks = torch.randn((num_heads,), dtype=torch.float32, device="cuda")

    expected = shared_kv_swa_attention_reference(query, kv_rows, attn_sinks)
    actual = deepseek_v4_fused_sliding_window_attention(query, kv_rows, attn_sinks)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
```

- [ ] **Step 4: Run test to verify it fails**

```bash
uv run pytest tests/deepseek_v4_flash/test_attention_reference.py::test_fused_sliding_window_attention_matches_reference -v
```

Expected: FAIL with `ImportError` / `NameError` because the function does not exist yet.

- [ ] **Step 5: Implement and run test**

After adding the kernel and wrapper, run:

```bash
uv run pytest tests/deepseek_v4_flash/test_attention_reference.py::test_fused_sliding_window_attention_matches_reference -v
```

Expected: PASS.

- [ ] **Step 6: Add a no-sinks test**

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_sliding_window_attention_no_sinks() -> None:
    from vllm.kernels.triton.deepseek_v4_flash.attention import (
        deepseek_v4_fused_sliding_window_attention,
    )
    num_heads = 8
    head_dim = 32
    window = 7
    torch.manual_seed(7)
    query = torch.randn((num_heads, head_dim), dtype=torch.float32, device="cuda")
    kv_rows = torch.randn((window, head_dim), dtype=torch.float32, device="cuda")

    expected = shared_kv_swa_attention_reference(query, kv_rows, None)
    actual = deepseek_v4_fused_sliding_window_attention(query, kv_rows, None)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
```

Run and verify PASS.

- [ ] **Step 7: Commit**

```bash
git add vllm/kernels/triton/deepseek_v4_flash/attention.py tests/deepseek_v4_flash/test_attention_reference.py
git commit -m "feat(kernels): fused sliding-window attention for DeepSeek V4 Flash"
```

---

### Task 2: Backend method and layer integration

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`
- Test: `tests/deepseek_v4_flash/test_gpu_backend_contract.py`, `tests/deepseek_v4_flash/test_attention_reference.py`

**Interfaces:**
- Consumes: `DeepSeekV4FlashGPUBackend.fused_sliding_window_attention(query, kv_rows, attn_sinks, token_idx)`.
- Produces: `[num_heads, head_dim]` fp32 tensor.

- [ ] **Step 1: Add the backend method**

In `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`:

1. Add to the import from `vllm.kernels.triton.deepseek_v4_flash.attention`:

```python
from vllm.kernels.triton.deepseek_v4_flash.attention import (
    DeepSeekV4AttentionKernelInputs,
    deepseek_v4_attention,
    deepseek_v4_fused_sliding_window_attention,
)
```

2. Add `"fused_sliding_attention_api_calls": 0` to `self._stats` in `__init__`.

3. Add the method after `sliding_attention`:

```python
def fused_sliding_window_attention(
    self,
    *,
    query: torch.Tensor,
    kv_rows: torch.Tensor,
    attn_sinks: torch.Tensor | None,
    token_idx: int,
) -> torch.Tensor:
    tensors = (query, kv_rows, attn_sinks)
    if any(tensor is not None and not tensor.is_cuda for tensor in tensors):
        raise ValueError(
            "DeepSeek V4 Flash fused sliding attention inputs must be CUDA tensors"
        )
    self._stats["fused_sliding_attention_api_calls"] += 1
    return deepseek_v4_fused_sliding_window_attention(
        query=query.to(torch.float32),
        kv_rows=kv_rows.to(torch.float32),
        attn_sinks=attn_sinks.to(torch.float32) if attn_sinks is not None else None,
    )
```

- [ ] **Step 2: Wire into `_run_real_sliding_attention`**

Modify `vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py`:

1. Add `backend` to the signature:

```python
def _run_real_sliding_attention(
    attn_input: torch.Tensor,
    *,
    layer: DeepSeekV4FlashLayerSemanticBindings,
    stager: DeepSeekV4FlashGPUWeightStager,
    backend: DeepSeekV4FlashGPUBackend | _SlidingLayerBackend,
    state: DeepSeekV4FlashGPURequestState | None,
    token_idx: int,
    kv_rows: torch.Tensor | None,
    extra_kv_rows: torch.Tensor | None = None,
) -> torch.Tensor:
```

2. After `kv_rows` is finalized (after the `extra_kv_rows` concat block) and the CUDA check, replace:

```python
    context = shared_kv_swa_attention_reference(
        query,
        kv_rows,
        stager.stage_vector(attn_sinks),
    )
```

with:

```python
    staged_sinks = stager.stage_vector(attn_sinks)
    fused_attn = getattr(backend, "fused_sliding_window_attention", None)
    context: torch.Tensor | None = None
    if callable(fused_attn):
        try:
            context = fused_attn(
                query=query,
                kv_rows=kv_rows,
                attn_sinks=staged_sinks,
                token_idx=token_idx,
            )
        except (RuntimeError, NotImplementedError, ValueError):
            context = None
    if context is None:
        context = shared_kv_swa_attention_reference(
            query,
            kv_rows,
            staged_sinks,
        )
```

3. Pass `backend=backend` at both call sites:
   - `deepseek_v4_flash_sliding_layer_forward` around line 565.
   - `deepseek_v4_flash_compressed_layer_forward` around line 897.

- [ ] **Step 3: Write the backend contract test**

Add to `tests/deepseek_v4_flash/test_gpu_backend_contract.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_sliding_window_attention_matches_reference() -> None:
    from vllm.model_executor.models.deepseek_v4_flash.attention import (
        shared_kv_swa_attention_reference,
    )
    backend = DeepSeekV4FlashGPUBackend()
    num_heads = 64
    head_dim = 512
    window = 128
    torch.manual_seed(123)
    query = torch.randn((num_heads, head_dim), dtype=torch.float32, device="cuda")
    kv_rows = torch.randn((window, head_dim), dtype=torch.float32, device="cuda")
    attn_sinks = torch.randn((num_heads,), dtype=torch.float32, device="cuda")

    expected = shared_kv_swa_attention_reference(query, kv_rows, attn_sinks)
    actual = backend.fused_sliding_window_attention(
        query=query,
        kv_rows=kv_rows,
        attn_sinks=attn_sinks,
        token_idx=7,
    )
    assert backend.stats()["fused_sliding_attention_api_calls"] == 1
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/deepseek_v4_flash/test_attention_reference.py tests/deepseek_v4_flash/test_gpu_backend_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py vllm/model_executor/models/deepseek_v4_flash/gpu_layers.py tests/deepseek_v4_flash/test_gpu_backend_contract.py
git commit -m "feat(deepseek): fused sliding-window attention backend and layer integration"
```

---

### Task 3: Correctness gates and benchmark

**Files:**
- Modify: `docs/superpowers/plans/2026-06-28-deepseek-v4-flash-performance.md` (benchmark results)

- [ ] **Step 1: Run targeted tests**

```bash
uv run pytest tests/deepseek_v4_flash/test_attention_reference.py tests/deepseek_v4_flash/test_gpu_backend_contract.py tests/deepseek_v4_flash/test_gpu_moe_path.py -q
```

Expected: PASS.

- [ ] **Step 2: Run regression suites**

```bash
bash tests/run_regression_suite.sh
bash tests/run_deepseek_v4_flash_real_smoke.sh
SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh
```

Expected: all green.

- [ ] **Step 3: Quality smoke benchmark (3 runs after warmup)**

```bash
uv run python tests/tools/deepseek_v4_flash_quality_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --context-length 4096 --max-tokens 32 --min-output-chars 8 \
  --prompt-text "What is the capital of France?" --json-out /tmp/ds_m2_run1.json
# repeat for run2, run3
```

Capture the 3-run average `decode_tps` and compare to Milestone 1 average (~0.563 tok/s).

- [ ] **Step 4: Record results**

Append a one-line Milestone 2 result to `docs/superpowers/plans/2026-06-28-deepseek-v4-flash-performance.md` under a new "Milestone 2 Results" section and commit:

```bash
git add docs/superpowers/plans/2026-06-28-deepseek-v4-flash-performance.md
git commit -m "docs: milestone 2 benchmark results"
```

---

## Self-Review

- **Spec coverage:** The plan adds a fused kernel (Task 1), backend/layer hook with fallback (Task 2), and gates/benchmark (Task 3). The high-level design from the parent plan is fully covered.
- **Placeholder scan:** All steps contain concrete code, commands, and expected outputs; no TBD/placeholder language.
- **Type consistency:** `query [num_heads, head_dim]`, `kv_rows [window, head_dim]`, `attn_sinks [num_heads]` match the existing reference `shared_kv_swa_attention_reference` signature. `token_idx` is passed through the backend method for API symmetry but is not needed by the kernel.
