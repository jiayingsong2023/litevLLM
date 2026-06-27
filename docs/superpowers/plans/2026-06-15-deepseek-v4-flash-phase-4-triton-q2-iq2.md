# DeepSeek V4 Flash Phase 4 Triton Q2/IQ2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Phase 3 CPU-reference Q2_K/IQ2_XXS expert matvec fallback with real Triton dequant-matvec kernels for DeepSeek V4 Flash batch=1 greedy decode.

**Architecture:** Keep the public backend contract in `q2_iq2_moe.py` stable: CUDA raw payload + CUDA hidden vector in, CUDA fp32 vector out. Implement type-specific Triton kernels behind that contract, then wire `quantized_expert_gemm()` to use them by default with a debug fallback only for kernel-disabled tests. Do not expand model support beyond batch=1 greedy decode in this phase.

**Tech Stack:** Python 3.12, `uv`, PyTorch CUDA/ROCm tensors, Triton imported only through `vllm.triton_utils`, pytest, real GGUF smoke profiling.

---

## Current Baseline

Phase 3 added:

- raw grouped expert payload staging as CUDA `uint8`;
- routed MoE preference for raw quantized expert payloads;
- `deepseek_v4_q2_k_matvec()` and `deepseek_v4_iq2_xxs_matvec()` contracts;
- `DeepSeekV4FlashGPUBackend.quantized_expert_gemm()`;
- stable `phase3_metrics` in `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`.

The current limitation is explicit: both matvec functions still copy CUDA payload
bytes to CPU, decode with the Python reference decoder, copy a full fp32 matrix
back to CUDA, then call `matmul`. The Phase 3 real smoke was correct but slow:

- `max_tokens=1`, context 4096, output `[1, 32974]`;
- `runs[0].elapsed_ms=424678.46875`;
- `phase3_metrics.quantized_expert_calls=258`.

Phase 4 must remove this CPU decode path from the common expert hot path.

## File Responsibilities

- `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`: public wrappers, Triton kernels, fallback gating, kernel launch parameters, layout comments.
- `vllm/model_executor/models/deepseek_v4_flash/quant.py`: reference decoders and small test payload builders if needed.
- `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`: quantized expert counters and optional fallback controls.
- `tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py`: synthetic correctness and shape tests for both quant types.
- `tests/deepseek_v4_flash/test_gpu_backend_contract.py`: backend-level expert GEMM tests.
- `tests/deepseek_v4_flash/test_deepseek_q2_iq2_kernel_profile.py`: microbenchmark/profiling tests kept opt-in where needed.
- `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`: profile schema additions for Phase 4 kernel usage.
- `docs/design/deepseek_v4_flash_q2_native.md`: Phase 4 design notes and measured results.

## Acceptance Target

- Q2_K and IQ2_XXS matvec wrappers no longer call `_payload_bytes()` in the default path.
- Synthetic CUDA tests compare Triton output to existing Python reference decode for multiple row counts and hidden vectors.
- Backend `quantized_expert_gemm()` uses Triton Q2/IQ2 matvec by default.
- Real GGUF smoke for `max_tokens=1`, `context_length=4096` returns `[1, 32974]`.
- Real one-token smoke is materially faster than the Phase 3 `424678 ms` baseline, or the design doc records the blocker with profiler evidence.
- No direct `import triton`; use `from vllm.triton_utils import tl, triton`.
- Every new Triton kernel has ASCII comments describing memory layout and thread/block tiling.

## Task 1: Add Kernel Feature Gates and CPU-Fallback Tests

**Files:**
- Modify: `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Test: `tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py`
- Test: `tests/deepseek_v4_flash/test_gpu_backend_contract.py`

- [ ] **Step 1: Add tests that expose fallback usage**

Add tests that monkeypatch `_payload_bytes` to raise and assert the new default
kernel path does not call it when Triton is available:

```python
def test_q2_k_default_path_does_not_copy_payload_to_cpu(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is required")
    monkeypatch.setattr(
        q2_iq2_moe,
        "_payload_bytes",
        lambda _payload: (_ for _ in ()).throw(AssertionError("CPU fallback used")),
    )
    payload = torch.zeros(84, dtype=torch.uint8, device="cuda")
    hidden = torch.ones(256, dtype=torch.float32, device="cuda")
    out = q2_iq2_moe.deepseek_v4_q2_k_matvec(payload, hidden, rows=1, columns=256)
    assert out.shape == (1,)
    assert out.is_cuda
```

Add the same test shape for IQ2_XXS with a `66` byte payload.

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py -q
```

Expected before implementation: failure because `_payload_bytes()` is called.

- [ ] **Step 2: Add explicit fallback switch**

Add keyword arguments to wrappers:

```python
def deepseek_v4_q2_k_matvec(..., use_triton: bool = True) -> torch.Tensor: ...
def deepseek_v4_iq2_xxs_matvec(..., use_triton: bool = True) -> torch.Tensor: ...
```

Keep fallback in a private function:

```python
def _q2_k_matvec_reference_cuda(...): ...
def _iq2_xxs_matvec_reference_cuda(...): ...
```

Default `use_triton=True`; tests can call `use_triton=False` for fallback parity.

- [ ] **Step 3: Add backend stats**

In `DeepSeekV4FlashGPUBackend`, track:

- `q2_k_triton_calls`
- `iq2_xxs_triton_calls`
- `q2_iq2_reference_fallback_calls`

Do not remove existing `quantized_expert_calls`.

- [ ] **Step 4: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py tests/deepseek_v4_flash/test_gpu_backend_contract.py -q
uv run --no-sync ruff check vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py tests/deepseek_v4_flash/test_gpu_backend_contract.py
```

Commit:

```bash
git add vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py \
  vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py \
  tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py \
  tests/deepseek_v4_flash/test_gpu_backend_contract.py
git commit -m "feat: gate deepseek q2 iq2 triton kernels"
```

## Task 2: Implement Q2_K Triton Matvec Kernel

**Files:**
- Modify: `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`
- Test: `tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py`

- [ ] **Step 1: Add Q2_K correctness tests**

Add parameterized CUDA tests:

```python
@pytest.mark.parametrize("rows", [1, 2, 17, 64])
@pytest.mark.parametrize("hidden_scale", [0.25, 1.0])
def test_q2_k_triton_matches_reference_for_multiple_rows(rows, hidden_scale) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is required")
    columns = 256
    payload_bytes = _deterministic_q2_k_payload(rows)
    payload = torch.tensor(list(payload_bytes), dtype=torch.uint8, device="cuda")
    hidden = torch.linspace(-1.0, 1.0, columns, device="cuda") * hidden_scale
    expected = q2_k_matrix_from_gguf_payload(
        payload_bytes,
        rows=rows,
        columns=columns,
    ).to("cuda").matmul(hidden)
    actual = deepseek_v4_q2_k_matvec(payload, hidden, rows=rows, columns=columns)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
```

Use deterministic bytes with valid fp16 scale fields at bytes `80..83` for each
84-byte block. Keep values bounded to avoid excessive tolerance needs.

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py::test_q2_k_triton_matches_reference_for_multiple_rows -q
```

Expected before implementation: failure or fallback-use assertion failure.

- [ ] **Step 2: Implement `_q2_k_matvec_kernel`**

Use:

```python
from vllm.triton_utils import tl, triton
```

Kernel shape:

- one program computes a block of output rows, `BLOCK_ROWS` default `16`;
- each output row spans `columns / 256` Q2_K blocks;
- for the first implementation, require `columns == 256` or handle multiple
  blocks with a static loop over `num_k_blocks`;
- output tensor is `[rows]` fp32.

Memory layout comments must state:

- Q2_K block = 84 bytes;
- bytes `0..15` scales/mins nibbles;
- bytes `16..79` 2-bit quants;
- bytes `80..81` fp16 `d`;
- bytes `82..83` fp16 `dmin`;
- row-major block order.

Use the same decode formula as `decode_q2_k_gguf_blocks_reference()`:

```text
scale_low = scale_byte & 0x0F
scale_high = scale_byte >> 4
decoded = d * scale_low * code - dmin * scale_high
```

- [ ] **Step 3: Add wrapper allocation and launch**

In `deepseek_v4_q2_k_matvec()`:

- validate CUDA tensors and shapes;
- allocate `out = torch.empty((rows,), device=hidden.device, dtype=torch.float32)`;
- launch `_q2_k_matvec_kernel[(triton.cdiv(rows, BLOCK_ROWS),)](...)`;
- return `out`.

- [ ] **Step 4: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py -q
uv run --no-sync ruff check vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py
```

Commit:

```bash
git add vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py \
  tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py
git commit -m "feat(kernels): add deepseek q2 k matvec"
```

## Task 3: Implement IQ2_XXS Triton Matvec Kernel

**Files:**
- Modify: `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/quant.py`
- Test: `tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py`

- [ ] **Step 1: Add IQ2_XXS correctness tests**

Add parameterized CUDA tests:

```python
@pytest.mark.parametrize("rows", [1, 2, 17, 64])
def test_iq2_xxs_triton_matches_reference_for_multiple_rows(rows) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is required")
    columns = 256
    payload_bytes = _deterministic_iq2_xxs_payload(rows)
    payload = torch.tensor(list(payload_bytes), dtype=torch.uint8, device="cuda")
    hidden = torch.linspace(-0.5, 0.5, columns, device="cuda")
    expected = iq2_xxs_matrix_from_gguf_payload(
        payload_bytes,
        rows=rows,
        columns=columns,
    ).to("cuda").matmul(hidden)
    actual = deepseek_v4_iq2_xxs_matvec(payload, hidden, rows=rows, columns=columns)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
```

Use deterministic payloads that exercise non-zero scales and sign/grid metadata.

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py::test_iq2_xxs_triton_matches_reference_for_multiple_rows -q
```

Expected before implementation: failure or fallback-use assertion failure.

- [ ] **Step 2: Add device constants for IQ2 lookup tables**

The Python reference uses `_IQ2_XXS_KSIGNS` and `_IQ2_XXS_GRID_HEX`. For Triton,
avoid a large dynamic table in the first kernel. Add one of these two approaches:

- preferred: inline the same compact arithmetic/table lookup for the 256-entry
  sign table and grid table in Triton using constexpr values split into small
  vectors;
- acceptable first pass: stage precomputed lookup tensors from Python and pass
  them as CUDA tensors to the kernel.

If using lookup tensors, expose:

```python
def iq2_xxs_lookup_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

and cache by device.

- [ ] **Step 3: Implement `_iq2_xxs_matvec_kernel`**

Kernel shape:

- one program computes `BLOCK_ROWS` output rows;
- one 66-byte block decodes 256 values;
- accumulate 256 hidden products in fp32;
- output `[rows]` fp32.

Memory layout comments must state:

- IQ2_XXS block = 66 bytes;
- bytes `0..1` fp16 `d`;
- bytes `2..65` eight 8-byte groups;
- each group contains packed grid bytes and `q_sign_scale` metadata;
- row-major block order.

The kernel must match `decode_iq2_xxs_gguf_blocks_reference()` within test
tolerance.

- [ ] **Step 4: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py -q
uv run --no-sync ruff check vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py vllm/model_executor/models/deepseek_v4_flash/quant.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py
```

Commit:

```bash
git add vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py \
  vllm/model_executor/models/deepseek_v4_flash/quant.py \
  tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py
git commit -m "feat(kernels): add deepseek iq2 xxs matvec"
```

## Task 4: Fuse Gate and Up Matvec for Routed Experts

**Files:**
- Modify: `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Test: `tests/deepseek_v4_flash/test_gpu_backend_contract.py`
- Test: `tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py`

- [ ] **Step 1: Add backend fusion tests**

Add a test that calls `quantized_expert_gemm()` with gate/up IQ2_XXS payloads
and down Q2_K payload, then compares to reference:

```python
expected_gate = iq2_xxs_matrix_from_gguf_payload(gate_bytes, rows=i, columns=h)
expected_up = iq2_xxs_matrix_from_gguf_payload(up_bytes, rows=i, columns=h)
expected_down = q2_k_matrix_from_gguf_payload(down_bytes, rows=h, columns=i)
expected = expected_down.to("cuda").matmul(
    torch.nn.functional.silu(expected_gate.to("cuda").matmul(hidden))
    * expected_up.to("cuda").matmul(hidden)
)
actual = backend.quantized_expert_gemm(...)
torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)
```

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_backend_contract.py::test_quantized_expert_gemm_matches_reference_decode -q
```

- [ ] **Step 2: Add optional fused gate/up helper**

Add:

```python
def deepseek_v4_iq2_xxs_gate_up(
    gate_payload: torch.Tensor,
    up_payload: torch.Tensor,
    hidden: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

First implementation may launch two IQ2 kernels if that is simpler. If a single
kernel is implemented, it should write two fp32 outputs `[rows]` and share hidden
loads.

- [ ] **Step 3: Update backend**

Update `quantized_expert_gemm()`:

- use `deepseek_v4_iq2_xxs_gate_up()` when gate and up are IQ2_XXS with matching
  shape;
- use Q2_K matvec for down;
- keep explicit unsupported-type errors.

- [ ] **Step 4: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_backend_contract.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py -q
uv run --no-sync ruff check vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py tests/deepseek_v4_flash/test_gpu_backend_contract.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py
```

Commit:

```bash
git add vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py \
  vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py \
  tests/deepseek_v4_flash/test_gpu_backend_contract.py \
  tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py
git commit -m "feat: fuse deepseek quantized expert gate up"
```

## Task 5: Add Microbenchmark and Register-Pressure Guardrails

**Files:**
- Create: `tests/deepseek_v4_flash/test_deepseek_q2_iq2_kernel_profile.py`
- Modify: `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`
- Test: `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py`

- [ ] **Step 1: Add opt-in microbenchmark**

Create an opt-in test:

```python
def test_q2_iq2_kernel_profile_smoke() -> None:
    if os.environ.get("RUN_DEEPSEEK_Q2_IQ2_PROFILE") != "1":
        pytest.skip("set RUN_DEEPSEEK_Q2_IQ2_PROFILE=1")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is required")
    ...
```

Measure:

- Q2_K rows `4096`, columns `2048` if memory allows, else rows `1024`;
- IQ2_XXS rows `4096`, columns `2048` if memory allows, else rows `1024`;
- median milliseconds over at least 10 repeats after 3 warmups.

The test should print JSON with:

- `q2_k_ms`
- `iq2_xxs_ms`
- `q2_k_effective_gbps`
- `iq2_xxs_effective_gbps`

- [ ] **Step 2: Add smoke profile fields**

Extend `phase3_metrics()` or add `phase4_metrics()` in
`tests/tools/run_deepseek_v4_flash_gpu_smoke.py`:

- `q2_k_triton_calls`
- `iq2_xxs_triton_calls`
- `q2_iq2_reference_fallback_calls`

Add unit tests in `test_real_smoke_tool_cli.py` that schema fields are always
present.

- [ ] **Step 3: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_real_smoke_tool_cli.py tests/deepseek_v4_flash/test_deepseek_q2_iq2_kernel_profile.py -q
uv run --no-sync ruff check tests/deepseek_v4_flash/test_deepseek_q2_iq2_kernel_profile.py tests/tools/run_deepseek_v4_flash_gpu_smoke.py tests/deepseek_v4_flash/test_real_smoke_tool_cli.py
```

Commit:

```bash
git add tests/deepseek_v4_flash/test_deepseek_q2_iq2_kernel_profile.py \
  tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  tests/deepseek_v4_flash/test_real_smoke_tool_cli.py
git commit -m "feat: profile deepseek q2 iq2 kernels"
```

## Task 6: Run Real GGUF Smoke and Remove Hot-Path CPU Fallback

**Files:**
- Modify: `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Test: `tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py`

- [ ] **Step 1: Add a no-fallback backend test**

Add a test that asserts a full `quantized_expert_gemm()` increments Triton-call
counters and leaves `q2_iq2_reference_fallback_calls == 0`.

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_backend_contract.py::test_quantized_expert_gemm_uses_triton_without_reference_fallback -q
```

- [ ] **Step 2: Disable reference fallback in default path**

Keep reference fallback callable only through `use_triton=False` in low-level
tests. The backend default path must not use it.

- [ ] **Step 3: Run real one-token smoke**

Run:

```bash
timeout --foreground --kill-after=60s 2400s \
  uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --context-length 4096 \
  --max-tokens 1 \
  --repeat 1 \
  --profile-json /tmp/deepseek-v4-flash-phase4-smoke-profile.json
```

Expected:

- output token ids remain `[1, 32974]`;
- `q2_iq2_reference_fallback_calls == 0`;
- one-token elapsed time is below the Phase 3 `424678 ms` baseline.

- [ ] **Step 4: Commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py tests/deepseek_v4_flash/test_gpu_backend_contract.py -q
uv run --no-sync ruff check vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py tests/deepseek_v4_flash/test_gpu_backend_contract.py
```

Commit:

```bash
git add vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py \
  vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py \
  tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py \
  tests/deepseek_v4_flash/test_gpu_backend_contract.py
git commit -m "feat: use deepseek q2 iq2 triton by default"
```

## Task 7: Real Max-8 Smoke and Documentation

**Files:**
- Modify: `docs/design/deepseek_v4_flash_q2_native.md`
- Modify: `docs/superpowers/plans/2026-06-15-deepseek-v4-flash-phase-4-triton-q2-iq2.md`

- [ ] **Step 1: Run max-8 real smoke**

Run:

```bash
timeout --foreground --kill-after=60s 7200s \
  uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --context-length 4096 \
  --max-tokens 8 \
  --repeat 1 \
  --profile-json /tmp/deepseek-v4-flash-phase4-max8-profile.json
```

Record:

- output token ids;
- elapsed ms;
- tokens/sec;
- q2/iq2 Triton calls;
- fallback calls;
- cache hits/misses;
- largest layer timings.

- [ ] **Step 2: Update design doc**

Add a `Phase 4 Performance Notes` section:

- exact commands run;
- one-token and max-8 timings;
- whether output tokens match prior smoke;
- remaining bottlenecks, especially output projection, attention, cache update,
  or kernel register pressure;
- next phase recommendation.

- [ ] **Step 3: Final verification**

Run:

```bash
bash -n tests/run_inference_correctness_regression.sh
uv run --no-sync pytest tests/deepseek_v4_flash/test_profiler.py tests/deepseek_v4_flash/test_real_smoke_tool_cli.py tests/deepseek_v4_flash/test_gpu_weight_staging.py tests/deepseek_v4_flash/test_model_kernel_generate.py tests/deepseek_v4_flash/test_gpu_backend_contract.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py tests/deepseek_v4_flash/test_expert_cache_policy.py tests/deepseek_v4_flash/test_quantized_expert_payload.py tests/deepseek_v4_flash/test_expert_prefetch.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash vllm/kernels/triton/deepseek_v4_flash tests/deepseek_v4_flash tests/tools/run_deepseek_v4_flash_gpu_smoke.py
```

- [ ] **Step 4: Commit**

Commit:

```bash
git add docs/design/deepseek_v4_flash_q2_native.md \
  docs/superpowers/plans/2026-06-15-deepseek-v4-flash-phase-4-triton-q2-iq2.md
git commit -m "docs: record deepseek phase 4 triton performance"
```

## Self-Review

- Spec coverage: the plan covers real Triton Q2_K matvec, real Triton IQ2_XXS
  matvec, backend integration, no hot-path CPU fallback, profiling, real smoke,
  and documentation.
- Placeholder scan: no unresolved placeholder markers or unspecified
  implementation tasks are used.
- Type consistency: wrapper names, backend counters, smoke metrics, and test
  names are defined before later tasks reference them.
