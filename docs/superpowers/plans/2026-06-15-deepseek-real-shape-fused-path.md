# DeepSeek Real Expert Shape Fused Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the target DeepSeek V4 Flash GGUF routed experts hit the fused IQ2 gate/up activation path in real one-token smoke runs.

**Architecture:** Keep the existing correctness fallback intact, but remove the synthetic-only `columns==256` limitation from the fused gate/up path by first measuring the real payload shapes, then implementing multi-block IQ2 accumulation for `columns == 256 * N`. Backend dispatch must only widen after tests prove the fused helper matches the reference for multi-block shapes.

**Tech Stack:** Python 3.12 via `uv`, PyTorch CUDA/ROCm tensors, Triton through `vllm.triton_utils`, pytest, target GGUF `models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.

---

## File Map

- `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`: fused dispatch conditions and backend stats.
- `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`: IQ2 fused gate/up activation kernels.
- `vllm/model_executor/models/deepseek_v4_flash/weight_store.py`: source of real grouped expert `rows`/`columns` semantics.
- `tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py`: synthetic multi-block kernel correctness.
- `tests/deepseek_v4_flash/test_gpu_backend_contract.py`: backend dispatch and stats.
- `tests/tools/inspect_deepseek_v4_flash_expert_shapes.py`: new lightweight shape inspection helper.
- `tests/tools/run_deepseek_v4_flash_gpu_smoke.py`: real-smoke metrics already report fused call counts.
- `docs/design/deepseek_v4_flash_q2_native.md`: record real shape and smoke result.

---

### Task 1: Add Real Expert Shape Inspection

**Files:**
- Create: `tests/tools/inspect_deepseek_v4_flash_expert_shapes.py`
- Test: `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py`

- [ ] **Step 1: Add shape inspector helper**

Create a small CLI that opens the target GGUF, walks semantic layer bindings, and prints JSON records for grouped routed expert tensors:

```json
{
  "layer_idx": 2,
  "projection": "gate",
  "tensor_name": "blk.2.ffn_gate_exps.weight",
  "ggml_type": 19,
  "rows": 2048,
  "columns": 7168,
  "expert_count": 256,
  "columns_blocks": 28,
  "nbytes_per_expert": 118272
}
```

The helper must not decode weights or allocate CUDA tensors. It should use `ggml_tensor_nbytes((input_size, output_size), tensor.tensor_type)` and the existing `(input_size, output_size, expert_count)` dims convention.

- [ ] **Step 2: Add CLI parser test**

Add a parser/import test in `tests/deepseek_v4_flash/test_real_smoke_tool_cli.py` similar to the existing smoke helper tests. Do not require the real model file in normal pytest.

- [ ] **Step 3: Run inspector on the real GGUF**

Run:

```bash
uv run --no-sync python tests/tools/inspect_deepseek_v4_flash_expert_shapes.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --limit 12 \
  --json /tmp/deepseek-v4-flash-expert-shapes.json
```

Expected: JSON shows actual gate/up/down `rows`, `columns`, tensor types, and confirms whether gate/up are `IQ2_XXS` with `columns % 256 == 0`.

- [ ] **Step 4: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_real_smoke_tool_cli.py -q
uv run --no-sync ruff check tests/tools/inspect_deepseek_v4_flash_expert_shapes.py tests/deepseek_v4_flash/test_real_smoke_tool_cli.py
git add tests/tools/inspect_deepseek_v4_flash_expert_shapes.py tests/deepseek_v4_flash/test_real_smoke_tool_cli.py
git commit -m "test: inspect deepseek real expert shapes"
```

Expected: tests and ruff pass; `/tmp/deepseek-v4-flash-expert-shapes.json` exists for implementation reference.

---

### Task 2: Add Multi-Block IQ2 Fused Gate/Up Correctness Tests

**Files:**
- Modify: `tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py`

- [ ] **Step 1: Extend deterministic payload helper**

Add a helper that can build IQ2 payloads for `rows` and `columns_blocks`:

```python
def _iq2_xxs_deterministic_payload_blocks(rows: int, blocks: int) -> bytes:
    return b"".join(
        _iq2_xxs_deterministic_payload(rows)
        for _ in range(blocks)
    )
```

If the current helper is row-major by row, adjust the new helper so payload order matches GGUF row-major blocks: all 256-column blocks for row 0, then all blocks for row 1, and so on.

- [ ] **Step 2: Add failing multi-block test**

Add a CUDA test:

```python
@pytest.mark.parametrize("columns", [512, 1024])
def test_iq2_xxs_gate_up_activation_matches_reference_multi_block(columns: int) -> None:
    rows = 4
    gate_payload = _iq2_xxs_deterministic_payload_blocks(rows, columns // 256)
    up_payload = _iq2_xxs_deterministic_payload_blocks(rows, columns // 256)
    hidden = torch.linspace(-0.75, 0.9, columns, dtype=torch.float32, device="cuda")
    actual = deepseek_v4_iq2_xxs_gate_up_activation(
        _cuda_payload(gate_payload),
        _cuda_payload(up_payload),
        hidden,
        rows=rows,
        columns=columns,
    )
    gate = iq2_xxs_matrix_from_gguf_payload(gate_payload, rows=rows, columns=columns).to("cuda").matmul(hidden)
    up = iq2_xxs_matrix_from_gguf_payload(up_payload, rows=rows, columns=columns).to("cuda").matmul(hidden)
    torch.testing.assert_close(actual, torch.nn.functional.silu(gate) * up, rtol=4e-2, atol=4e-2)
```

Expected before implementation: `NotImplementedError` for `columns != 256`.

- [ ] **Step 3: Verify failure**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py::test_iq2_xxs_gate_up_activation_matches_reference_multi_block -q
```

Expected: fails because multi-block fused activation is not implemented.

---

### Task 3: Implement Multi-Block IQ2 Gate/Up Activation Kernel

**Files:**
- Modify: `vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py`
- Test: `tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py`

- [ ] **Step 1: Add multi-block Triton kernel**

Keep the existing single-block kernel for `columns==256`. Add a second kernel for `columns_blocks > 1`:

```text
program_id(0): output row
program_id(1): optional row tile or single row only in first version
lanes: 256 values for one GGUF block
loop: BLOCK_IDX from 0 to COLUMNS_BLOCKS - 1
payload offset: (row * COLUMNS_BLOCKS + block_idx) * 66
hidden offset: block_idx * 256 + lane
accumulate gate_total and up_total across blocks
store silu(gate_total) * up_total
```

Use `tl.static_range` only if `COLUMNS_BLOCKS` is a constexpr. The first version can use one Triton program per row and one loop over blocks; correctness and real dispatch are more important than peak speed.

- [ ] **Step 2: Dispatch by block count**

In `deepseek_v4_iq2_xxs_gate_up_activation()`:

```python
if columns % _GGUF_BLOCK_COLUMNS != 0:
    raise ValueError(...)
columns_blocks = columns // _GGUF_BLOCK_COLUMNS
if columns_blocks == 1:
    return _iq2_xxs_gate_up_activation_triton_cuda(...)
return _iq2_xxs_gate_up_activation_multiblock_triton_cuda(...)
```

Do not widen `deepseek_v4_iq2_xxs_matvec()` or `deepseek_v4_iq2_xxs_gate_up()` in this task unless needed for tests. The goal is only fused activation.

- [ ] **Step 3: Run kernel tests**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py -q
uv run --no-sync ruff check vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py
```

Expected: all q2/iq2 tests pass, including `columns=512` and `columns=1024`.

- [ ] **Step 4: Commit**

Run:

```bash
git add vllm/kernels/triton/deepseek_v4_flash/q2_iq2_moe.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py
git commit -m "perf(kernels): support multiblock deepseek iq2 gate up"
```

---

### Task 4: Widen Backend Fused Dispatch To Real Shapes

**Files:**
- Modify: `vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py`
- Modify: `tests/deepseek_v4_flash/test_gpu_backend_contract.py`

- [ ] **Step 1: Add backend multi-block dispatch test**

Add a test where gate/up are `IQ2_XXS`, matching rows/columns, `columns=512`, and down is a supported payload. The backend should:

```python
assert backend.stats()["iq2_xxs_gate_up_fused_calls"] == 1
assert backend.stats()["iq2_xxs_triton_calls"] does not include gate/up fallback calls
```

If current down projection only supports `columns==256`, build the test so down receives the fused activation with `columns=rows` that remains supported, or explicitly expect the down limitation. Do not hide fallback to CPU.

- [ ] **Step 2: Widen dispatch condition**

Change:

```python
and gate_payload.columns == 256
```

to:

```python
and gate_payload.columns % 256 == 0
```

Keep all type and matching-shape checks.

- [ ] **Step 3: Track real shape misses**

Add a backend counter such as `iq2_xxs_gate_up_fused_shape_misses` only if dispatch still needs to skip valid IQ2 gate/up pairs. This counter should remain zero once Task 3 supports real columns.

- [ ] **Step 4: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_gpu_backend_contract.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py tests/deepseek_v4_flash/test_gpu_backend_contract.py
git add vllm/model_executor/models/deepseek_v4_flash/gpu_backend.py tests/deepseek_v4_flash/test_gpu_backend_contract.py
git commit -m "perf: dispatch deepseek real iq2 gate up fused path"
```

Expected: backend stats prove multi-block gate/up uses the fused activation helper.

---

### Task 5: Prove Real GGUF Fused Path Hit

**Files:**
- Modify: `tests/tools/run_deepseek_v4_flash_gpu_smoke.py` if extra metrics are needed
- Modify: `docs/design/deepseek_v4_flash_q2_native.md`

- [ ] **Step 1: Run one-token real smoke**

Run:

```bash
timeout --foreground --kill-after=60s 2400s \
  uv run --no-sync python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
  --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --context-length 4096 \
  --max-tokens 1 \
  --repeat 1 \
  --profile-json /tmp/deepseek-v4-flash-real-fused-one.json
```

Required success criteria:

```text
output_token_ids == [1, 32974]
q2_iq2_reference_fallback_calls == 0
iq2_xxs_gate_up_fused_calls > 0
iq2_xxs_triton_calls < previous 516 gate/up-heavy count
```

Performance improvement is desirable but not the primary gate for this plan. The primary gate is real fused-path hit.

- [ ] **Step 2: Decide whether to run max-8**

Run max-8 only if one-token fused calls are positive and elapsed time is not worse than the previous usable-pass baseline:

```text
previous one-token baseline: 432429.125 ms
```

If one-token is still slower, skip max-8 and record the blocker.

- [ ] **Step 3: Update design doc**

Record:

- inspected real expert shapes
- fused dispatch conditions
- one-token smoke output
- fused call counts
- remaining misses or shape gaps

- [ ] **Step 4: Verify and commit**

Run:

```bash
uv run --no-sync pytest tests/deepseek_v4_flash/test_real_smoke_tool_cli.py tests/deepseek_v4_flash/test_gpu_backend_contract.py tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash vllm/kernels/triton/deepseek_v4_flash tests/deepseek_v4_flash tests/tools
git add docs/design/deepseek_v4_flash_q2_native.md tests/tools/run_deepseek_v4_flash_gpu_smoke.py
git commit -m "docs: record deepseek real fused expert path"
```

---

## Final Verification

Run:

```bash
bash -n tests/run_inference_correctness_regression.sh
uv run --no-sync pytest tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py tests/deepseek_v4_flash/test_gpu_backend_contract.py tests/deepseek_v4_flash/test_real_smoke_tool_cli.py -q
uv run --no-sync ruff check vllm/model_executor/models/deepseek_v4_flash vllm/kernels/triton/deepseek_v4_flash tests/deepseek_v4_flash tests/tools
```

Push after the branch is clean:

```bash
git push origin feat/deepseek-v4-flash-q2-native
```
