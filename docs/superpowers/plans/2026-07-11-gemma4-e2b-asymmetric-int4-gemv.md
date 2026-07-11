# Gemma4-E2B Asymmetric AWQ-INT4 M=1 GEMV Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one Triton kernel for compressed-tensors asymmetric packed-int4 M=1 GEMV and dispatch to it during Gemma4-E2B decode, then measure whether whole-model BS=1 decode TPS improves while remaining bit-exact.

**Architecture:** Add `_packed_int4_asymmetric_group32_gemv_m1` in `awq_fused_gemm.py` and a CPU wrapper. Add a kernel-policy key `awq_asymmetric_gemv` wired from `FASTINFERENCE_AWQ_ASYMMETRIC_GEMV` tuning env through the Gemma4 adapter. When enabled, `AWQWeight.matmul()` dispatches M=1 asymmetric inputs to the new kernel; the baseline path is the same code with the policy disabled. QKV fusion, gate-up fusion, PLE fusion, and lm_head fast path are explicitly out of scope.

**Tech Stack:** Python 3.12, Triton, PyTorch, ROCm/CUDA, `uv run`.

## Global Constraints

- Python 3.12 only; always use `uv run`.
- No runtime `os.environ` reads in `vllm/`; tuning env is passed via `runtime_config.tuning_env` / `fastinference_config.tuning_keyvals`.
- No C++ kernels; Triton only.
- Every Triton kernel must include ASCII comments describing memory layout and thread/block tiling.
- Treat Triton compilation warnings as failures.
- Run `uv run ruff check . && uv run ruff format .` after code changes.
- New kernel requires PyTorch reference correctness test plus edge cases (FP16/BF16, invalid shapes, real checkpoint tensor).
- Whole-model performance claims require 3-run median decode TPS per context bucket and bit-exact token comparison.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `vllm/kernels/triton/awq_fused_gemm.py` | Asymmetric group32 M=1 GEMV kernel and CPU wrapper. |
| `vllm/model_executor/layers/quantization/tensor.py` | Kernel-policy helper `awq_asymmetric_gemv_enabled`; dispatch in `AWQWeight.matmul()`. |
| `vllm/adapters/policy_keys.py` | Constant `GEMMA4_AWQ_ASYMMETRIC_GEMV = "awq_asymmetric_gemv"`. |
| `vllm/adapters/gemma4.py` | Map tuning env `FASTINFERENCE_AWQ_ASYMMETRIC_GEMV` into `kernel_policy`. |
| `vllm/engine/env_registry.py` | Register `FASTINFERENCE_AWQ_ASYMMETRIC_GEMV` as tool-only/deprecated env. |
| `tests/test_project_governance.py` | Add the new env to the AWQ tensor production-env read allow-list test. |
| `tests/tools/gemma4_e2b_audit.py` | P0 audit: cold audit (fallback evidence) + perf run (decode-only TPS, token IDs). |
| `tests/tools/gemma4_e2b_quant_helpers.py` | Shared asymmetric packing helper for tests and microbench. |
| `tests/tools/gemma4_e2b_gemv_microbench.py` | Standalone kernel microbench. |
| `tests/test_awq_gemm_m1_specialization.py` | Kernel correctness tests (FP16/BF16, invalid shapes, real checkpoint). |
| `tests/test_gemma4_e2b_e4b_support.py` | Dispatch correctness test with audit-counter assertion. |

---

## Task 1: P0 Audit — Cold Audit (Fallback Evidence) + Perf Run (Decode-Only TPS)

**Files:**
- Create: `tests/tools/gemma4_e2b_audit.py`
- Test: run tool manually.

**Interfaces:**
- Consumes: `LLM` with `max_num_seqs=1`, token-ID requests via `engine.request_builder.build()`.
- Produces: JSON with `cold_audit` (fallback events, cache bytes) and `perf_run` (decode-only TPS, generated token IDs).

- [ ] **Step 1: Create audit tool without layer profile**

Create `tests/tools/gemma4_e2b_audit.py`:

```python
#!/usr/bin/env python3
"""P0 audit for Gemma4-E2B decode overhead.

Cold audit: no warmup, captures asymmetric fallback events and dense cache bytes.
Perf run:   warmup, then decode-only TPS and generated token IDs.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization.tensor import (
    get_awq_runtime_audit_summary,
    get_awq_runtime_prefix_stats,
    get_awq_runtime_stats,
    reset_awq_runtime_stats,
)


def build_prompt_of_length(tokenizer, text_seed: str, target_len: int) -> list[int]:
    seed_ids = tokenizer.encode(text_seed, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("seed text encoded to empty ids")
    repeats = (target_len // len(seed_ids)) + 2
    long_ids = (seed_ids * repeats)[:target_len]
    assert len(long_ids) == target_len
    return long_ids


def _add_request_with_token_ids(
    llm: LLM,
    request_id: str,
    input_ids: list[int],
    sampling_params: SamplingParams,
) -> None:
    engine = llm.engine
    if engine.request_builder is None:
        engine.add_request(
            "__init_request_builder__",
            "",
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        engine.abort_request("__init_request_builder__")
    assert engine.request_builder is not None
    req = engine.request_builder.build(
        request_id=request_id,
        prompt="",
        sampling_params=sampling_params,
    )
    req.input_ids = list(input_ids)
    req.guarded_prompt = ""
    engine.scheduler.enqueue_request(request_id, req)
    admitted = engine.scheduler.admit_queued_requests(max_new=1)
    assert request_id in admitted, f"request {request_id} was not admitted"


def decode_only_generate(
    llm: LLM,
    request_id: str,
    input_ids: list[int],
    max_new_tokens: int,
    reset_stats: bool,
) -> dict[str, Any]:
    if reset_stats:
        reset_awq_runtime_stats()

    _add_request_with_token_ids(
        llm,
        request_id,
        input_ids,
        SamplingParams(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=True),
    )

    step_budget = max(64, (len(input_ids) + max_new_tokens) * 4)
    final_output: Any | None = None
    first_decode_start: float | None = None
    step_count = 0
    start = time.perf_counter()

    try:
        while llm.engine.active_request_count > 0 and step_count < step_budget:
            step_count += 1
            outs = llm.engine.step()
            for out in outs:
                if out.request_id != request_id:
                    continue
                if (
                    first_decode_start is None
                    and out.outputs
                    and len(out.outputs[0].token_ids) > 0
                ):
                    first_decode_start = time.perf_counter()
                if out.finished:
                    final_output = out
    finally:
        if final_output is None:
            try:
                llm.engine.abort_request(request_id)
            except Exception:
                pass

    if final_output is None:
        raise RuntimeError(
            f"request {request_id} did not finish within step_budget={step_budget}"
        )

    end = time.perf_counter()
    first_decode_start = first_decode_start or end
    decode_elapsed = end - first_decode_start
    token_ids = list(final_output.outputs[0].token_ids)
    decode_tokens = max(0, len(token_ids) - 1)

    return {
        "decode_time_s": round(decode_elapsed, 3),
        "decode_tps": round(decode_tokens / decode_elapsed, 3) if decode_elapsed > 0 else 0.0,
        "decode_tokens": decode_tokens,
        "token_ids": token_ids,
        "awq_stats": get_awq_runtime_stats(),
    }


def run_cold_audit(llm, prompt_ids, max_new_tokens):
    """Single un-warmed request; do NOT reset stats before or after.
    Captures first-fallback events and dense cache establishment."""
    req_id = f"e2b_cold_{time.time_ns()}"
    result = decode_only_generate(llm, req_id, prompt_ids, max_new_tokens, reset_stats=False)
    return {
        "decode_tps": result["decode_tps"],
        "token_ids": result["token_ids"],
        "awq_stats": result["awq_stats"],
        "awq_prefix_stats": get_awq_runtime_prefix_stats(),
        "awq_audit_summary": get_awq_runtime_audit_summary(),
    }


def run_perf(llm, prompt_ids, max_new_tokens, repetitions):
    """Warmup once, then measured repetitions with per-run stats reset."""
    warmup_id = f"e2b_warmup_{time.time_ns()}"
    decode_only_generate(llm, warmup_id, prompt_ids, max_new_tokens, reset_stats=True)

    runs = []
    for i in range(repetitions):
        req_id = f"e2b_perf_{i}_{time.time_ns()}"
        runs.append(
            decode_only_generate(llm, req_id, prompt_ids, max_new_tokens, reset_stats=True)
        )

    runs_sorted = sorted(runs, key=lambda r: r["decode_tps"])
    median = runs_sorted[len(runs_sorted) // 2]
    return {
        "repetitions": repetitions,
        "median_decode_tps": median["decode_tps"],
        "median_decode_time_s": median["decode_time_s"],
        "token_ids": median["token_ids"],
        "awq_stats": median["awq_stats"],
    }


def run_audit(
    model: str,
    text_seed: str,
    context_bucket: int,
    max_new_tokens: int,
    repetitions: int,
    kernel_policy_json: str,
) -> dict[str, Any]:
    policy = json.loads(kernel_policy_json)
    fastinference_config = {
        "tuning_keyvals": {
            "FASTINFERENCE_AWQ_ASYMMETRIC_GEMV": "1" if policy.get("awq_asymmetric_gemv") else "0",
        }
    }

    llm = LLM(
        model=model,
        max_model_len=context_bucket + max_new_tokens + 16,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        fastinference_config=fastinference_config,
    )
    tokenizer = llm.tokenizer
    prompt_ids = build_prompt_of_length(tokenizer, text_seed, context_bucket)

    return {
        "context_bucket": context_bucket,
        "actual_context_tokens": len(prompt_ids),
        "kernel_policy": policy,
        "cold_audit": run_cold_audit(llm, prompt_ids, max_new_tokens),
        "perf_run": run_perf(llm, prompt_ids, max_new_tokens, repetitions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gemma-4-E2B-it-AWQ-INT4")
    parser.add_argument("--context-bucket", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--kernel-policy", default="{}")
    parser.add_argument("--out", default="/tmp/gemma4_e2b_audit.json")
    args = parser.parse_args()

    text_seed = "The capital of France is Paris. "
    result = run_audit(
        args.model,
        text_seed,
        args.context_bucket,
        args.max_new_tokens,
        args.repetitions,
        args.kernel_policy,
    )
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run cold audit and inspect fallback evidence**

```bash
uv run python tests/tools/gemma4_e2b_audit.py --kernel-policy '{"awq_asymmetric_gemv": false}' --out /tmp/e2b_cold.json
```

Inspect `cold_audit`:

- `awq_stats.cache_bytes` should be > 0.
- `awq_prefix_stats[<some layer prefix>]["reason:packed_int4_asymmetric_layout"]` should be > 0.
- `awq_audit_summary.first_fallbacks` may contain asymmetric layout entries.

If asymmetric fallback is not the dominant cost, stop.

- [ ] **Step 3: Run perf audit**

```bash
uv run python tests/tools/gemma4_e2b_audit.py --out /tmp/e2b_perf_baseline.json
```

Record `perf_run.median_decode_tps` and `perf_run.token_ids`.

- [ ] **Step 4: Commit**

```bash
git add tests/tools/gemma4_e2b_audit.py
git commit -m "feat(tools): E2B cold audit + decode-only perf audit"
```

---

## Task 2: Asymmetric Group32 M=1 GEMV Kernel, Helpers, and Correctness Tests

**Files:**
- Modify: `vllm/kernels/triton/awq_fused_gemm.py`
- Create: `tests/tools/gemma4_e2b_quant_helpers.py`
- Create: `tests/tools/gemma4_e2b_gemv_microbench.py`
- Test: `tests/test_awq_gemm_m1_specialization.py`

**Interfaces:**
- Consumes: `x [1, K]`, `qweight [N, K//8]`, `scales [N, K//32]`, `qzeros [N//8, K//32]`.
- Produces: `out [1, N]` matching `dequantize_asymmetric_packed_int4_pytorch` reference.

- [ ] **Step 1: Add asymmetric group32 GEMV kernel**

In `vllm/kernels/triton/awq_fused_gemm.py`, add after `_packed_int4_symmetric_group32_gemv_m1`:

```python
@triton.jit
def _packed_int4_asymmetric_group32_gemv_m1(
    a_ptr,
    b_ptr,
    s_ptr,
    z_ptr,
    c_ptr,
    N,
    K,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    stride_zn,
    stride_zk,
    stride_cn,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
):
    """
    M=1 asymmetric packed-int4 GEMV specialized for group_size=32.

    Memory layout:
      a:        [K]                input activations (fp16/bf16)
      b:        [N, K // 8]        uint32 packed weights along K
      s:        [N, K // 32]       fp16/bf16 per-row-per-group scales
      z:        [N // 8, K // 32]  uint32 packed zero points along N
      c:        [N]                output (fp16/bf16)

    Tiling:
      One program per BLOCK_N output rows.
      Outer loop over quant groups (32 elements each).
      Inner loop consumes 4 packed uint32 per group (32 weights).
      Zero points are packed along N: each uint32 holds 8 row zeros for one group.
      Dequant: (q_unsigned - z_unsigned) * scale, because the +8 offsets cancel.
    """
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    num_groups = tl.cdiv(K, 32)
    offs_g = tl.arange(0, BLOCK_GROUPS)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    z_row = offs_n // 8
    z_shift = (offs_n % 8) * 4

    for group_base in range(0, tl.cdiv(num_groups, BLOCK_GROUPS)):
        group_idx = group_base * BLOCK_GROUPS + offs_g
        mask_g = group_idx < num_groups

        scale = tl.load(
            s_ptr + offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sk,
            mask=mask_n[:, None] & mask_g[None, :],
            other=0.0,
        ).to(tl.float32)

        zero_packed = tl.load(
            z_ptr + z_row[:, None] * stride_zn + group_idx[None, :] * stride_zk,
            mask=mask_n[:, None] & mask_g[None, :],
            other=0,
        ).to(tl.int32)
        zero = ((zero_packed >> z_shift[:, None]) & 0x0F).to(tl.float32)

        group_partial = tl.zeros((BLOCK_N, BLOCK_GROUPS), dtype=tl.float32)

        for pack_in_group in tl.static_range(0, 4):
            pack_idx = group_idx * 4 + pack_in_group
            packed = tl.load(
                b_ptr + offs_n[:, None] * stride_bn + pack_idx[None, :] * stride_bk,
                mask=mask_n[:, None] & mask_g[None, :],
                other=0,
            ).to(tl.int32)

            for nibble in tl.static_range(0, 8):
                k_idx = group_idx * 32 + pack_in_group * 8 + nibble
                aval = tl.load(
                    a_ptr + k_idx * stride_ak,
                    mask=mask_g & (k_idx < K),
                    other=0.0,
                )
                q = ((packed >> (nibble * 4)) & 0x0F).to(tl.float32)
                group_partial += aval[None, :].to(tl.float32) * (q - zero) * scale

        acc += tl.sum(group_partial, axis=1)

    out = acc.to(tl.bfloat16) if USE_BF16_OUTPUT else acc.to(tl.float16)
    tl.store(c_ptr + offs_n * stride_cn, out, mask=mask_n)
```

- [ ] **Step 2: Add CPU-side safe wrapper**

Add `packed_int4_asymmetric_group32_gemv_m1` launcher function:

```python
def packed_int4_asymmetric_group32_gemv_m1(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int = 32,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert group_size == 32, "only group_size=32 is supported"
    assert x.ndim == 2 and x.shape[0] == 1, "M=1 only"
    n_rows, n_cols_packed = qweight.shape
    k = n_cols_packed * 8
    n = n_rows
    assert x.shape[1] == k
    assert scales.shape == (n, k // group_size)
    assert qzeros.shape == (n // 8, k // group_size)

    if out is None:
        out = torch.empty(1, n, dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)
    _packed_int4_asymmetric_group32_gemv_m1[grid](
        x, qweight, scales, qzeros, out,
        n, k,
        x.stride(1), qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        out.stride(1),
        BLOCK_GROUPS=4,
        BLOCK_N=128,
        USE_BF16_OUTPUT=(x.dtype == torch.bfloat16),
    )
    return out
```

- [ ] **Step 3: Create shared asymmetric packing helper**

Create `tests/tools/gemma4_e2b_quant_helpers.py`:

```python
"""Shared helpers for E2B asymmetric packed-int4 tests and microbench."""

import torch


def pack_asymmetric_int4(weights: torch.Tensor, zeros: torch.Tensor) -> tuple:
    """Pack signed int4 weights and zeros into uint32 tensors."""
    n, k = weights.shape
    assert k % 8 == 0 and n % 8 == 0
    n_groups = k // 32
    assert zeros.shape == (n, n_groups)

    qweight = torch.zeros(n, k // 8, dtype=torch.int32, device=weights.device)
    for i in range(8):
        qweight |= (weights[:, i::8].to(torch.int32) + 8) << (i * 4)

    qzeros = torch.zeros(n // 8, n_groups, dtype=torch.int32, device=zeros.device)
    for i in range(8):
        qzeros |= (zeros[i::8, :].to(torch.int32) + 8) << (i * 4)

    return qweight, qzeros


def make_asymmetric_packed_int4(n: int, k: int, group_size: int = 32):
    assert k % group_size == 0 and n % 8 == 0
    n_groups = k // group_size
    weights = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
    zeros = torch.randint(-8, 8, (n, n_groups), dtype=torch.int8, device="cuda")
    scales = torch.randn(n, n_groups, dtype=torch.float16, device="cuda") * 0.01
    qweight, qzeros = pack_asymmetric_int4(weights, zeros)
    return qweight, scales, qzeros
```

- [ ] **Step 4: Write correctness tests**

Append to `tests/test_awq_gemm_m1_specialization.py`:

```python
import pytest
import torch

from tests.tools.gemma4_e2b_quant_helpers import make_asymmetric_packed_int4
from vllm.kernels.triton.awq_fused_gemm import packed_int4_asymmetric_group32_gemv_m1
from vllm.model_executor.layers.quantization.tensor import (
    dequantize_asymmetric_packed_int4_pytorch,
)


@pytest.mark.parametrize("n,k", [(2048, 1536), (6144, 1536), (1536, 6144)])
def test_packed_int4_asymmetric_group32_gemv_m1_matches_reference_fp16(n, k):
    group_size = 32
    qweight, scales, qzeros = make_asymmetric_packed_int4(n, k, group_size)
    x = torch.randn(1, k, dtype=torch.float16, device="cuda")

    out = packed_int4_asymmetric_group32_gemv_m1(
        x, qweight, scales, qzeros, group_size=group_size
    )
    dense = dequantize_asymmetric_packed_int4_pytorch(
        qweight, scales, qzeros, group_size
    )
    ref = torch.nn.functional.linear(x, dense)
    torch.testing.assert_close(out, ref, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("n,k", [(2048, 1536)])
def test_packed_int4_asymmetric_group32_gemv_m1_matches_reference_bf16(n, k):
    group_size = 32
    qweight, scales, qzeros = make_asymmetric_packed_int4(n, k, group_size)
    x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")

    out = packed_int4_asymmetric_group32_gemv_m1(
        x, qweight, scales, qzeros, group_size=group_size
    )
    dense = dequantize_asymmetric_packed_int4_pytorch(
        qweight, scales, qzeros, group_size
    ).to(dtype=x.dtype)
    ref = torch.nn.functional.linear(x, dense)
    torch.testing.assert_close(out, ref, atol=0.03, rtol=0.03)


def test_packed_int4_asymmetric_group32_gemv_m1_rejects_non_m1():
    qweight, scales, qzeros = make_asymmetric_packed_int4(2048, 1536)
    x = torch.randn(2, 1536, dtype=torch.float16, device="cuda")
    with pytest.raises(AssertionError):
        packed_int4_asymmetric_group32_gemv_m1(x, qweight, scales, qzeros)


def test_packed_int4_asymmetric_group32_gemv_m1_real_e2b_checkpoint():
    """Load the real E2B model and test the first asymmetric LiteLinear."""
    from vllm import LLM
    from vllm.model_executor.layers.lite_linear import LiteLinear

    llm = LLM(
        model="models/gemma-4-E2B-it-AWQ-INT4",
        max_model_len=256,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    model = llm.model

    target: LiteLinear | None = None
    for module in model.modules():
        if isinstance(module, LiteLinear):
            qz = getattr(module, "qzeros", None)
            if qz is not None and qz.numel() > 1:
                target = module
                break
    assert target is not None, "no asymmetric LiteLinear found in E2B model"

    qweight = target.qweight
    scales = target.scales
    qzeros = target.qzeros
    n, k_div_8 = qweight.shape
    k = k_div_8 * 8
    x = torch.randn(1, k, dtype=torch.float16, device="cuda")

    out = packed_int4_asymmetric_group32_gemv_m1(
        x, qweight, scales, qzeros, group_size=32
    )
    dense = dequantize_asymmetric_packed_int4_pytorch(
        qweight, scales, qzeros, 32
    )
    ref = torch.nn.functional.linear(x, dense)
    torch.testing.assert_close(out, ref, atol=0.02, rtol=0.02)
```

- [ ] **Step 5: Run correctness tests**

```bash
uv run pytest tests/test_awq_gemm_m1_specialization.py -k asymmetric -v
```

Expected: PASS.

- [ ] **Step 6: Create standalone microbench tool**

Create `tests/tools/gemma4_e2b_gemv_microbench.py`:

```python
#!/usr/bin/env python3
"""Microbench asymmetric GEMV vs dense reference for E2B shapes."""
import time
import torch

from tests.tools.gemma4_e2b_quant_helpers import make_asymmetric_packed_int4
from vllm.kernels.triton.awq_fused_gemm import packed_int4_asymmetric_group32_gemv_m1
from vllm.model_executor.layers.quantization.tensor import (
    dequantize_asymmetric_packed_int4_pytorch,
)

SHAPES = [(2048, 1536), (6144, 1536), (1536, 6144)]


def bench():
    for n, k in SHAPES:
        qweight, scales, qzeros = make_asymmetric_packed_int4(n, k, 32)
        x = torch.randn(1, k, dtype=torch.float16, device="cuda")
        dense = dequantize_asymmetric_packed_int4_pytorch(qweight, scales, qzeros, 32)

        for _ in range(10):
            packed_int4_asymmetric_group32_gemv_m1(x, qweight, scales, qzeros)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(1000):
            packed_int4_asymmetric_group32_gemv_m1(x, qweight, scales, qzeros)
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - t0) / 1000 * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(1000):
            torch.nn.functional.linear(x, dense)
        torch.cuda.synchronize()
        dense_ms = (time.perf_counter() - t0) / 1000 * 1000

        print(
            f"N={n:5d} K={k:5d} fused={fused_ms:.4f}ms "
            f"dense={dense_ms:.4f}ms speedup={dense_ms/fused_ms:.2f}x"
        )


if __name__ == "__main__":
    bench()
```

- [ ] **Step 7: Run microbench**

```bash
uv run python tests/tools/gemma4_e2b_gemv_microbench.py
```

Expected: fused faster than dense.

- [ ] **Step 8: Commit**

```bash
git add vllm/kernels/triton/awq_fused_gemm.py tests/tools/gemma4_e2b_quant_helpers.py tests/tools/gemma4_e2b_gemv_microbench.py tests/test_awq_gemm_m1_specialization.py
git commit -m "feat(kernels): asymmetric group32 M=1 GEMV with tests and microbench"
```

---

## Task 3: Kernel-Policy Dispatch

**Files:**
- Modify: `vllm/model_executor/layers/quantization/tensor.py`
- Modify: `vllm/adapters/policy_keys.py`
- Modify: `vllm/adapters/gemma4.py`
- Modify: `vllm/engine/env_registry.py`
- Modify: `tests/test_project_governance.py`
- Test: `tests/test_gemma4_e2b_e4b_support.py`

**Interfaces:**
- Consumes: `x` of shape `[*, K]`, asymmetric packed-int4 weights, `config.kernel_policy["awq_asymmetric_gemv"]`.
- Produces: when policy is true, dispatch to asymmetric GEMV; otherwise continue to original symmetric/dense fallback path.

- [ ] **Step 1: Add policy key constant**

In `vllm/adapters/policy_keys.py`, add:

```python
GEMMA4_AWQ_ASYMMETRIC_GEMV = "awq_asymmetric_gemv"
```

- [ ] **Step 2: Wire env into Gemma4 adapter kernel_policy**

In `vllm/adapters/gemma4.py`, inside `runtime_policy()`:

```python
from vllm.adapters.policy_keys import GEMMA4_AWQ_ASYMMETRIC_GEMV
from vllm.utils.text_utils import truthy

asymmetric_gemv = truthy(
    runtime_config.tuning_env.get("FASTINFERENCE_AWQ_ASYMMETRIC_GEMV", "0")
)

kernel_policy = {
    # ... existing entries ...
    GEMMA4_AWQ_ASYMMETRIC_GEMV: asymmetric_gemv,
}

tuning_env_overrides = {
    # ... existing entries ...
    "FASTINFERENCE_AWQ_ASYMMETRIC_GEMV": "1" if asymmetric_gemv else "0",
}
```

- [ ] **Step 3: Add kernel-policy helper in tensor.py**

In `vllm/model_executor/layers/quantization/tensor.py`, after `awq_decode_gemv_enabled`:

```python
def awq_asymmetric_gemv_enabled(config: object | None = None) -> bool:
    return _bool_like(
        _kernel_policy_value(config, "awq_asymmetric_gemv", False),
        False,
    )
```

- [ ] **Step 4: Register env name**

In `vllm/engine/env_registry.py`, add `FASTINFERENCE_AWQ_ASYMMETRIC_GEMV` to the tool-only or deprecated env list.

- [ ] **Step 5: Update governance test allow-list**

In `tests/test_project_governance.py`, add `"FASTINFERENCE_AWQ_ASYMMETRIC_GEMV"` to the `prefixes` tuple in `test_awq_tensor_production_env_reads_are_tool_only`.

- [ ] **Step 6: Add asymmetric M=1 dispatch in matmul**

In `AWQWeight.matmul`, before existing logic:

```python
def matmul(self, x, bias=None, *, config: object | None = None):
    _awq_stat_inc("awq_matmul_calls")
    _awq_prefix_stat_inc(self.prefix, "matmul_calls")

    x2 = x.reshape(-1, x.shape[-1])
    if (
        x2.shape[0] == 1
        and self.qzeros is not None
        and int(self.group_size) == 32
        and awq_asymmetric_gemv_enabled(config)
        and _is_packed_int4_asymmetric_layout(
            self.qweight, self.scales, self.qzeros, int(self.group_size)
        )
        and not getattr(self, "_asymmetric_gemv_disabled", False)
    ):
        from vllm.kernels.triton.awq_fused_gemm import (
            packed_int4_asymmetric_group32_gemv_m1,
        )
        try:
            out = packed_int4_asymmetric_group32_gemv_m1(
                x2, self.qweight, self.scales, self.qzeros,
                group_size=int(self.group_size),
            )
            if bias is not None:
                out = out + bias
            _awq_stat_inc("awq_asymmetric_gemv_success")
            _awq_prefix_stat_inc(self.prefix, "asymmetric_gemv_success")
            return out.view(*x.shape[:-1], out.shape[-1])
        except Exception as exc:
            self._asymmetric_gemv_disabled = True
            record_awq_audit_event(
                self.prefix,
                "asymmetric_gemv_m1_exception_fallback",
                shape=_shape_for_matmul(x, self.qweight, int(self.group_size)),
                dtypes=_dtypes_for_quant_matmul(
                    x, self.qweight, self.scales, self.qzeros
                ),
                reason=type(exc).__name__,
            )

    # Existing symmetric fused / dense fallback logic continues unchanged.
    ...
```

- [ ] **Step 7: Add dispatch test with audit-counter assertion**

Append to `tests/test_gemma4_e2b_e4b_support.py`:

```python
import torch

from tests.tools.gemma4_e2b_quant_helpers import make_asymmetric_packed_int4
from vllm.model_executor.layers.quantization.tensor import (
    AWQWeight,
    dequantize_asymmetric_packed_int4_pytorch,
    get_awq_runtime_stats,
    reset_awq_runtime_stats,
)


def test_e2b_asymmetric_matmul_dispatches_to_gemv():
    qweight, scales, qzeros = make_asymmetric_packed_int4(2048, 1536)
    w = AWQWeight(qweight, scales, qzeros, group_size=32, prefix="test")
    x = torch.randn(1, 1, 1536, dtype=torch.float16, device="cuda")

    class FakeConfig:
        kernel_policy = {"awq_asymmetric_gemv": True}

    reset_awq_runtime_stats()
    out = w.matmul(x, config=FakeConfig())

    dense = dequantize_asymmetric_packed_int4_pytorch(qweight, scales, qzeros, 32)
    ref = torch.nn.functional.linear(x, dense)
    torch.testing.assert_close(out, ref, atol=0.02, rtol=0.02)

    stats = get_awq_runtime_stats()
    assert stats.get("awq_asymmetric_gemv_success", 0) >= 1, stats
```

- [ ] **Step 8: Run tests**

```bash
uv run pytest tests/test_gemma4_e2b_e4b_support.py::test_e2b_asymmetric_matmul_dispatches_to_gemv tests/test_project_governance.py::test_awq_tensor_production_env_reads_are_tool_only -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add vllm/model_executor/layers/quantization/tensor.py vllm/adapters/policy_keys.py vllm/adapters/gemma4.py vllm/engine/env_registry.py tests/test_project_governance.py tests/test_gemma4_e2b_e4b_support.py
git commit -m "feat(quant): kernel-policy dispatch for asymmetric M=1 GEMV"
```

---

## Task 4: Whole-Model A/B Gate Across Context Buckets with Bit-Exact Check

**Files:**
- Modify: `tests/tools/gemma4_e2b_audit.py` (already created in Task 1)
- Create: `tests/tools/gemma4_e2b_a_b_compare.py`
- Test: manual run.

**Interfaces:**
- Consumes: audit tool with `--kernel-policy` flag; separate processes for baseline and optimized.
- Produces: aggregated speedup across 128/512/2048 context buckets plus bit-exact token comparison.

- [ ] **Step 1: Run baseline across three buckets**

```bash
for ctx in 128 512 2048; do
  uv run python tests/tools/gemma4_e2b_audit.py \
    --context-bucket $ctx \
    --kernel-policy '{"awq_asymmetric_gemv": false}' \
    --out /tmp/e2b_baseline_${ctx}.json
done
```

- [ ] **Step 2: Run optimized across three buckets**

```bash
for ctx in 128 512 2048; do
  uv run python tests/tools/gemma4_e2b_audit.py \
    --context-bucket $ctx \
    --kernel-policy '{"awq_asymmetric_gemv": true}' \
    --out /tmp/e2b_optimized_${ctx}.json
done
```

- [ ] **Step 3: Compare speedup and token IDs**

Create `tests/tools/gemma4_e2b_a_b_compare.py`:

```python
#!/usr/bin/env python3
import json
import sys
from pathlib import Path

BUCKETS = [128, 512, 2048]


def main() -> None:
    results = []
    bit_exact = True
    for ctx in BUCKETS:
        b = json.loads(Path(f"/tmp/e2b_baseline_{ctx}.json").read_text())
        o = json.loads(Path(f"/tmp/e2b_optimized_{ctx}.json").read_text())
        speedup = o["perf_run"]["median_decode_tps"] / b["perf_run"]["median_decode_tps"]
        tokens_match = o["perf_run"]["token_ids"] == b["perf_run"]["token_ids"]
        bit_exact = bit_exact and tokens_match
        results.append({
            "context": ctx,
            "baseline_tps": b["perf_run"]["median_decode_tps"],
            "optimized_tps": o["perf_run"]["median_decode_tps"],
            "speedup": speedup,
            "tokens_match": tokens_match,
        })
        print(
            f"ctx={ctx:4d} baseline={b['perf_run']['median_decode_tps']:.3f} "
            f"optimized={o['perf_run']['median_decode_tps']:.3f} "
            f"speedup={speedup:.3f}x tokens_match={tokens_match}"
        )

    median_speedup = sorted(r["speedup"] for r in results)[len(results) // 2]
    min_speedup = min(r["speedup"] for r in results)
    print(f"median_speedup={median_speedup:.3f}x min_speedup={min_speedup:.3f}x bit_exact={bit_exact}")

    passed = (
        bit_exact
        and median_speedup >= 1.20
        and min_speedup >= 1.00
    )
    print(f"A/B gate PASSED={passed}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

Run:

```bash
uv run python tests/tools/gemma4_e2b_a_b_compare.py
```

Gate:

- `bit_exact` must be true.
- `median_speedup >= 1.20`.
- `min_speedup >= 1.00` (no bucket regresses).
- If PASS: document result and consider next steps.
- If FAIL: stop. Do not implement QKV fusion, gate-up fusion, PLE streaming, or lm_head fast path.

- [ ] **Step 4: Commit comparison helper**

```bash
git add tests/tools/gemma4_e2b_a_b_compare.py
git commit -m "feat(tools): E2B asymmetric GEMV A/B comparison with bit-exact gate"
```

---

## Explicitly Out of Scope

The following are **not** in this plan and should only be reconsidered if Task 4 passes:

- Fused asymmetric QKV.
- Fused asymmetric gate-up activation.
- PLE streaming kernel.
- Greedy lm_head fast path.
- Engine-level speculative decoding integration.

---

## Self-Review

**1. Spec coverage:**

| Review finding | Fix in this plan |
|---|---|
| Layer profile produces no data | Removed layer profile entirely; P0 uses AWQ audit and cache_bytes only. |
| Warmup resets fallback evidence | Split into cold audit (no reset) and perf run (warmup + reset). |
| Missing whole-model correctness gate | Task 4 compares `token_ids` bit-exact and exits nonzero on mismatch. |
| Kernel test coverage | Added FP16, BF16, M!=1 rejection, and real E2B checkpoint tests. |

**2. Placeholder scan:**

- No "TBD", "TODO", "implement later".
- Every code step contains full function bodies and exact file paths.
- Gate values are numeric.

**3. Type consistency:**

- `awq_asymmetric_gemv_enabled` follows existing `_bool_like(_kernel_policy_value(...))` pattern.
- Adapter policy key constant matches kernel_policy string.
- Audit tool passes policy through `fastinference_config.tuning_keyvals`.
- A/B comparison script reads the same JSON schema the audit tool writes.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-11-gemma4-e2b-asymmetric-int4-gemv.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
