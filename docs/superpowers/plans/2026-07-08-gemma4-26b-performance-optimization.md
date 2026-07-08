# Gemma4-26B-A4B 推理性能优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过 P0 profiling、P1 MoE int4 decode 优化、P2 batched AWQ decode kernel（仅在 microbench 证明收益后），将 Gemma4-26B-A4B 的 BS=1 decode 从 11.4 tok/s 提升至 15 tok/s，并消除 BS>1 的灾难性退化。

**Architecture:** 在现有 `vllm/model_executor/models/gemma4/` 和 `vllm/kernels/triton/` 路径上增量优化，不改动 LiteEngine 调度语义；所有新 kernel 必须有离线 microbench 和 PyTorch reference 正确性测试，新 tuning 参数通过 `RuntimeConfig` / TOML 暴露。

**Tech Stack:** Python 3.12, PyTorch ROCm, Triton (via `vllm/triton_utils`), `uv`, pytest.

## Global Constraints

- Python 3.12 only；使用 `uv run` 执行所有命令。
- 所有新 Triton kernel 必须包含 ASCII 注释描述 memory layout 和 thread/block tiling（AGENTS.md HPC 规则）。
- Runtime 代码禁止新增 `os.environ` 读取；最终 tuning 参数通过 `RuntimeConfig` / TOML `[tuning_keyvals]` 暴露。`tests/tools/` 和 A/B 实验脚本可为 subprocess 设置环境变量。
- 禁止引入 C++ 代码。
- 任何 kernel 数值变更必须通过 `tests/run_regression_suite.sh` 和 `SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh`（Gemma4-26B 部分）。
- 优先保持 M=1 路径不变；batched kernel 失败时自动 fallback 到现有路径。
- 不因为模型目录存在就自动运行重型 e2e 测试；此类测试受显式环境变量控制。

---

## File Structure

| 文件 | 责任 |
|---|---|
| `tests/tools/gemma4_26b_profile.py` | P0：26B 专属端到端 profile 工具，产出 kernel_stats.csv + AWQ audit log + baseline JSON；section timer 汇总从 harness stdout/artifact 人工提取 |
| `tests/test_gemma4_26b_fastpath_audit.py` | P0：轻量 AWQ audit log 解析器单元测试（不加载模型） |
| `benchmarks/kernels/benchmark_gemma4_moe_26b.py` | P1：26B 形状 MoE int4 decode microbench，扫策略和 tile |
| `tests/test_gemma4_moe_26b_tile.py` | P1：验证 MoE int4 kernel 在 26B 形状下的数值正确性（对比 reference） |
| `benchmarks/kernels/benchmark_awq_fused_gemm_batched.py` | P2 决策门：离线 microbench，评估 M=2/M=4 batched AWQ 是否值得做 |
| `docs/superpowers/specs/2026-07-08-gemma4-26b-performance-optimization-design.md` | 设计文档（已存在） |

---

## Out of Scope

- **P3 speculative decoding**：完全不属于本计划范围。不添加任何占位符代码，不参与 P0–P2 完成判断。若 P0–P2 完成后 BS=1 仍不达标，由独立计划评估 n-gram / prompt-lookup draft。
- **Task 4 router/top-k 优化**：暂不实施。除非 P0 profiling 证明 router 是真实热点，否则不引入每个 token 的检查。
- **Model-side M=2/M=4 wiring**：暂不实施。仅在 `benchmark_awq_fused_gemm_batched.py` 证明 M=2/M=4 有收益后，再设计 kernel API 并连接模型。

---

## Task 1: P0 Profiling Harness for Gemma4-26B

**Files:**
- Create: `tests/tools/gemma4_26b_profile.py`
- Test: `tests/test_gemma4_26b_profile_harness.py`（仅测试 harness 结构，不运行 e2e）

**Interfaces:**
- Consumes: `subprocess.run`, `rocprofv3`, `tests/e2e_full_benchmark.py`
- Produces: `/tmp/gemma26b_profile/kernel_stats.csv`, `/tmp/gemma26b_profile/awq_audit.log`, `/tmp/gemma26b_profile/baseline.json`, `/tmp/gemma26b_profile/summary.json`; section timer 汇总从 harness stdout/artifact 人工提取

- [ ] **Step 1: Write the harness script**

Create `tests/tools/gemma4_26b_profile.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""One-stop profiling harness for Gemma4-26B-A4B on Radeon 8060S."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], env: dict[str, str] | None = None, capture: bool = False) -> str:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        text=True,
        capture_output=capture,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed with rc={result.returncode}")
    return result.stdout if capture else ""


def run_e2e_baseline(out_dir: Path, max_new_tokens: int = 24) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "baseline.json"
    env = os.environ.copy()
    env.update({
        "FASTINFERENCE_KV_TYPE": "turbo_int4",
        "FASTINFERENCE_GEMMA4_ALLOW_INT4_KV": "0",
    })
    _run([
        sys.executable, "tests/e2e_full_benchmark.py",
        "--models", "gemma4_26b_a4b",
        "--gemma26b-concurrent", "1",
        "--gemma26b-max-new-tokens", str(max_new_tokens),
        "--json-out", str(json_path),
    ], env=env)
    return json_path


def run_rocprof(out_dir: Path, max_new_tokens: int = 8) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rocprof.json"
    env = os.environ.copy()
    env.update({
        "FASTINFERENCE_KV_TYPE": "turbo_int4",
        "FASTINFERENCE_GEMMA4_ALLOW_INT4_KV": "0",
    })
    _run([
        "rocprofv3", "--kernel-trace", "--stats",
        "--output-dir", str(out_dir / "rocprof"),
        "--", sys.executable, "tests/e2e_full_benchmark.py",
        "--models", "gemma4_26b_a4b",
        "--gemma26b-max-new-tokens", str(max_new_tokens),
        "--json-out", str(json_path),
    ], env=env)
    return out_dir / "rocprof" / "kernel_stats.csv"


def run_awq_audit(out_dir: Path, max_new_tokens: int = 24) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "audit.json"
    log_path = out_dir / "awq_audit.log"
    env = os.environ.copy()
    env.update({
        "FASTINFERENCE_KV_TYPE": "turbo_int4",
        "FASTINFERENCE_GEMMA4_ALLOW_INT4_KV": "0",
        "FASTINFERENCE_AWQ_AUDIT": "1",
    })
    stdout = _run([
        sys.executable, "tests/e2e_full_benchmark.py",
        "--models", "gemma4_26b_a4b",
        "--gemma26b-concurrent", "1",
        "--gemma26b-max-new-tokens", str(max_new_tokens),
        "--json-out", str(json_path),
    ], env=env, capture=True)
    log_path.write_text(stdout, encoding="utf-8")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/gemma26b_profile")
    parser.add_argument("--rocprof-max-new-tokens", type=int, default=8)
    parser.add_argument("--baseline-max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    baseline_json = run_e2e_baseline(out_dir, args.baseline_max_new_tokens)
    kernel_stats_csv = run_rocprof(out_dir, args.rocprof_max_new_tokens)
    audit_log = run_awq_audit(out_dir, args.baseline_max_new_tokens)

    summary = {
        "baseline_json": str(baseline_json),
        "kernel_stats_csv": str(kernel_stats_csv),
        "awq_audit_log": str(audit_log),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Profile complete.")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write a unit test for the harness structure**

Create `tests/test_gemma4_26b_profile_harness.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from tests.tools.gemma4_26b_profile import run_e2e_baseline


def test_harness_imports() -> None:
    # Guard only: the harness must be importable and its helper signatures
    # must match what the tool expects. Heavy e2e execution is opt-in only.
    assert callable(run_e2e_baseline)
```

- [ ] **Step 3: Run the unit test**

```bash
uv run pytest tests/test_gemma4_26b_profile_harness.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/tools/gemma4_26b_profile.py tests/test_gemma4_26b_profile_harness.py
git commit -m "feat(tools): add Gemma4-26B profiling harness"
```

---

## Task 2: P0 Fast-Path Coverage Audit

**Files:**
- Create: `tests/test_gemma4_26b_fastpath_audit.py`

**Interfaces:**
- Consumes: AWQ audit log format, regex parsing
- Produces: Structured coverage report dictionary

**Scope note:** This task provides a brittle log-parser starting point only. The authoritative fast-path coverage comes from the P0 harness artifacts (section timer + AWQ audit log produced by `tests/tools/gemma4_26b_profile.py`).

- [ ] **Step 1: Write the audit parser and its unit test**

Create `tests/test_gemma4_26b_fastpath_audit.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

from tests.tools.gemma4_26b_profile import run_awq_audit


def _parse_awq_audit_log(log_text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in log_text.splitlines():
        if "qkv_fused_decode" in line:
            counts["qkv_fused_decode"] = counts.get("qkv_fused_decode", 0) + 1
        elif "qkv_separate_decode" in line:
            counts["qkv_separate_decode"] = counts.get("qkv_separate_decode", 0) + 1
        elif "moe_int4_decode_used" in line:
            counts["moe_int4_decode_used"] = counts.get("moe_int4_decode_used", 0) + 1
        elif "moe_int4_decode_fallback" in line:
            counts["moe_int4_decode_fallback"] = counts.get("moe_int4_decode_fallback", 0) + 1
    return counts


def test_parse_awq_audit_log() -> None:
    sample = (
        "event=qkv_fused_decode\n"
        "event=qkv_fused_decode\n"
        "event=qkv_separate_decode\n"
        "event=moe_int4_decode_used\n"
    )
    result = _parse_awq_audit_log(sample)
    assert result["qkv_fused_decode"] == 2
    assert result["qkv_separate_decode"] == 1
    assert result["moe_int4_decode_used"] == 1
    assert result.get("moe_int4_decode_fallback", 0) == 0
```

- [ ] **Step 2: Run the parser test**

```bash
uv run pytest tests/test_gemma4_26b_fastpath_audit.py::test_parse_awq_audit_log -v
```

Expected: PASS.

- [ ] **Step 3: Add opt-in e2e coverage assertion**

Append to `tests/test_gemma4_26b_fastpath_audit.py`:

```python
import os
import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_GEMMA4_26B_PERF") != "1",
    reason="Set RUN_GEMMA4_26B_PERF=1 to run heavy model load tests",
)
@pytest.mark.skipif(
    not Path("models/gemma-4-26B-A4B-it-AWQ-4bit").exists(),
    reason="Gemma4-26B model not present",
)
def test_26b_fastpath_coverage(tmp_path: Path) -> None:
    log_path = run_awq_audit(tmp_path, max_new_tokens=4)
    log_text = log_path.read_text(encoding="utf-8")
    counts = _parse_awq_audit_log(log_text)
    # In default BS=1 decode, fused QKV and MoE int4 should dominate.
    # NOTE: this parser is a starting point only; authoritative coverage is
    # derived from the P0 harness artifacts, not from this brittle regex.
    assert counts.get("qkv_fused_decode", 0) > counts.get("qkv_separate_decode", 0)
    assert counts.get("moe_int4_decode_used", 0) > 0
```

- [ ] **Step 4: Run both tests**

```bash
uv run pytest tests/test_gemma4_26b_fastpath_audit.py -v
```

Expected: `test_parse_awq_audit_log` PASS；`test_26b_fastpath_coverage` SKIP（未设置 env）。

- [ ] **Step 5: Commit**

```bash
git add tests/test_gemma4_26b_fastpath_audit.py
git commit -m "test(gemma4): add 26B fast-path coverage audit (opt-in)"
```

---

## Task 3: P1 MoE INT4 Decode Microbench and Correctness for 26B Shapes

**Files:**
- Create: `benchmarks/kernels/benchmark_gemma4_moe_26b.py`
- Create: `tests/test_gemma4_moe_26b_tile.py`
- Read: `vllm/kernels/triton/gemma4_moe_int4.py`

**Interfaces:**
- Consumes: `gemma4_moe_int4_decode*` kernel family
- Produces: `benchmark_gemma4_moe_26b.py` CSV report; numerical correctness test PASS/FAIL

- [ ] **Step 1: Write the microbench script**

Create `benchmarks/kernels/benchmark_gemma4_moe_26b.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark Gemma4-26B MoE int4 decode kernels."""
from __future__ import annotations

import argparse
import csv
import time

import torch

from vllm.kernels.triton.gemma4_moe_int4 import (
    gemma4_moe_int4_decode,
    gemma4_moe_int4_decode_batched,
    gemma4_moe_int4_decode_batched_chunked,
    gemma4_moe_int4_decode_batched_grouped,
    gemma4_moe_int4_decode_batched_tuned,
)


def _reference_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    qweight_gu: torch.Tensor,
    scales_gu: torch.Tensor,
    qweight_d: torch.Tensor,
    scales_d: torch.Tensor,
    intermediate_dim: int,
) -> torch.Tensor:
    """PyTorch reference: dequantize int4 -> matmul per expert -> reduce."""
    from vllm.model_executor.layers.quantization.tensor import (
        dequantize_symmetric_packed_int4_pytorch,
    )

    m, hidden = x.shape
    top_k = topk_ids.shape[1]
    out = torch.zeros(m, hidden, dtype=x.dtype, device=x.device)
    for tok in range(m):
        for k in range(top_k):
            eid = int(topk_ids[tok, k])
            w1e = dequantize_symmetric_packed_int4_pytorch(
                qweight_gu[eid:eid + 1].to(torch.int32),
                scales_gu[eid:eid + 1],
                group_size=hidden // scales_gu.shape[2],
            )[0, : 2 * intermediate_dim, :hidden].to(x.dtype)
            w2e = dequantize_symmetric_packed_int4_pytorch(
                qweight_d[eid:eid + 1].to(torch.int32),
                scales_d[eid:eid + 1],
                group_size=intermediate_dim // scales_d.shape[2],
            )[0, :hidden, :intermediate_dim].to(x.dtype)

            h = x[tok : tok + 1]
            gu = torch.matmul(h, w1e.t())
            g, u = gu.chunk(2, dim=-1)
            h = torch.nn.functional.silu(g) * u
            y = torch.matmul(h, w2e.t()) * topk_weights[tok, k]
            out[tok] += y.squeeze(0)
    return out


def benchmark_strategy(
    strategy: str,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    batch_sizes: list[int],
    device: torch.device,
    check_correctness: bool = False,
) -> list[dict[str, float]]:
    kernel_map = {
        "two_stage": gemma4_moe_int4_decode,
        "batched": gemma4_moe_int4_decode_batched,
        "batched_chunked": gemma4_moe_int4_decode_batched_chunked,
        "batched_grouped": gemma4_moe_int4_decode_batched_grouped,
        "batched_tuned": gemma4_moe_int4_decode_batched_tuned,
    }
    kernel = kernel_map[strategy]
    results: list[dict[str, float]] = []

    qweight_gu = torch.randint(
        0, 16, (num_experts, 2 * intermediate_dim, hidden_dim // 8),
        dtype=torch.int32, device=device,
    )
    scales_gu = torch.randn(
        num_experts, 2 * intermediate_dim, hidden_dim // 32,
        dtype=torch.bfloat16, device=device,
    )
    qweight_d = torch.randint(
        0, 16, (num_experts, hidden_dim, intermediate_dim // 8),
        dtype=torch.int32, device=device,
    )
    scales_d = torch.randn(
        num_experts, hidden_dim, intermediate_dim // 32,
        dtype=torch.bfloat16, device=device,
    )

    for m in batch_sizes:
        x = torch.randn(m, hidden_dim, dtype=torch.bfloat16, device=device)
        topk_ids = torch.randint(0, num_experts, (m, top_k), device=device)
        topk_weights = torch.randn(m, top_k, dtype=torch.bfloat16, device=device)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out, used, _ = kernel(
            x, topk_weights, topk_ids,
            qweight_gu, scales_gu, qweight_d, scales_d,
            intermediate_dim=intermediate_dim,
            activation="silu",
        )
        if not used:
            continue

        max_err = float("nan")
        if check_correctness:
            ref = _reference_moe(
                x, topk_ids, topk_weights,
                qweight_gu, scales_gu, qweight_d, scales_d,
                intermediate_dim,
            )
            max_err = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
            if max_err > 0.1:
                print(f"WARN {strategy} M={m} max_err={max_err}")

        torch.cuda.synchronize()
        start = time.perf_counter()
        iterations = 100 if m == 1 else 50
        for _ in range(iterations):
            out, used, _ = kernel(
                x, topk_weights, topk_ids,
                qweight_gu, scales_gu, qweight_d, scales_d,
                intermediate_dim=intermediate_dim,
                activation="silu",
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        results.append({
            "strategy": strategy,
            "m": m,
            "time_ms": elapsed * 1000 / iterations,
            "used": int(used),
            "max_err": max_err,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2816)
    parser.add_argument("--intermediate", type=int, default=704)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--out", default="/tmp/gemma26b_moe_microbench.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    strategies = ["two_stage", "batched", "batched_chunked", "batched_grouped", "batched_tuned"]

    all_results: list[dict[str, float]] = []
    for strategy in strategies:
        all_results.extend(benchmark_strategy(
            strategy, args.hidden, args.intermediate,
            args.num_experts, args.top_k, batch_sizes, device,
            check_correctness=args.check_correctness,
        ))

    out_path = args.out
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "m", "time_ms", "used", "max_err"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write numerical correctness test**

Create `tests/test_gemma4_moe_26b_tile.py`:

```python
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.kernels.triton.gemma4_moe_int4 import gemma4_moe_int4_decode_batched_chunked
from vllm.model_executor.layers.quantization.tensor import (
    dequantize_symmetric_packed_int4_pytorch,
)


def _reference_moe_single(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    qweight_gu: torch.Tensor,
    scales_gu: torch.Tensor,
    qweight_d: torch.Tensor,
    scales_d: torch.Tensor,
    intermediate_dim: int,
) -> torch.Tensor:
    m, hidden = x.shape
    top_k = topk_ids.shape[1]
    out = torch.zeros(m, hidden, dtype=x.dtype, device=x.device)
    for tok in range(m):
        for k in range(top_k):
            eid = int(topk_ids[tok, k])
            gsz_gu = hidden // scales_gu.shape[2]
            gsz_d = intermediate_dim // scales_d.shape[2]
            w1e = dequantize_symmetric_packed_int4_pytorch(
                qweight_gu[eid:eid + 1].to(torch.int32),
                scales_gu[eid:eid + 1],
                group_size=gsz_gu,
            )[0, : 2 * intermediate_dim, :hidden].to(x.dtype)
            w2e = dequantize_symmetric_packed_int4_pytorch(
                qweight_d[eid:eid + 1].to(torch.int32),
                scales_d[eid:eid + 1],
                group_size=gsz_d,
            )[0, :hidden, :intermediate_dim].to(x.dtype)

            gu = torch.matmul(x[tok : tok + 1], w1e.t())
            g, u = gu.chunk(2, dim=-1)
            h = torch.nn.functional.silu(g) * u
            y = torch.matmul(h, w2e.t()) * topk_weights[tok, k]
            out[tok] += y.squeeze(0)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_moe_int4_numerical_small_shape() -> None:
    # Use a small shape for fast reference comparison. The microbench script
    # uses the real 26B shape (hidden=2816, intermediate=704); this test only
    # verifies numerical correctness of the kernel path.
    device = torch.device("cuda")
    hidden_dim = 256
    intermediate_dim = 64
    num_experts = 16
    top_k = 4
    m = 2

    torch.manual_seed(42)
    x = torch.randn(m, hidden_dim, dtype=torch.bfloat16, device=device)
    topk_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=device)
    topk_weights = torch.full((m, top_k), 1.0 / top_k, dtype=torch.bfloat16, device=device)

    qweight_gu = torch.randint(
        0, 16, (num_experts, 2 * intermediate_dim, hidden_dim // 8),
        dtype=torch.int32, device=device,
    )
    scales_gu = torch.randn(
        num_experts, 2 * intermediate_dim, hidden_dim // 32,
        dtype=torch.bfloat16, device=device,
    )
    qweight_d = torch.randint(
        0, 16, (num_experts, hidden_dim, intermediate_dim // 8),
        dtype=torch.int32, device=device,
    )
    scales_d = torch.randn(
        num_experts, hidden_dim, intermediate_dim // 32,
        dtype=torch.bfloat16, device=device,
    )

    out, used, reason = gemma4_moe_int4_decode_batched_chunked(
        x, topk_weights, topk_ids,
        qweight_gu, scales_gu, qweight_d, scales_d,
        intermediate_dim=intermediate_dim,
        activation="silu",
    )
    assert used, f"Fast path not used: {reason}"
    assert out.shape == (m, hidden_dim)

    ref = _reference_moe_single(
        x, topk_ids, topk_weights,
        qweight_gu, scales_gu, qweight_d, scales_d,
        intermediate_dim,
    )
    max_err = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    assert max_err < 0.1, f"Numerical mismatch: max_err={max_err}"
```

- [ ] **Step 3: Run the correctness test**

```bash
uv run pytest tests/test_gemma4_moe_26b_tile.py::test_moe_int4_numerical_small_shape -v
```

Expected: PASS on GPU，SKIP on CPU-only host。

- [ ] **Step 4: Run the microbench**

```bash
# Default: performance only (fast)
uv run python benchmarks/kernels/benchmark_gemma4_moe_26b.py

# Optional: also verify full-shape numerical correctness (slower)
uv run python benchmarks/kernels/benchmark_gemma4_moe_26b.py --check-correctness
```

Expected: CSV written to `/tmp/gemma26b_moe_microbench.csv` with all strategies. `max_err` is `nan` unless `--check-correctness` is passed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/kernels/benchmark_gemma4_moe_26b.py tests/test_gemma4_moe_26b_tile.py
git commit -m "feat(kernels): add Gemma4-26B MoE int4 microbench + numerical correctness test"
```

---

## Task 4: P2 Baseline Only — Measure M=1 AWQ Fast Path for Future Batched Decision Gate

**Files:**
- Create: `benchmarks/kernels/benchmark_awq_fused_gemm_batched.py`

**Interfaces:**
- Consumes: Existing `packed_int4_symmetric_fused_qkv_m1_safe` and `packed_int4_symmetric_fused_gate_up_m1_safe`
- Produces: M=1 baseline throughput artifact for future M=2/M=4 comparison

**Scope note:** This task only establishes the M=1 baseline. It does not implement or decide on batched kernels. A future plan will extend this microbench with M=2/M=4 variants and compare against the baseline before any model-side wiring is changed.

- [ ] **Step 1: Write the microbench for M=1 baseline only**

Create `benchmarks/kernels/benchmark_awq_fused_gemm_batched.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Baseline microbench for AWQ fused QKV / gate-up.

This tool measures the M=1 fast path first. Future work will add M=2/M=4
batched variants here and compare against the M=1 baseline before any
model-side wiring is changed.
"""
from __future__ import annotations

import argparse
import csv
import time

import torch

from vllm.kernels.triton.awq_fused_gemm import (
    packed_int4_symmetric_fused_gate_up_m1_safe,
    packed_int4_symmetric_fused_qkv_m1_safe,
)


def _make_qkv_weights(
    hidden: int,
    q_size: int,
    kv_size: int,
    group_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    q_qweight = torch.randint(0, 16, (q_size, hidden // 8), dtype=torch.int32, device=device)
    q_scales = torch.randn(q_size, hidden // group_size, dtype=torch.bfloat16, device=device)
    k_qweight = torch.randint(0, 16, (kv_size, hidden // 8), dtype=torch.int32, device=device)
    k_scales = torch.randn(kv_size, hidden // group_size, dtype=torch.bfloat16, device=device)
    v_qweight = torch.randint(0, 16, (kv_size, hidden // 8), dtype=torch.int32, device=device)
    v_scales = torch.randn(kv_size, hidden // group_size, dtype=torch.bfloat16, device=device)
    return q_qweight, q_scales, k_qweight, k_scales, v_qweight, v_scales


def _bench_qkv_m1(
    hidden: int,
    q_size: int,
    kv_size: int,
    group_size: int,
    device: torch.device,
) -> dict[str, float]:
    x = torch.randn(1, hidden, dtype=torch.bfloat16, device=device)
    weights = _make_qkv_weights(hidden, q_size, kv_size, group_size, device)
    out, used, reason = packed_int4_symmetric_fused_qkv_m1_safe(x, *weights, group_size)
    assert used, f"QKV M=1 not used: {reason}"

    torch.cuda.synchronize()
    start = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        out, _, _ = packed_int4_symmetric_fused_qkv_m1_safe(x, *weights, group_size)
    torch.cuda.synchronize()
    return {
        "kernel": "qkv_m1",
        "m": 1,
        "time_ms": (time.perf_counter() - start) * 1000 / iterations,
    }


def _bench_gate_up_m1(
    hidden: int,
    intermediate: int,
    group_size: int,
    device: torch.device,
) -> dict[str, float]:
    x = torch.randn(1, hidden, dtype=torch.bfloat16, device=device)
    qwg = torch.randint(0, 16, (intermediate, hidden // 8), dtype=torch.int32, device=device)
    gate_scales = torch.randn(intermediate, hidden // group_size, dtype=torch.bfloat16, device=device)
    qwu = torch.randint(0, 16, (intermediate, hidden // 8), dtype=torch.int32, device=device)
    up_scales = torch.randn(intermediate, hidden // group_size, dtype=torch.bfloat16, device=device)
    out, used, reason = packed_int4_symmetric_fused_gate_up_m1_safe(
        x, qwg, qwu, gate_scales, up_scales, group_size, activation="silu",
    )
    assert used, f"Gate-up M=1 not used: {reason}"

    torch.cuda.synchronize()
    start = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        out, _, _ = packed_int4_symmetric_fused_gate_up_m1_safe(
            x, qwg, qwu, gate_scales, up_scales, group_size, activation="silu",
        )
    torch.cuda.synchronize()
    return {
        "kernel": "gate_up_m1",
        "m": 1,
        "time_ms": (time.perf_counter() - start) * 1000 / iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2816)
    parser.add_argument("--q-size", type=int, default=2816)
    parser.add_argument("--kv-size", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=704)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--out", default="/tmp/gemma26b_awq_baseline_microbench.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [
        _bench_qkv_m1(args.hidden, args.q_size, args.kv_size, args.group_size, device),
        _bench_gate_up_m1(args.hidden, args.intermediate, args.group_size, device),
    ]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kernel", "m", "time_ms"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the baseline microbench**

```bash
uv run python benchmarks/kernels/benchmark_awq_fused_gemm_batched.py
```

Expected: CSV written to `/tmp/gemma26b_awq_baseline_microbench.csv`.

- [ ] **Step 3: Document the baseline artifact**

Append a note to `docs/superpowers/notes/2026-07-08-gemma4-26b-p2-baseline.md` (create the directory if needed):

> M=1 AWQ baseline for QKV and gate-up recorded at `/tmp/gemma26b_awq_baseline_microbench.csv`. A future batched-AWQ plan will extend this microbench with M=2/M=4 variants and proceed to kernel implementation **only if** batched throughput >= 80% of 2× M=1 throughput. No model-side wiring changes until then.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/kernels/benchmark_awq_fused_gemm_batched.py
git commit -m "feat(kernels): add M=1 AWQ baseline microbench for P2 gate"
```

---

## Self-Review

### Spec Coverage

| Spec Section | Implementing Task |
|---|---|
| P0 profiling + fixed commands + artifacts | Task 1 |
| P0 fast-path coverage audit | Task 2 |
| P1 MoE strategy confirmation + tile tuning | Task 3 |
| P2 batched AWQ baseline artifact | Task 4 |
| P3 speculative viability | Out of scope |
| Router/top-k optimization | Deferred until P0 proves hotspot |

### Placeholder Scan

- No "TBD" / "TODO" / placeholder API left.
- No model-side M=2/M=4 wiring until kernel is proven.
- No speculative decoding placeholder code.

### Type Consistency

- `_reference_moe` and `_reference_moe_single` use the same dequantization helper.
- All kernel calls match existing signatures.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-08-gemma4-26b-performance-optimization.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

**Which approach would you like to use?**
