# FastInference P0 性能优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Source report:** `docs/performance_evaluation_gemma4_deepseek_2026_07_07.md`
> **Plan date:** 2026-07-07

**Goal:** 根据性能评估报告的 P0 结论，固化 DeepSeek V4 Flash kept-path warm-cache 回归 gate，并验证/启用 Gemma4 26B/31B batch=2 真正并行 decode。

**Architecture:** 不改动 DeepSeek 内核（graph/capture 已排除），仅在测试/回归层加 gate；Gemma4 先通过 env override 验证 batch=2 并行收益，确认后再修改默认 recommended env。

**Tech Stack:** Python 3.12, `uv`, ROCm/PyTorch, Triton (via `vllm/triton_utils`), bash smoke scripts.

## Global Constraints

- Python 3.12 only; use `uv run` for all commands.
- No C++/CUDA extensions; all kernels remain Triton.
- No speculative decoding or graph/capture in this plan.
- All changes must pass `bash tests/run_regression_suite.sh` (fast unit/smoke, no GPU model loads).
- GPU-involving changes must be validated on Radeon 8060S 96 GB unified memory.
- Conventional commit subjects (`feat:`, `fix:`, `fix(kernels):`); `Signed-off-by` appended by hook.

---

## File Map

| File | Responsibility | Change Type |
|---|---|---|
| `tests/e2e_full_benchmark.py` | DeepSeek e2e benchmark wrapper; builds smoke command and env | Modify: pass `--min-steady-decode-tps` to smoke tool; optionally expose `--deepseek-min-decode-tps` CLI arg |
| `tests/run_deepseek_v4_flash_real_smoke.sh` | Real-GPU smoke entrypoint for DeepSeek | Modify: add cold-cache + warm-cache gate commands |
| `tests/e2e_full_benchmark.py` (Gemma4 `ModelSpec` env) | Gemma4 26B/31B default recommended env (`_GEMMA4_26B_RECOMMENDED_ENV`, `_GEMMA4_31B_RECOMMENDED_ENV`) | Conditional modify: remove or raise `FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS` after Task 3 verification succeeds |
| `docs/performance_evaluation_gemma4_deepseek_2026_07_07.md` | Decision doc | Update P0 status after each task lands |

---

## Task 1: DeepSeek e2e warm-cache gate

**Files:**
- Modify: `tests/e2e_full_benchmark.py:2853-2864` (smoke command building inside `_run_deepseek_v4_flash_direct_benchmark`)
- Test: run e2e benchmark for DeepSeek

**Interfaces:**
- Consumes: existing `_run_deepseek_v4_flash_direct_benchmark` builds `[sys.executable, script, --model, ...]` and runs subprocess.
- Produces: smoke command includes `--min-steady-decode-tps 1.5`; subprocess failure (returncode != 0) now also catches TPS-below-threshold and surfaces as benchmark failure.

**Rationale:** The kept-path warm-cache command already uses `--repeat 3` + warmup=max_new_tokens (16) + `FULL_RESIDENT=1/PIN_HOT_EXPERTS=1`. Passing the existing `--min-steady-decode-tps` gate makes the warm-cache 1.6–1.9 tok/s target a hard regression check.

- [ ] **Step 1: Modify smoke command to include warm TPS gate**

  In `tests/e2e_full_benchmark.py`, inside `_run_deepseek_v4_flash_direct_benchmark`, after the existing `--repeat 3` extend block, append:

  ```python
  command.extend(
      [
          "--min-steady-decode-tps",
          os.environ.get(
              "FASTINFERENCE_DEEPSEEK_V4_FLASH_MIN_STEADY_DECODE_TPS", "1.5"
          ),
      ]
  )
  ```

  Exact location: after line 2863 (`"--repeat", "3",` block closes at line 2864).

- [ ] **Step 2: Verify the subprocess failure path still reports correctly**

  The existing `proc.returncode != 0` block at lines 2889–2893 already raises `RuntimeError` with stderr; no change needed.

- [ ] **Step 3: Run DeepSeek e2e benchmark and confirm it passes**

  Command:
  ```bash
  cd /home/jack/work/FastInference
  uv run python tests/e2e_full_benchmark.py \
    --models deepseek_v4_flash_q2_gguf \
    --json-out /tmp/ds_gate_test.json
  ```

  Expected: exit code 0; stdout contains `decode_tps=1.60` or higher and no "steady-state decode TPS below threshold" error.

- [ ] **Step 4: Commit**

  ```bash
  git add tests/e2e_full_benchmark.py
  git commit -m "feat(bench): add DeepSeek V4 Flash warm-cache decode TPS gate to e2e benchmark"
  ```

---

## Task 2: DeepSeek cold/warm smoke gate in real-smoke script

**Files:**
- Modify: `tests/run_deepseek_v4_flash_real_smoke.sh`
- Test: run the script

**Interfaces:**
- Consumes: `tests/tools/run_deepseek_v4_flash_gpu_smoke.py` supports `--min-steady-decode-tps`, `--warmup-tokens`, `--repeat`, `--profile-json`.
- Produces: two explicit gate invocations (cold + warm) saved as fixed regression commands; JSON artifacts written to `/tmp/ds_gate_*_run.json`.

**Rationale:** `tests/run_deepseek_v4_flash_real_smoke.sh` is the maintained real-GPU smoke entrypoint. Adding the two gates there makes them discoverable and runnable in CI.

- [ ] **Step 1: Append cold-cache and warm-cache gate commands to the smoke script**

  Add the following block at the end of `tests/run_deepseek_v4_flash_real_smoke.sh`, after the existing pytest/http smoke blocks:

  ```bash
  MODEL=models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf

  echo "===== DeepSeek V4 Flash cold-cache gate ====="
  FASTINFERENCE_KV_TYPE=fp16 \
  FASTINFERENCE_BLOCK_SIZE=32 \
  FASTINFERENCE_DEEPSEEK_V4_FLASH_PIN_HOT_EXPERTS=1 \
  uv run python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
    --model "$MODEL" \
    --context-length 4096 \
    --max-tokens 16 \
    --warmup-tokens 1 \
    --min-steady-decode-tps 0.4 \
    --profile-json /tmp/ds_gate_cold_run.json

  echo "===== DeepSeek V4 Flash warm-cache gate ====="
  FASTINFERENCE_KV_TYPE=fp16 \
  FASTINFERENCE_BLOCK_SIZE=32 \
  FASTINFERENCE_DEEPSEEK_V4_FLASH_FULL_RESIDENT=1 \
  FASTINFERENCE_DEEPSEEK_V4_FLASH_PIN_HOT_EXPERTS=1 \
  FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB=1 \
  uv run python tests/tools/run_deepseek_v4_flash_gpu_smoke.py \
    --model "$MODEL" \
    --context-length 4096 \
    --prompt-length 32 \
    --max-tokens 16 \
    --warmup-tokens 16 \
    --repeat 3 \
    --min-steady-decode-tps 1.5 \
    --profile-json /tmp/ds_gate_warm_run.json
  ```

- [ ] **Step 2: Run the real-smoke script and confirm both gates pass**

  Command:
  ```bash
  cd /home/jack/work/FastInference
  bash tests/run_deepseek_v4_flash_real_smoke.sh
  ```

  Expected: both gate commands exit 0; cold `decode_tps_steady_state >= 0.4`; warm `decode_tps_steady_state >= 1.5`.

- [ ] **Step 3: Commit**

  ```bash
  git add tests/run_deepseek_v4_flash_real_smoke.sh
  git commit -m "feat(smoke): add DeepSeek V4 Flash cold/warm decode TPS gates"
  ```

---

## Task 3: Verify Gemma4 batch=2 with increased active-request limit

**Files:**
- No source file changes yet (verification only)
- Test command uses `tests/e2e_full_benchmark.py` with env override

**Interfaces:**
- Consumes: `_GEMMA4_26B_RECOMMENDED_ENV` / `_GEMMA4_31B_RECOMMENDED_ENV` currently force `FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS=1`.
- Produces: measurement data in `/tmp/fastinference_eval_ab/e2e_gemma*_b2_active2.json` used to decide Task 4.

**Rationale:** The report showed batch=2 currently degrades because `FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS=1`. Before changing defaults, verify whether raising it to 2 (or unsetting to let profile default 4 take effect) actually enables parallel decode and improves throughput.

- [ ] **Step 1: Run 26B batch=2 with active requests = 2**

  Command:
  ```bash
  cd /home/jack/work/FastInference
  FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS=2 \
  FASTINFERENCE_BENCH_WARMUP_PRESET=default \
  uv run python tests/e2e_full_benchmark.py \
    --models gemma4_26b_a4b \
    --gemma26b-concurrent 2 \
    --json-out /tmp/fastinference_eval_ab/e2e_gemma26b_b2_active2.json
  ```

- [ ] **Step 2: Run 31B batch=2 with active requests = 2**

  Command:
  ```bash
  cd /home/jack/work/FastInference
  FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS=2 \
  FASTINFERENCE_BENCH_WARMUP_PRESET=default \
  uv run python tests/e2e_full_benchmark.py \
    --models gemma4_31b_q4 \
    --gemma31b-concurrent 2 \
    --json-out /tmp/fastinference_eval_ab/e2e_gemma31b_b2_active2.json
  ```

- [ ] **Step 3: Compare with batch=1 baseline and batch=2 active=1**

  Parse JSON summaries:
  ```bash
  python3 - << 'PY'
  import json
  for path in [
      "/tmp/fastinference_eval_ab/e2e_gemma4_26b_a4b_26b_b1_fp8_warm.json",
      "/tmp/fastinference_eval_ab/e2e_gemma4_26b_a4b_26b_b2_fp8_warm.json",
      "/tmp/fastinference_eval_ab/e2e_gemma26b_b2_active2.json",
      "/tmp/fastinference_eval_ab/e2e_gemma4_31b_q4_31b_b1_fp8_warm.json",
      "/tmp/fastinference_eval_ab/e2e_gemma4_31b_q4_31b_b2_fp8_warm.json",
      "/tmp/fastinference_eval_ab/e2e_gemma31b_b2_active2.json",
  ]:
      try:
          d = json.load(open(path))
          key = d["models"][0]
          s = d["summary"][key]
          print(f"{path.split('/')[-1]}: decode_tps_agg={s['decode_tps_aggregate']:.2f}")
      except Exception as e:
          print(f"{path}: not ready ({e})")
  PY
  ```

  Decision criteria (from report §P0-3):
  - 26B: success if batch=2 active=2 `decode_tps_aggregate` is close to or exceeds 2× batch=1 (~22 tok/s), or at least significantly above batch=2 active=1 (1.29 tok/s).
  - 31B: success if batch=2 active=2 aggregate TPS is close to 2× 3.7 ≈ 7.4 tok/s, or at least above batch=2 active=1 (1.04 tok/s).

- [ ] **Step 4: Record results in report**

  Add a small summary paragraph to `docs/performance_evaluation_gemma4_deepseek_2026_07_07.md` §P0-3 with measured numbers and go/no-go for Task 4.

- [ ] **Step 5: Commit verification notes (optional, if Task 4 proceeds)**

  If Task 4 is a no-go, commit only the report update; if go, include the report update in Task 4 commit.

---

## Task 4 (conditional on Task 3 success): Update Gemma4 default active requests

**Files:**
- Modify: `tests/e2e_full_benchmark.py:90-117` (`_GEMMA4_31B_RECOMMENDED_ENV` and `_GEMMA4_26B_RECOMMENDED_ENV`)
- Test: re-run e2e batch=2 and correctness regression

**Interfaces:**
- Consumes: Task 3 measurement showing batch=2 improvement with `FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS=2`.
- Produces: new default `FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS` for Gemma4 26B/31B e2e benchmarks.

**Rationale:** If raising the limit works, remove the explicit `=1` cap so the runtime profile default (4) or throughput profile (16) can take effect. This is the minimal change; if a lower cap is still needed for memory safety on 96 GB, set it to 2 instead of removing.

- [ ] **Step 1: Choose the new value based on Task 3 data**

  Two options:
  - **Option A (preferred if memory headroom allows):** Remove `FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS` from both `_GEMMA4_31B_RECOMMENDED_ENV` and `_GEMMA4_26B_RECOMMENDED_ENV`, letting the runtime profile default of 4 apply.
  - **Option B (conservative):** Change both to `"FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS": "2"`.

  Decision rule: if batch=2 active=2 shows expected parallel speedup and no OOM, use Option A; if only active=2 works but active=4 OOMs, use Option B.

- [ ] **Step 2: Apply the chosen change**

  For Option A, delete these lines from `_GEMMA4_31B_RECOMMENDED_ENV` and `_GEMMA4_26B_RECOMMENDED_ENV`:
  ```python
      "FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS": "1",
  ```

  For Option B, change `"1"` to `"2"` in both dicts.

- [ ] **Step 3: Re-run e2e batch=2 to confirm default path**

  Command:
  ```bash
  cd /home/jack/work/FastInference
  uv run python tests/e2e_full_benchmark.py \
    --models gemma4_26b_a4b,gemma4_31b_q4 \
    --gemma26b-concurrent 2 \
    --gemma31b-concurrent 2 \
    --json-out /tmp/gemma_default_b2.json
  ```

  Expected: both models complete without OOM; 26B decode_tps_aggregate significantly higher than the old 1.29; 31B higher than old 1.04.

- [ ] **Step 4: Run fast regression suite (no GPU model loads)**

  Command:
  ```bash
  bash tests/run_regression_suite.sh
  ```

  Expected: all tests pass.

- [ ] **Step 5: Run correctness regression for Gemma4 (GPU required)**

  Command:
  ```bash
  SKIP_A_TIER=1 bash tests/run_inference_correctness_regression.sh
  ```

  Expected: Gemma4 26B/31B correctness gates pass; no semantic regression from batching.

- [ ] **Step 6: Commit**

  ```bash
  git add tests/e2e_full_benchmark.py docs/performance_evaluation_gemma4_deepseek_2026_07_07.md
  git commit -m "feat(bench): raise Gemma4 default KV_MAX_ACTIVE_REQUESTS to enable batch=2 parallel decode"
  ```

---

## Self-Review

1. **Spec coverage:**
   - P0-1 (DeepSeek kept-path warm-cache gate): covered by Task 1.
   - P0-2 (`--min-steady-decode-tps` cold/warm regression): covered by Task 2.
   - P0-3 (Gemma4 batch=2 parallel verification and conditional default change): covered by Task 3 and Task 4.

2. **Placeholder scan:** No TBD/TODO/fill-in-details steps. Each step has exact file paths, line ranges, commands, and expected outputs.

3. **Type consistency:** All function/arg names (`_run_deepseek_v4_flash_direct_benchmark`, `--min-steady-decode-tps`, `FASTINFERENCE_KV_MAX_ACTIVE_REQUESTS`, `decode_tps_aggregate`) match the codebase and report.
