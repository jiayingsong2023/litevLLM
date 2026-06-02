# Gemma4 QKV Fused Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Microbenchmark Gemma4-31B AWQ decode QKV/QK fused kernel configs and hardcode the fastest exact-shape launch configs.

**Architecture:** Keep the existing fused QKV kernel and add a narrow config resolver for the two audited Gemma4-31B shapes. A benchmark tool sweeps `BLOCK_N` and `BLOCK_GROUPS` using the current safe wrapper, then the winning exact-shape configs are encoded in `awq_fused_gemm.py`.

**Tech Stack:** Python 3.12, uv, PyTorch, Triton through `vllm.triton_utils`, pytest.

---

### Task 1: Add Exact-Shape Resolver Tests

**Files:**
- Modify: `tests/test_awq_gemm_m1_specialization.py`
- Modify: `vllm/kernels/triton/awq_fused_gemm.py`

- [ ] Add tests that call `_qkv_group32_gemv_launch_config` for local QKV and global QK shapes and assert the selected tuple.
- [ ] Run `uv run pytest tests/test_awq_gemm_m1_specialization.py::test_qkv_group32_exact_shape_launch_config -q` and verify it fails before implementation.
- [ ] Implement `_qkv_group32_gemv_launch_config(total_n, k, has_v)` with exact-shape rules plus tool override precedence.
- [ ] Run the targeted test and verify it passes.

### Task 2: Add Microbenchmark Sweep Tool

**Files:**
- Create: `tests/tools/benchmark_awq_qkv_fused_m1.py`

- [ ] Add a tool that creates the two real Gemma4-31B shapes, sweeps `BLOCK_N in {128,192,256,384}` and `BLOCK_GROUPS in {2,4,8}`, and reports JSON.
- [ ] Use `set_awq_fused_tuning_config()` rather than process env mutation for each candidate.
- [ ] Run the tool on GPU and record the winning local QKV and global QK configs.

### Task 3: Hardcode Winning Configs

**Files:**
- Modify: `vllm/kernels/triton/awq_fused_gemm.py`
- Modify: `docs/GEMMA4_31B_P0_AWQ_AUDIT.md`

- [ ] Replace placeholder exact-shape values with benchmark winners.
- [ ] Document the winners and the before/after profile gate.
- [ ] Run targeted tests, ruff, and `bash tests/run_regression_suite.sh`.
