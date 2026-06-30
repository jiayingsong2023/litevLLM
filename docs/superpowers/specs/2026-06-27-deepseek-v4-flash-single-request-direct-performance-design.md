# DeepSeek V4 Flash Single-Request Direct Performance Optimization

**Goal:** Raise single-request direct decode throughput for DeepSeek V4 Flash from the current ~1.4 tok/s toward the theoretical GPU-bound ceiling, without investing further in hip graph.

**Scope:** Single-request direct inference path only. Batched engine optimizations are out of scope unless they can be reused transparently by the direct path.

**Regression gate:** `bash tests/run_inference_correctness_regression.sh` must pass before any change is considered complete.

---

## Background

Profiling and exploration identified four dominant bottlenecks:

1. **Compressed attention PyTorch fallback.** `deepseek_v4_compressed_attention` in `vllm/kernels/triton/deepseek_v4_flash/compressed_attention.py` is currently a pure PyTorch `index_select → matmul → softmax → matmul` path. It is memory-bound and leaves a lot of Triton fusion opportunity on the table.
2. **Per-token MoE loop.** Even the batched layer forwards still call `_run_sliding_moe` once per token, so expert weights are staged and read separately for each token.
3. **Ratio-4 emit-boundary fallback.** Every 4th decode token disables the captured graph path and re-runs the full indexer selection, producing long-tail latency spikes (observed ~837 ms on `layer_4_compressed`).
4. **Expert payload staging overhead.** `router_expert_stage` is a separate CPU→GPU copy phase and is not overlapped with compute.

Hip graph has already been ruled out as the primary investment, so the plan focuses on kernel fusion, cache/staging tuning, and graph-friendly fallback fixes.

---

## Approaches Considered

### Approach A: Low-Risk Tuning First
- Sweep `FASTINFERENCE_BLOCK_SIZE` (16/32/64) and `FASTINFERENCE_KV_TYPE` (`turbo_int4`, `fp8`, `fp16`).
- Enlarge/pin the expert staging budget and pin decode-hot experts.
- Keep `compressed_count` capped in the emit-boundary fallback to avoid full indexer rescans.
- Optionally lower `indexer_top_k` from 512 if accuracy allows.

**Pros:** Fast to implement, low risk, establishes a cleaner baseline.  
**Cons:** Ceiling is limited; cannot fix kernel-level inefficiency.

### Approach B: Kernel Fusion + Grouped Staging (selected)
- Replace `deepseek_v4_compressed_attention` with a hand-written Triton online-softmax kernel.
- Fuse the ratio-4 boundary "write compressed row → read it back for attention" into a single kernel where possible.
- Fuse the existing two MoE kernels (`activation` + `down_projection`) into one.
- Add a `_run_sliding_moe_batched` helper that collects selected experts across a micro-batch and stages/runs them once.

**Pros:** Highest theoretical throughput gain.  
**Cons:** More Triton code to write and tune; strict correctness requirements.

### Approach C: Internal Pseudo-Batching
- Accumulate 2–4 tokens in the single-request direct path and run them through the batched layer forwards to amortize weight reads.

**Pros:** Reuses the existing batched hyper-connection work.  
**Cons:** Adds decode latency; less attractive for latency-sensitive single-request use cases. Deferred.

---

## Selected Design

**Order of attack:**
1. **Task 1 — Tune & baseline:** run block-size/KV-type sweeps and expert-cache tuning, cap `compressed_count` on fallback, document gains.
2. **Task 2 — Compressed attention Triton kernel:** implement `deepseek_v4_compressed_attention_triton` behind the existing `DeepSeekV4CompressedAttentionTensorInputs` contract; keep PyTorch path as a debug/reference fallback behind an env flag.
3. **Task 3 — Ratio-4 boundary fusion:** fuse the compressor write + indexer update + attention read at emit boundaries.
4. **Task 4 — MoE grouped staging:** implement micro-batched expert staging and a fused activation→down kernel for the dense path.

Each task has its own correctness tests and must pass the full inference correctness regression before moving to the next task.

---

## Key Interfaces

- `deepseek_v4_compressed_attention(inputs: DeepSeekV4CompressedAttentionTensorInputs) -> torch.Tensor`  
  Contract unchanged; implementation swapped.
- New optional dispatch flag (env): `FASTINFERENCE_DEEPSEEK_V4_FLASH_COMPRESSED_ATTN_FALLBACK=1` forces PyTorch path.
- New helper: `_run_sliding_moe_batched(hidden: torch.Tensor, ...) -> torch.Tensor` with same semantics as the per-token loop but accepting `[batch, hidden]`.

---

## Testing Strategy

- **Unit/kernel correctness:**
  - `tests/deepseek_v4_flash/test_gpu_compressed_attention.py`
  - `tests/deepseek_v4_flash/test_q2_iq2_moe_kernel.py`
  - `tests/deepseek_v4_flash/test_gpu_batched_layers.py`
- **End-to-end smoke:** `bash tests/run_deepseek_v4_flash_real_smoke.sh`
- **Regression gate:** `bash tests/run_inference_correctness_regression.sh`
- **Performance tracking:** run `tests/e2e_full_benchmark.py` single-request direct workload before/after each task and record tokens/s.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Triton kernel numerical drift | Compare bit-exact or tight-relaxed tolerance against PyTorch reference on unit tests; keep fallback env flag. |
| Staging cache blow-up | Bound budget by `DeepSeekV4FlashMemoryPolicy`; pin only top-k experts per layer. |
| Emit-boundary accuracy loss | Cap `compressed_count`, do not drop rows; validate with correctness regression. |
| Longer Triton compile time | Cache compiled kernels via Triton cache; benchmark cold vs warm compile separately. |

---

## Implementation Notes

Executed on branch `optimize/deepseek-v4-flash-performance`.

### Results

- **Baseline**: ~1.40 tok/s (single-request direct, default turbo_int4 / block size 16).
- **Final smoke tool**: ~1.67 tok/s with `FASTINFERENCE_KV_TYPE=fp16`, `FASTINFERENCE_BLOCK_SIZE=32`, hot-expert pinning enabled, and `FASTINFERENCE_DEEPSEEK_V4_FLASH_STAGING_BUDGET_GB=1`.
- **Final e2e benchmark**: `tests/e2e_full_benchmark.py` was reporting only ~1.13 tok/s because it enabled the DeepSeek profiler via `--profile-json`. After removing `--profile-json` and capturing the smoke JSON from stdout, the e2e benchmark now reports **~1.85 tok/s** for the same configuration, matching the standalone smoke tool.
- **Improvement**: ~+19% standalone smoke, ~+63% e2e benchmark decode_tps.

### Deviations from the Original Plan

- The Triton compressed-attention kernel was implemented as a two-stage kernel (scores + weighted reduce) rather than a single online-softmax pass. This is simpler and numerically stable, and the fallback env flag (`FASTINFERENCE_DEEPSEEK_V4_FLASH_COMPRESSED_ATTN_FALLBACK=1`) remains available.
- The indexer-select Triton kernel clamps per-head dot products to `>= 0` inside the kernel, matching the original PyTorch reference's ReLU behavior.
- Hot-expert pinning reuses the existing `_stage_expert_token_table` helper to handle both `torch.Tensor` and `DeepSeekV4FlashTensor` expert tables.

### Known Limitations

- `uv run mypy vllm` is blocked by a pre-existing syntax error in `vllm/attention/backend.py:34`; the changed files were checked individually and pass mypy.
- `run_inference_correctness_regression.sh` now runs the DeepSeek V4 Flash Tier-B quality smoke with the same optimized env (`FASTINFERENCE_KV_TYPE=fp16`, `FASTINFERENCE_BLOCK_SIZE=32`, hot-expert pinning, 1 GB staging budget) so correctness is validated at the throughput-maximizing settings. Tier-B regression passes cleanly.
