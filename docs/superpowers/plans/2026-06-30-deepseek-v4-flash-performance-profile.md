# DeepSeek V4 Flash Decode Profile

## Baseline Evidence

Command:

```bash
uv run python tests/tools/run_deepseek_v4_flash_gpu_smoke.py --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --max-tokens 16 --warmup-tokens 1 --profile-json /tmp/deepseek_v4_flash_profile_workspace.json
```

Result after selected payload workspaces:

- Decode TPS aggregate: 0.9045 tok/s
- Decode TPS steady: 0.8956 tok/s
- `layer_moe`: 6695.44 ms total, 9.73 ms avg
- `moe_routed_experts`: 5082.01 ms total, 7.94 ms avg
- `router_selected_experts_kernel`: 3440.46 ms total, 5.00 ms avg
- `router_expert_stage`: 1902.71 ms total, 2.77 ms avg

ROCm kernel profile command:

```bash
/opt/rocm/bin/rocprofv3 --kernel-trace --stats --output-format csv --output-file /tmp/deepseek_v4_flash_decode_rocprof -- uv run python tests/tools/run_deepseek_v4_flash_gpu_smoke.py --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --max-tokens 4 --warmup-tokens 1 --profile-json /tmp/deepseek_v4_flash_profile_rocprof.json
```

Top ROCm kernels:

- `_iq2_xxs_selected_experts_activation_kernel`: 29.09%
- `_q8_0_raw_matvec_kernel`: 21.89%
- `_q2_k_selected_experts_down_projection_kernel`: 11.76%
- `__amd_rocclr_copyBuffer`: 6.06%

## Tiling Check

- Selected Q2/IQ2 rows-per-block `4/4`: steady 0.8956 tok/s, selected kernel 5.00 ms avg.
- Selected Q2/IQ2 rows-per-block `8/8`: steady 0.8708 tok/s, selected kernel 5.24 ms avg.
- Selected Q2/IQ2 rows-per-block `2/2`: steady 0.7648 tok/s, selected kernel 5.36 ms avg.

Decision: keep `4/4`. It is the fastest measured option and preserves output token IDs across runs.

## Direct Selected Payloads

Command:

```bash
uv run python tests/tools/run_deepseek_v4_flash_gpu_smoke.py --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --max-tokens 16 --warmup-tokens 1 --profile-json /tmp/deepseek_v4_flash_profile_direct6_all.json
```

Result after direct gate/up and direct down payload kernels:

- Decode TPS aggregate: 1.0182 tok/s
- Decode TPS steady: 1.0062 tok/s
- `layer_moe`: 6284.28 ms total, 9.13 ms avg
- `moe_routed_experts`: 4923.79 ms total, 7.69 ms avg
- `router_selected_experts_kernel`: 3109.07 ms total, 4.52 ms avg
- Output token IDs match the previous profiled runs.

ROCm direct payload profile:

- `_iq2_xxs_selected_experts_activation_direct6_kernel`: 29.80%
- `_q2_k_selected_experts_down_projection_direct6_kernel`: 12.83%
- `__amd_rocclr_copyBuffer`: 1.86%

Decision: keep direct selected payload kernels. They remove selected expert
payload stack copies from the hot path and improve the 16-token smoke from the
sub-0.9 tok/s range to about 1.0 tok/s on this run.

## Q8_0 Raw Matvec

Negative check: a batched Q8 raw matvec for 2-D hidden tensors did not enter the
single-request decode hot path. ROCm stats still showed only
`_q8_0_raw_matvec_kernel`, so the experimental batched path was removed.

Kept change: Q8 raw kernels now sign-extend payload values with
`uint8 -> int8 -> fp32` instead of materializing an explicit
`raw >= 128 ? raw - 256 : raw` select. This preserves Q8_0 math and output token
IDs.

Same-command ROCm comparison on a 4-token smoke:

- Before: `_q8_0_raw_matvec_kernel` 4536 calls, 520.70 ms total, 114.79 us avg.
- After: `_q8_0_raw_matvec_kernel` 4536 calls, 428.48 ms total, 94.46 us avg.
- Before: `_q8_0_raw_gate_up_activation_kernel` 344 calls, 65.12 ms total,
  189.31 us avg.
- After: `_q8_0_raw_gate_up_activation_kernel` 344 calls, 50.18 ms total,
  145.87 us avg.

16-token smoke after the Q8 sign-extension change:

- Decode TPS aggregate: 1.5513 tok/s
- Decode TPS steady: 1.5342 tok/s
- `output_projection`: 97.34 ms total, 6.08 ms avg
- Output token IDs match the previous profiled runs.

## IQ2/Q2 Kernel Trims

Kept IQ2 change: direct selected-expert IQ2 sign decoding now uses
`tl.where(sign_bit == 0, 1.0, -1.0)` instead of `1.0 - sign_bit * 2.0`.
This is mathematically equivalent and keeps output token IDs unchanged.

Same-command ROCm comparison on a 4-token smoke:

- Before: `_iq2_xxs_selected_experts_activation_direct6_kernel` 344 calls,
  464.52 ms total, 1.350 ms avg.
- After: `_iq2_xxs_selected_experts_activation_direct6_kernel` 344 calls,
  427.78 ms total, 1.244 ms avg.

Rejected Q2 change: static-unrolling the six direct down-projection expert
pointers made `_q2_k_selected_experts_down_projection_direct6_kernel` slower
from about 241.02 ms to 273.87 ms, so it was removed.

Small PyTorch-kernel cleanup: `deepseek_v4_indexer_select_scores` now applies
its scale inside the Triton store, removing the separate `scores * scale`
PyTorch elementwise launch. The 16-token smoke after the kept trims reported:

- Decode TPS aggregate: 1.6153 tok/s
- Decode TPS steady: 1.5926 tok/s
- `router_selected_experts_kernel`: 1539.09 ms total, 2.24 ms avg
- Output token IDs match the previous profiled runs.

## Compressor Update Breakdown

Added nested profile sections inside `_update_compressor_state` to split
`compressed_kv_update` and `compressed_indexer_update`.

16-token smoke with breakdown sections:

- `compressed_kv_update`: 1213.81 ms total, 656 calls
- `compressed_indexer_update`: 878.86 ms total, 336 calls
- `compressor_project_kv`: 703.42 ms total, 992 calls
- `compressor_project_score`: 664.34 ms total, 992 calls
- `compressor_emit_indexer_qat`: 342.05 ms total, 84 calls
- `compressor_emit_fp8_qat`: 108.31 ms total, 84 calls
- `compressor_runtime_update`: 66.31 ms total, 992 calls
- `compressor_candidate_norm`: 75.32 ms total, 992 calls
- `compressor_emit_pool`: 16.55 ms total, 168 calls
- `compressor_emit_rope`: 19.28 ms total, 168 calls
- `compressor_carry_update`: 9.23 ms total, 168 calls

Decision at this point: focus next on `deepseek_indexer_qat` and only validate
compressor Q8 projections if the real tensor table supports that path. Runtime
row copies, pooling, rope, and carry updates are too small to justify new
kernels now.

## Triton Indexer QAT

Replaced the emitted indexer-row `deepseek_indexer_qat_reference` call in the GPU
compressor update path with a two-kernel Triton implementation:

- `_indexer_hadamard128_kernel`: computes the fixed 128-point Hadamard output.
- `_indexer_e2m1_roundtrip_kernel`: applies the 4x32-value E2M1 roundtrip.

The CPU/reference function remains unchanged and the Triton path is covered by a
direct reference-equivalence test.

16-token smoke comparison:

- Before: `compressor_emit_indexer_qat` 342.05 ms total, 84 calls.
- After: `compressor_emit_indexer_qat` 8.86 ms total, 84 calls.
- Before: `compressed_indexer_update` 878.86 ms total, 336 calls.
- After: `compressed_indexer_update` 576.04 ms total, 336 calls.
- Decode TPS steady in the profiled run: 1.5632 tok/s.
- Output token IDs match the previous profiled runs.

## Compressor Dual Q8 Projection Check

Experiment: add a technical-only dual Q8 raw matvec path for compressor
`kv/gate` projection and gate it behind the existing Q8_0 raw projection
preconditions.

Result: the 16-token smoke profile did not enter the dual path. It still
reported separate projection sections:

- `compressor_project_kv`: 729.59 ms total, 992 calls.
- `compressor_project_score`: 687.41 ms total, 992 calls.
- No `compressor_project_dual_q8` section was emitted.

Stop-condition rocprof baseline:

```bash
/opt/rocm/bin/rocprofv3 --kernel-trace --stats --output-format csv --output-file /tmp/deepseek_v4_flash_dual_q8_rejected_rocprof -- uv run python tests/tools/run_deepseek_v4_flash_gpu_smoke.py --model models/DeepSeek-V4-Flash-ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --max-tokens 4 --warmup-tokens 1 --profile-json /tmp/deepseek-v4-flash-dual-q8-rejected-profile.json
```

That run still reported:

- `compressor_project_kv`: 183.73 ms total, 248 calls.
- `compressor_project_score`: 165.75 ms total, 248 calls.
- Output token IDs: `[1, 223, 42498, 92831, 305]`.

Tensor-table check on the profiled DS4 GGUF showed why:

- `blk.2.attn_compressor_kv.weight`: dims `(4096, 1024)`, type `1` (`F16`).
- `blk.2.attn_compressor_gate.weight`: dims `(4096, 1024)`, type `1` (`F16`).
- `blk.2.indexer_compressor_kv.weight`: dims `(4096, 256)`, type `1` (`F16`).
- `blk.2.indexer_compressor_gate.weight`: dims `(4096, 256)`, type `1` (`F16`).

Decision: reject and remove the dual Q8 experiment. Current
`compressor_project_*` hotspots are F16 projection work in this model, not Q8_0
raw matvec work, so Q8 dual projection cannot lower the measured compressor
projection totals. Per the stop condition, do not continue speculative
compressor projection optimization from this path; move remaining work to
stability, documentation, and benchmark baselines.
