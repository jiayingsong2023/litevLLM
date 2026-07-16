# Gemma4 12B Verified Batch-Decode Envelope

Date: 2026-07-15

## Purpose

This document records the Gemma4 12B batch-decode investigation. The M=4 path
is greedy token-ID equivalent and improves aggregate throughput, but it fails
the per-agent p95 latency gate. It is therefore not a production envelope.

## Execution Contract

The verified environment is:

| Property | Required value |
| --- | --- |
| Model | `models/gemma-4-12B-it-AWQ-INT4` |
| Production decode batch | M=1 only |
| Scheduler capacity | `max_num_seqs=4` |
| KV cache | `fp8`; do not set `FASTINFERENCE_GEMMA4_ALLOW_INT4_KV=1` |
| Sampling | `temperature=0`, fixed `max_tokens`, no penalties, no structured output, no LoRA or image requests |
| Context | 128, 512, or 2048 initial tokens |
| Chat template SHA-256 | `36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0` |

The candidate scheduler path is:

```text
DecodePrefillPlanner.use_fast_path
  -> RuntimeController.decode_step_sync
  -> DecodeExecutor.execute_sync_fast
  -> InputBatchBuilder.build_decode_fast
```

Candidate runs use `max_num_seqs=4` to prevent an unverified M>4 shape. The
production scheduler advertises `(1,)`; M=2, M=3, and M=4 remain M=1 until a
candidate passes every gate.

## Acceptance Gates

The benchmark uses eight fixed chat-template prompts in
`tests/tools/fixtures/gemma4_12b_batch_prompts.json`. A candidate M=4 run must:

1. use a current M=1 result produced by the same tool as its reference;
2. match model path, KV dtype, and chat-template hash;
3. match every generated token ID for every fixture;
4. have no context bucket regress relative to M=1; and
5. reach at least `1.05x` median decode-TPS speedup over M=1 across the
   128/512/2048 context buckets; and
6. keep every agent's GPU-event decode p95 within `2.0x` of its M=1 reference.

The end-to-end check is intentionally separate from CUDA unit tests. Unit tests
catch the numeric risks cheaply: the deep BF16 M=2 projection cannot narrow to
FP16, and the M=2 fused QKV kernel matches three individual projections. The
tool confirms the complete scheduler, paged-KV, model, and sampler contract.

```bash
uv run python tests/tools/gemma4_12b_exact_batch_baseline.py \
  --batch-size 1 --max-new-tokens 32 \
  --out /tmp/gemma4-12b-m1.json

uv run python tests/tools/gemma4_12b_exact_batch_baseline.py \
  --batch-size 4 --max-new-tokens 32 \
  --reference /tmp/gemma4-12b-m1.json --min-speedup 1.05 \
  --out /tmp/gemma4-12b-m4.json
```

The tool exits nonzero on contract mismatch, token mismatch, throughput-gate,
or p95-gate failure. Its JSON records the AWQ dispatch audit and GPU-event p95.

To cover the public offline API rather than direct engine stepping, repeat the
same gate with `--surface llm`. This reports end-to-end output TPS, including
prefill and API orchestration; it must only be compared with an M=1 result from
the same surface.

```bash
uv run python tests/tools/gemma4_12b_exact_batch_baseline.py \
  --surface llm --batch-size 1 --max-new-tokens 32 \
  --out /tmp/gemma4-12b-llm-m1.json

uv run python tests/tools/gemma4_12b_exact_batch_baseline.py \
  --surface llm --batch-size 4 --max-new-tokens 32 \
  --reference /tmp/gemma4-12b-llm-m1.json --min-speedup 1.05 \
  --out /tmp/gemma4-12b-llm-m4.json
```

## Measured Result: Rejected

On the local Radeon 8060S Graphics system with 32 generated tokens per
fixture, strict M=4 remained token-ID exact and exceeded the throughput gate,
but failed the p95 gate before the 2048 rerun was needed:

| Context | M=1 decode TPS | M=4 decode TPS | Token-ID parity | Speedup | p95 ratio |
| --- | ---: | ---: | --- | ---: |
| 128 | 8.834 | 14.813 | exact | 1.677x | 2.542x |
| 512 | 8.328 | 13.394 | exact | 1.608x | 2.696x |

The M=4 p95 gate fails in both completed buckets. M=2 also regressed to
`0.771x` aggregate TPS at context 512. The production scheduler capability is
therefore `(1,)`. The 2048 M=4 candidate was stopped after the earlier gates
already rejected the shape.

The public `LLM.generate()` end-to-end gate also passed on the same fixture:

| Surface | M=1 output TPS | M=4 output TPS | Token-ID parity | Speedup |
| --- | ---: | ---: | --- | ---: |
| `LLM.generate()` | 7.063 | 16.828 | exact | 2.383x |

This is an end-to-end output-throughput metric: it includes prefill, request
orchestration, and output handling. It is intentionally not compared with the
decode-only table above.

## Regression Entrypoints

`tests/run_inference_correctness_regression.sh` runs the M=1/M=4 public-API
token-ID parity diagnostic when `models/gemma-4-12B-it-AWQ-INT4` is present. Set
`RUN_GEMMA4_12B_BATCH=0` to skip it or `MODEL_GEMMA4_12B_AWQ` to override the
model path.

`tests/e2e_full_benchmark.py --models gemma4_12b_awq` runs the standard
AsyncLLM end-to-end performance harness with the candidate FP8, M=4 and
512-token configuration. It is diagnostic only, not a production envelope.

On the local Radeon 8060S, its 128-token prompt and 32-token output workload
reported `tokens/s=6.67`. This is the framework's full wall-clock throughput,
including prefill and asynchronous streaming. Its per-request decode-window
metric must not be compared with the batch decode TPS measured above.

## Boundaries

The envelope does not validate different checkpoints, KV dtypes, chat
templates, context lengths above 2048, sampling policies, LoRA, multimodal
requests, M=2 as a throughput tier, or M>4. Any extension must first run the
same M=1 reference and M=N candidate gate rather than inherit this result.
