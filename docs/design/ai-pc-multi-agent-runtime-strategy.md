# AI PC Multi-Agent Runtime Strategy

## Purpose

This document defines FastInference's product and architecture direction. It
records the current strengths and limits of the lite runtime, its boundary with
llama.cpp, and the performance strategy for AI PCs running multiple agents.

## Product Position

FastInference is a local, single-GPU inference runtime for AI PCs. Its purpose
is to execute frequent, parallelisable agent work locally and reduce dependence
on costly closed-model tokens. It is not a general CPU inference library and it
does not need to replace every llama.cpp deployment.

The intended stack is:

```text
Agent / workflow orchestrator
  -> FastInference local service
     -> model routing, admission, batching, paged KV, prefix reuse
     -> AI PC GPU
  -> optional closed-model escalation for exceptional tasks
```

FastInference owns inference execution. The agent layer owns planning, tool use,
task decomposition, and the decision to escalate to a cloud model.

## Current Architecture Strengths

- A narrow single-GPU control plane: `LiteEngine`, `RuntimeController`,
  `RequestScheduler`, `StepScheduler`, executors, sampling, and output pipeline.
- Paged KV with a flat pool, block allocation, request slots, and decode-oriented
  Triton PagedAttention.
- AMD ROCm-first AWQ execution, FP8 KV guardrails for Gemma4, and model-specific
  GEMV, QKV, gate/up, and MoE kernels.
- Adapter-owned model policy and direct runtime installation. Model-specific
  data planes can remain local while the request lifecycle stays shared.
- OpenAI-compatible chat subset, streaming, LoRA, structured-output support,
  runtime metrics, and image support for the maintained Gemma4 path.
- Benchmark-gated engineering: numerical correctness and measured end-to-end
  performance are required before a fast path becomes a product path.

## Current Limits

- The maintained model surface is deliberately narrow. It is not a generic
  upstream-vLLM replacement.
- Python, PyTorch, and Triton are appropriate for AI PC GPUs but unsuitable as a
  low-dependency CPU, mobile, browser, or embedded runtime.
- Scheduler policy has many service-class, fairness, LoRA, multimodal, and
  prefix-cache controls. Additional policy dimensions require evidence from
  actual multi-agent workloads, not speculative configurability.
- DeepSeek GGUF is an experimental adapter-owned direct runtime with its own KV
  representation. It must not widen the standard Safetensors/AWQ runtime.
- 26B and 31B models have insufficient decode capacity to be default concurrent
  agent workers on the measured AI PC target.

## Boundary With llama.cpp

llama.cpp/ggml is the better fit for broad edge deployment: CPU, Apple Silicon,
mobile-style environments, browser/WebGPU, low-dependency binaries, GGUF, and
CPU/GPU hybrid offload. FastInference should not duplicate that platform matrix.

FastInference is preferable when the device has a capable GPU and the product
needs a local service with request admission, small-batch scheduling, paged KV,
model-specific AWQ kernels, and OpenAI-style integration.

The two runtimes can coexist:

```text
AI PC local service and agent workers -> FastInference + Safetensors/AWQ
portable/offline CPU or broad device target -> llama.cpp + GGUF
```

GGUF may remain an adapter-owned compatibility path for exceptional models. It
is not the default model format or the architectural center of FastInference.

## Multi-Agent Performance Strategy

The primary metric is not a single large model's peak TPS. It is useful local
agent capacity:

- aggregate decode TPS at batch sizes 1, 2, 4, and 8;
- per-agent tail latency and queue wait;
- prefix-cache hit rate for shared prompts, tools, and project context;
- local task success rate and cloud escalation rate;
- closed-model tokens avoided per completed workflow.

The model hierarchy should be:

```text
7B-14B AWQ: default high-frequency local agent workers
26B:        local high-quality escalation and complex review
31B:        selective local expert, not a default worker pool
cloud:      long reasoning, high-risk decisions, or quality fallback
```

A 26B model at roughly 11 decode TPS cannot sustain several agents streaming
long outputs at once. It is useful as a queued expert, not as the default model
for concurrent workers. A 31B model with lower measured decode throughput has
the same limitation more strongly.

## Near-Term KPI

The next formal performance target is 7B-14B AWQ worker capacity on the target
AI PC. Initial acceptance targets are:

| Metric | 7B | 14B |
| --- | ---: | ---: |
| BS=1 decode | >= 35 tok/s | >= 20 tok/s |
| BS=4 aggregate decode | >= 80 tok/s | >= 50 tok/s |
| Four simultaneous short tasks | >= 10 tok/s per agent | >= 6 tok/s per agent |
| Shared-prefix hit rate | >= 70% | >= 70% |
| Cloud escalation rate | < 20% | < 10% |

These are product hypotheses. They must be calibrated against the selected
hardware, model, prompt distribution, and task-quality baseline before they
become regression thresholds.

## Engineering Priorities

1. Select supported 7B and 14B worker models and lock their tokenizer,
   chat-template, quality fixtures, and benchmark fixtures.
2. Add multi-agent benchmarks that report aggregate TPS, per-agent latency,
   queue time, prefix-cache hits, and cloud-token-avoidance inputs.
3. Profile decode by weight bytes/token, lm-head, attention/KV, AWQ linear,
   kernel launch overhead, and Python control-plane cost.
4. Optimise M=1, M=2, and M=4 AWQ linear and PagedAttention only when the
   selected production path is confirmed by audit counters.
5. Make prefix-cache reuse reliable for common agent prefixes before adding new
   speculative execution mechanisms.
6. Keep 26B/31B performance work separate and evidence-driven. A bandwidth
   feasibility calculation must precede a large-model TPS target.

## Explicit Non-Goals

- Rebuilding llama.cpp's CPU, mobile, browser, or generic GGUF ecosystem.
- Restoring upstream vLLM's distributed runtime or broad model compatibility.
- Adding speculative decoding because it is fashionable. Native MTP for Gemma4
  was evaluated and stopped because the exact verifier path did not satisfy the
  required end-to-end performance gate.
- Turning the scheduler into a generic data-center policy engine without a
  measured AI PC multi-agent workload that needs the added policy.

## Decision Rule

Keep a runtime or kernel change only when it improves the selected AI PC
multi-agent metrics without breaking quality, token-ID correctness, memory
headroom, or latency fairness. Otherwise retain the measurement as a report and
remove the production-path change.
