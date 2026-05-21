# Quickstart (LitevLLM)

This guide will help you quickly get started with **FastInference** (vLLM Lite) to perform:

- [Offline batched inference](#offline-batched-inference)
- [Online serving using gRPC/OpenAI-compatible server](#online-serving)

## Prerequisites

- **OS**: Linux
- **Python**: 3.12
- **Hardware**: Single GPU (NVIDIA or AMD)
- **Compiler**: None required (No C++ extensions!)

## Installation

FastInference is designed to be installed without a C++ compiler. We recommend using [uv](https://docs.astral.sh/uv/) for the fastest setup.

```bash
# Clone the repository
git clone https://github.com/jiayingsong2023/litevLLM.git
cd litevLLM

# Install dependencies
uv sync
```

## Offline Batched Inference

The simplest way to use FastInference is via the `LLM` class.

```python
from vllm import LLM, SamplingParams

# Define prompts
prompts = [
    "Hello, my name is",
    "The capital of France is",
]

# Configure sampling
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Initialize the engine (automatically uses Triton and LRU caching)
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

## 💡 Hugging Face Tips

### Automatic Mirror (For China Users)
If you cannot connect to Hugging Face Hub, use the official mirror by setting this environment variable before starting:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Offline Mode
To run FastInference in a completely firewalled environment:
1. Pre-download the model: `huggingface-cli download meta-llama/Llama-3-8B`
2. Run with the local path: `llm = LLM(model="/path/to/local/model")`
3. Force offline mode: `export HF_HUB_OFFLINE=1`

## Online Serving

## Online Serving

FastInference supports both gRPC and HTTP (OpenAI-compatible) entrypoints.

### Starting the gRPC Server
```bash
uv run python -m vllm.entrypoints.grpc_server --model models/TinyLlama-1.1B-Chat-v1.0 --port 50051
```

### Starting the OpenAI API Server
```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

## Performance Benchmarking

FastInference comes with specialized benchmarks to demonstrate its Triton-powered speed.

```bash
# Default full benchmark: Gemma4 26B A4B + Gemma4 31B Q4
uv run python tests/e2e_full_benchmark.py

# Measure a specific supported model
uv run python tests/e2e_full_benchmark.py --models gemma4_26b_a4b

# Run the fast non-model regression suite
bash tests/run_regression_suite.sh
```

On AMD Radeon 8060S 65GB, the 2026-05-21 Gemma4 default benchmark profile
measured `gemma4_26b_a4b` at `5.53` aggregate TPS / `12.03` decode TPS and
`gemma4_31b_q4` at `1.62` aggregate TPS / `3.71` decode TPS. The 26B default
uses the `two_stage` MoE int4 decode kernel; the benchmark still pins
`KV cap=512` for reproducible measurement shape. Both models leverage the vectorized 2D BatchSampler.
`FASTINFERENCE_GEMMA4_MOE_INT4_KERNEL_STRATEGY=batched_grouped_streaming` is
available for compact top-k tmp decode experiments. Grouped AWQ prefill is
currently opt-in with `FASTINFERENCE_GEMMA4_MOE_PREFILL_GROUPED=1`. The
`chunked` strategy directly consumes expert-major AWQ weights with a compact
intermediate chunk; `FASTINFERENCE_GEMMA4_MOE_PREFILL_GROUPED_STRATEGY=fused`
is a single-kernel experiment that avoids dense materialization and full tmp
traffic, but stays gated because it repeats gate/up work across output tiles.

## Core Backend: Triton Only

FastInference standardizes on **OpenAI Triton** for all performance-critical operators. 
- No need to select `FLASH_ATTN` or `FLASHINFER`.
- The engine automatically uses the optimized `TritonAttention` backend.
- Supports **AMD Strix Point** (gfx1151) out of the box with stability-first IO paths.
