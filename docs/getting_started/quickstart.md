# Quickstart (LitevLLM)

This guide will help you quickly get started with **FastInference** (vLLM Lite) to perform:

- [Offline batched inference](#offline-batched-inference)
- [Online serving using gRPC/OpenAI-compatible server](#online-serving)

## Prerequisites

- **OS**: Linux
- **Python**: 3.10 -- 3.13
- **Hardware**: Single GPU (NVIDIA or AMD)
- **Compiler**: None required (No C++ extensions!)

## Installation

FastInference is designed to be installed without a C++ compiler. We recommend using [uv](https://docs.astral.sh/uv/) for the fastest setup.

```bash
# Clone the repository
git clone https://github.com/jiayingsong2023/litevLLM.git
cd litevLLM

# Install in editable mode
uv pip install -e .
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

# Generate outputs
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

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
# Measure MoE throughput (Scaling up to BS=32)
uv run python tests/e2e_moe_batch_scaling.py

# Measure GGUF throughput with LRU Caching
uv run python tests/e2e_gguf_perf.py
```

## Core Backend: Triton Only

FastInference standardizes on **OpenAI Triton** for all performance-critical operators. 
- No need to select `FLASH_ATTN` or `FLASHINFER`.
- The engine automatically uses the optimized `TritonAttention` backend.
- Supports **AMD Strix Point** (gfx1151) out of the box with stability-first IO paths.
