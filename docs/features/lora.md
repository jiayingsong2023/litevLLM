# LoRA

LoRA support exists in the lite runtime, but it is currently experimental. Use
[CAPABILITY_MATRIX.md](../CAPABILITY_MATRIX.md) as the source of truth for
support status.

## Runtime Shape

FastInference keeps LoRA integration inside the Python/Triton lite path rather
than depending on upstream C++/CUDA LoRA extension stacks. Adapter-aware
scheduling and observability counters are present in the runtime, and LoRA
stats are included in benchmark/runtime summaries.

## Phase 1 Boundary

Phase 1 is deliberately narrow:

- PEFT `adapter_model.safetensors` only.
- `adapter_config.json` must provide `r`, `lora_alpha`, and `target_modules`.
- Supported target modules are `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, and `down_proj`.
- `adapter_model.bin` is not supported.
- `lm_head` LoRA is not supported.
- Mixed LoRA batches are not supported. A runtime batch may contain only base
  requests or requests for one adapter.
- DeepSeek V4 Flash direct runtime rejects LoRA.

## Offline Smoke

Register the adapter before generating, then pass the request through
`LiteEngine.add_request()` or `AsyncLLM.generate(..., lora_request=...)`.
The convenience `LLM.generate()` API does not yet accept `lora_request`.

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

base_model = "/path/to/base-model"
adapter_path = "/path/to/peft-adapter"

llm = LLM(model=base_model)
llm.register_lora_adapter(lora_name="adapter-a", lora_path=adapter_path)

sampling = SamplingParams(max_tokens=32, temperature=0.0)
lora_request = LoRARequest(
    lora_name="adapter-a",
    lora_int_id=1,
    lora_path=adapter_path,
)

llm.engine.add_request(
    "lora-smoke",
    "Hello, my name is",
    sampling,
    lora_request=lora_request,
)

while llm.engine.active_request_count:
    for output in llm.engine.step():
        if output.finished:
            print(output.outputs[0].text)
```

## Expected Adapter Layout

```text
/path/to/peft-adapter/
  adapter_config.json
  adapter_model.safetensors
```

PEFT weights are loaded as `lora_A.weight == (rank, input_size)` and
`lora_B.weight == (output_size, rank)`. The loader transposes them to the
runtime layout and raises `ValueError` when the shape does not match the target
`LiteLinear` layer.

## Verification

Run the focused LoRA tests:

```bash
uv run --no-sync pytest tests/lora -q
```

Run scheduler and batch-contract tests:

```bash
uv run --no-sync pytest tests/test_input_batch_builder.py tests/test_step_scheduler.py -q
```

Run the fast regression suite before publishing LoRA changes:

```bash
bash tests/run_regression_suite.sh
```

If `uv run` stalls in this ROCm workspace, run the equivalent pytest target with
`uv run --no-sync`.
