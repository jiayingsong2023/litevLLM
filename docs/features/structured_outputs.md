# Structured Outputs (JSON & Regex Constraints)

FastInference (vLLM Lite) includes a production-grade **StructuredOutputManager** that ensures model outputs strictly adhere to a predefined format, such as JSON or a specific Regular Expression.

## 🚀 How it Works

### 1. Outlines Integration
We utilize the **Outlines** engine as our primary backend for constrained decoding. It builds a finite state machine (FSM) from your schema and maps it to the model's vocabulary.

### 2. Triton-Aware Bitmasking
- **Efficiency**: The `StructuredOutputManager` generates a **Bitmask** for each decoding step.
- **Enforcement**: This bitmask is used to filter the model's logits before sampling, effectively making the probability of non-compliant tokens zero.
- **Integration**: The bitmask generation is optimized to work seamlessly with our high-speed Triton-based sampling paths.

## 🛠 Usage Example

### JSON Schema Enforcement
Ensure the model always outputs a valid JSON object matching your schema.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Define JSON schema
my_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
}

# Request structured output
params = SamplingParams(temperature=0, max_tokens=50)
outputs = llm.generate(
    "Generate a person record:",
    sampling_params=params,
    extra_kwargs={"json_schema": my_schema}
)

print(outputs[0].outputs[0].text) # Guaranteed to be valid JSON
```

### Regular Expression Constraint
Force the model to output data in a specific format (e.g., an IP address).

```python
outputs = llm.generate(
    "The server IP is:",
    extra_kwargs={"regex": r"(\d{1,3}\.){3}\d{1,3}"}
)
```

## 📊 Comparison with vLLM standard

| Feature | Standard vLLM | FastInference (vLLM Lite) |
| :--- | :--- | :--- |
| **Logic** | Distributed Aware | **Simplified & Decoupled** |
| **Backend** | Multiple | **Outlines (Optimized)** |
| **Reliability** | 100% | **100%** |
| **Overhead** | Medium | **Low (Cached Grammars)** |
