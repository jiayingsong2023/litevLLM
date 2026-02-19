import time
from vllm import LLM, SamplingParams

def benchmark_moe():
    model_path = "models/Qwen1.5-MoE-A2.7B-Chat"
    print(f"--- Benchmarking MoE Model (Eager Mode): {model_path} ---")
    
    prompts = [
        "What are the benefits of Mixture of Experts (MoE) in LLMs?",
        "Explain how PagedAttention works in 3 bullet points.",
        "Write a short poem about fast inference.",
        "How can I optimize vLLM for high throughput?"
    ]
    
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)

    # Measure load time
    start_load = time.time()
    # Use enforce_eager=True to avoid CUDA graph capture errors
    llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16", enforce_eager=True)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")

    # Warmup
    print("Warming up...")
    llm.generate(["Warmup prompt"], sampling_params)

    # Benchmark generation
    start_gen = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - start_gen

    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = total_tokens / gen_time

    print(f"\n--- Results ---")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    
    for output in outputs:
        print(f"\nPrompt: {output.prompt}")
        print(f"Response: {output.outputs[0].text[:100]}...")

if __name__ == "__main__":
    benchmark_moe()
