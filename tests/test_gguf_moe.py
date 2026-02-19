import time
from vllm import LLM, SamplingParams

def test_gguf_moe():
    model_path = "models/Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf"
    tokenizer_path = "models/Qwen1.5-MoE-A2.7B-Chat"
    
    print(f"--- Testing GGUF MoE Model: {model_path} ---")
    
    prompts = [
        "What are the benefits of Mixture of Experts (MoE)?",
        "How does PagedAttention improve throughput?"
    ]
    
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)

    # Load Model
    start_load = time.time()
    # Specify quantization="gguf" explicitly if needed, but vLLM usually auto-detects from filename
    llm = LLM(
        model=model_path, 
        tokenizer=tokenizer_path,
        quantization="gguf",
        dtype="float16", # GGUF internally handles its own types, but this is the compute dtype
        enforce_eager=True
    )
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")

    # Generate
    print("Generating...")
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for output in outputs:
        print(f"\nPrompt: {output.prompt}")
        print(f"Response: {output.outputs[0].text[:200]}...")
    
    print("\n--- GGUF MoE Test Complete ---")

if __name__ == "__main__":
    test_gguf_moe()
