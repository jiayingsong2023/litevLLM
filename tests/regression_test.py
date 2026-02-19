from vllm import LLM, SamplingParams
import time

def test_tinyllama():
    print("--- Testing TinyLlama-1.1B (Standard HF Model) ---")
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)

    # Load Model
    llm = LLM(model="models/TinyLlama-1.1B-Chat-v1.0")

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    print("--- TinyLlama Test Passed ---\n")

if __name__ == "__main__":
    test_tinyllama()
