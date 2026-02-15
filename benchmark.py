import time
import psutil
import os
from llama_cpp import Llama

def benchmark():
    model_path = "llama-2-7b-chat.Q4_K_M.gguf"
    
    print(f"Loading model: {model_path}...")
    start_load = time.time()
    # n_gpu_layers=-1 means offload all layers to GPU
    llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1) 
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")

    prompt = "Q: What is the capital of France? A:"
    max_tokens = 128

    print("Warmup...")
    llm(prompt, max_tokens=1)

    print(f"Benchmarking (max_tokens={max_tokens})...")
    start_gen = time.time()
    output = llm(prompt, max_tokens=max_tokens, echo=False)
    gen_time = time.time() - start_gen
    
    # Calculate tokens
    text = output["choices"][0]["text"]
    # Rough estimate of tokens (usually llama-cpp provides it in the output)
    tokens_generated = output["usage"]["completion_tokens"]
    
    tps = tokens_generated / gen_time
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss / (1024 * 1024 * 1024) # GB

    print("\n--- Benchmark Result ---")
    print(f"Model: {model_path}")
    print(f"Tokens Generated: {tokens_generated}")
    print(f"Time Taken: {gen_time:.2f}s")
    print(f"Generation Speed: {tps:.2f} tokens/sec")
    print(f"Memory Usage: {mem_info:.2f} GB")
    print(f"Response: {text}")
    print("------------------------")

if __name__ == "__main__":
    benchmark()
